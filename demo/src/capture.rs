use std::env;
use std::io::{self, Write, stdout};
use std::path::PathBuf;
use std::sync::{
    Arc, Mutex, MutexGuard, Once,
    atomic::{AtomicBool, Ordering},
};
use std::thread::{self, JoinHandle, sleep};
use std::time::Duration;

use anyhow::Result;
use crossbeam_channel::{Receiver, Sender, unbounded};
use df::{Complex32, tract::*};
use ndarray::prelude::*;
use pipewire as pw;
use pw::properties::properties;
use pw::spa;
use ringbuf::traits::*;
use ringbuf::{HeapCons, HeapProd, HeapRb};
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};
use spa::pod::Pod;

pub type RbProd = HeapProd<f32>;
pub type RbCons = HeapCons<f32>;
pub type SendLsnr = Sender<f32>;
pub type RecvLsnr = Receiver<f32>;
pub type SendSpec = Sender<Box<[f32]>>;
pub type RecvSpec = Receiver<Box<[f32]>>;
pub type SendControl = Sender<(DfControl, f32)>;
pub type RecvControl = Receiver<(DfControl, f32)>;

pub(crate) static INIT_LOGGER: Once = Once::new();
pub(crate) static MODEL_PATH: Mutex<Option<PathBuf>> = Mutex::new(None);
struct GlobalModel(Mutex<Option<DfTract>>);
// SAFETY: `DfTract` is not `Send`/`Sync` on its own (libDF itself relies on it
// being transferable across threads, see `unsafe impl Send for SendError`).
// Here the model is initialized once on the main thread *before* the worker
// thread is spawned, every access is serialized through the Mutex, and the
// worker only ever touches its own `clone()`. Nothing accesses the stored
// instance concurrently.
unsafe impl Sync for GlobalModel {}
static MODEL: GlobalModel = GlobalModel(Mutex::new(None));

fn model() -> MutexGuard<'static, Option<DfTract>> {
    MODEL.0.lock().expect("DF model mutex poisoned")
}

pub struct AudioSink {
    sr: u32,
    channels: u32,
    device_str: Option<String>,
    stop_sender: Option<pw::channel::Sender<()>>,
    thread_handle: Option<JoinHandle<()>>,
}

pub struct AudioSource {
    sr: u32,
    channels: u32,
    device_str: Option<String>,
    stop_sender: Option<pw::channel::Sender<()>>,
    thread_handle: Option<JoinHandle<()>>,
}

#[derive(PartialEq)]
pub enum DfControl {
    AttenLim,
    PostFilterBeta,
    MinThreshDb,
    MaxErbThreshDb,
    MaxDfThreshDb,
}

/// Initialize DF model and returns sample rate, frame size, and number of frequency bins
fn init_df(model_path: Option<PathBuf>, channels: usize) -> (usize, usize, usize) {
    if let Some(m) = model().as_ref() {
        if m.ch == channels {
            return (m.sr, m.hop_size, m.n_freqs);
        }
    }
    let df_params = if let Some(path) = model_path {
        DfParams::new(path).expect("Failed to read DF model")
    } else {
        DfParams::default()
    };
    let r_params = RuntimeParams::default_with_ch(channels);
    let df = DfTract::new(df_params, &r_params).expect("Could not initialize DeepFilter runtime");
    let (sr, frame_size, freq_size) = (df.sr, df.hop_size, df.n_freqs);
    *model() = Some(df);
    (sr, frame_size, freq_size)
}

impl AudioSink {
    fn new(sample_rate: u32, device_str: Option<String>) -> Result<Self> {
        Ok(Self {
            sr: sample_rate,
            channels: 1,
            device_str,
            stop_sender: None,
            thread_handle: None,
        })
    }

    fn start(&mut self, mut rb: RbCons) -> Result<()> {
        pw::init();
        let sr = self.sr;
        let channels = self.channels;
        let device_str = self.device_str.clone();

        let (tx_stop, rx_stop) = pw::channel::channel::<()>();

        let handle = thread::spawn(move || {
            let mainloop =
                pw::main_loop::MainLoopRc::new(None).expect("Failed to create PipeWire main loop");
            let context = pw::context::ContextRc::new(&mainloop, None)
                .expect("Failed to create PipeWire context");
            let core = context.connect_rc(None).expect("Failed to connect to PipeWire core");

            let mainloop_clone = mainloop.clone();
            let _stop_listener = rx_stop.attach(&mainloop.loop_(), move |_| {
                mainloop_clone.quit();
            });

            let mut props = properties! {
                *pw::keys::MEDIA_TYPE => "Audio",
                *pw::keys::MEDIA_CATEGORY => "Playback",
                *pw::keys::MEDIA_ROLE => "Music",
            };
            if let Some(ref dev) = device_str {
                props.insert("target.object", dev.as_str());
            }

            let stream = pw::stream::StreamBox::new(&core, "deepfilternet-playback", props)
                .expect("Failed to create PipeWire playback stream");

            let _listener = stream
                .add_local_listener::<()>()
                .process(move |stream, _data| {
                    if let Some(mut buffer) = stream.dequeue_buffer() {
                        let datas = buffer.datas_mut();
                        if datas.is_empty() {
                            return;
                        }
                        let data = &mut datas[0];
                        if let Some(slice) = data.data() {
                            let max_size = slice.len();
                            let max_samples = max_size / std::mem::size_of::<f32>();
                            let max_frames = max_samples / channels as usize;
                            if max_frames == 0 {
                                return;
                            }

                            let pcm_bytes = &mut slice
                                [..max_frames * channels as usize * std::mem::size_of::<f32>()];
                            let samples: &mut [f32] = unsafe {
                                std::slice::from_raw_parts_mut(
                                    pcm_bytes.as_mut_ptr() as *mut f32,
                                    max_frames * channels as usize,
                                )
                            };

                            let mut n = 0;
                            if channels > 1 {
                                let mut data_it = samples.chunks_mut(channels as usize);
                                while n < max_frames {
                                    for (i, o) in rb.pop_iter().zip(&mut data_it) {
                                        o.fill(i);
                                        n += 1;
                                    }
                                }
                            } else {
                                while n < max_frames {
                                    let popped = rb.pop_slice(&mut samples[n..max_frames]);
                                    if popped == 0 {
                                        samples[n..max_frames].fill(0.0);
                                        break;
                                    }
                                    n += popped;
                                }
                            }

                            let bytes_len =
                                max_frames * channels as usize * std::mem::size_of::<f32>();
                            let raw = data.chunk_mut() as *mut _ as *mut spa::sys::spa_chunk;
                            unsafe {
                                (*raw).offset = 0;
                                (*raw).size = bytes_len as u32;
                                (*raw).stride =
                                    (channels as usize * std::mem::size_of::<f32>()) as i32;
                            }
                        }
                    }
                })
                .register()
                .expect("Failed to register listener");

            let mut audio_info = spa::param::audio::AudioInfoRaw::new();
            audio_info.set_format(spa::param::audio::AudioFormat::F32LE);
            audio_info.set_channels(channels);
            audio_info.set_rate(sr);

            let obj = pw::spa::pod::Object {
                type_: pw::spa::utils::SpaTypes::ObjectParamFormat.as_raw(),
                id: pw::spa::param::ParamType::EnumFormat.as_raw(),
                properties: audio_info.into(),
            };
            let values: Vec<u8> = pw::spa::pod::serialize::PodSerializer::serialize(
                std::io::Cursor::new(Vec::new()),
                &pw::spa::pod::Value::Object(obj),
            )
            .expect("Failed to serialize SPA pod")
            .0
            .into_inner();

            let mut params = [Pod::from_bytes(&values).expect("Failed to parse Pod from bytes")];

            stream
                .connect(
                    spa::utils::Direction::Output,
                    None,
                    pw::stream::StreamFlags::AUTOCONNECT
                        | pw::stream::StreamFlags::MAP_BUFFERS
                        | pw::stream::StreamFlags::RT_PROCESS,
                    &mut params,
                )
                .expect("Failed to connect PipeWire playback stream");

            log::info!("Starting playback stream on PipeWire");
            mainloop.run();
        });

        self.stop_sender = Some(tx_stop);
        self.thread_handle = Some(handle);
        Ok(())
    }

    fn sr(&self) -> u32 {
        self.sr
    }

    fn pause(&mut self) -> Result<()> {
        if let Some(tx) = self.stop_sender.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
        Ok(())
    }
}

impl AudioSource {
    fn new(sample_rate: u32, device_str: Option<String>) -> Result<Self> {
        Ok(Self {
            sr: sample_rate,
            channels: 1,
            device_str,
            stop_sender: None,
            thread_handle: None,
        })
    }

    fn start(&mut self, mut rb: RbProd) -> Result<()> {
        pw::init();
        let sr = self.sr;
        let channels = self.channels;
        let device_str = self.device_str.clone();

        let (tx_stop, rx_stop) = pw::channel::channel::<()>();

        let handle = thread::spawn(move || {
            let mainloop =
                pw::main_loop::MainLoopRc::new(None).expect("Failed to create PipeWire main loop");
            let context = pw::context::ContextRc::new(&mainloop, None)
                .expect("Failed to create PipeWire context");
            let core = context.connect_rc(None).expect("Failed to connect to PipeWire core");

            let mainloop_clone = mainloop.clone();
            let _stop_listener = rx_stop.attach(&mainloop.loop_(), move |_| {
                mainloop_clone.quit();
            });

            let mut props = properties! {
                *pw::keys::MEDIA_TYPE => "Audio",
                *pw::keys::MEDIA_CATEGORY => "Capture",
                *pw::keys::MEDIA_ROLE => "Music",
            };
            if let Some(ref dev) = device_str {
                props.insert("target.object", dev.as_str());
            }

            let stream = pw::stream::StreamBox::new(&core, "deepfilternet-capture", props)
                .expect("Failed to create PipeWire capture stream");

            let _listener = stream
                .add_local_listener::<()>()
                .process(move |stream, _data| {
                    if let Some(mut buffer) = stream.dequeue_buffer() {
                        let datas = buffer.datas_mut();
                        if datas.is_empty() {
                            return;
                        }
                        let data = &mut datas[0];
                        let chunk = data.chunk();
                        let offset = chunk.offset() as usize;
                        let size = chunk.size() as usize;
                        if let Some(slice) = data.data() {
                            if offset + size <= slice.len() {
                                let pcm_bytes = &slice[offset..offset + size];
                                let samples: &[f32] = unsafe {
                                    std::slice::from_raw_parts(
                                        pcm_bytes.as_ptr() as *const f32,
                                        pcm_bytes.len() / std::mem::size_of::<f32>(),
                                    )
                                };
                                let len = samples.len() / channels as usize;
                                let mut n = 0;
                                if channels > 1 {
                                    let mut iter = samples.chunks(channels as usize).map(df::mean);
                                    while n < len {
                                        n += rb.push_iter(&mut iter);
                                    }
                                } else {
                                    while n < len {
                                        n += rb.push_slice(&samples[n..]);
                                    }
                                }
                            }
                        }
                    }
                })
                .register()
                .expect("Failed to register listener");

            let mut audio_info = spa::param::audio::AudioInfoRaw::new();
            audio_info.set_format(spa::param::audio::AudioFormat::F32LE);
            audio_info.set_channels(channels);
            audio_info.set_rate(sr);

            let obj = pw::spa::pod::Object {
                type_: pw::spa::utils::SpaTypes::ObjectParamFormat.as_raw(),
                id: pw::spa::param::ParamType::EnumFormat.as_raw(),
                properties: audio_info.into(),
            };
            let values: Vec<u8> = pw::spa::pod::serialize::PodSerializer::serialize(
                std::io::Cursor::new(Vec::new()),
                &pw::spa::pod::Value::Object(obj),
            )
            .expect("Failed to serialize SPA pod")
            .0
            .into_inner();

            let mut params = [Pod::from_bytes(&values).expect("Failed to parse Pod from bytes")];

            stream
                .connect(
                    spa::utils::Direction::Input,
                    None,
                    pw::stream::StreamFlags::AUTOCONNECT
                        | pw::stream::StreamFlags::MAP_BUFFERS
                        | pw::stream::StreamFlags::RT_PROCESS,
                    &mut params,
                )
                .expect("Failed to connect PipeWire capture stream");

            log::info!("Starting capture stream on PipeWire");
            mainloop.run();
        });

        self.stop_sender = Some(tx_stop);
        self.thread_handle = Some(handle);
        Ok(())
    }

    fn sr(&self) -> u32 {
        self.sr
    }

    fn pause(&mut self) -> Result<()> {
        if let Some(tx) = self.stop_sender.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
        Ok(())
    }
}

pub(crate) struct AtomicControls {
    has_init: Arc<AtomicBool>,
    should_stop: Arc<AtomicBool>,
}
impl AtomicControls {
    pub fn into_inner(self) -> (Arc<AtomicBool>, Arc<AtomicBool>) {
        (self.has_init, self.should_stop)
    }
}
pub(crate) struct GuiCom {
    pub s_lsnr: Option<SendLsnr>,
    pub s_spec: Option<(SendSpec, SendSpec)>,
    pub r_opt: Option<RecvControl>,
}
impl GuiCom {
    pub fn into_inner(
        self,
    ) -> (
        Option<SendLsnr>,
        Option<(SendSpec, SendSpec)>,
        Option<RecvControl>,
    ) {
        (self.s_lsnr, self.s_spec, self.r_opt)
    }
}

fn get_worker_fn(
    mut rb_in: RbCons,
    mut rb_out: RbProd,
    input_sr: usize,
    output_sr: usize,
    controls: AtomicControls,
    df_com: Option<GuiCom>,
) -> impl FnMut() {
    let (has_init, should_stop) = controls.into_inner();
    let (mut s_lsnr, mut s_spec, mut r_opt) = if let Some(df_com) = df_com {
        df_com.into_inner()
    } else {
        (None, None, None)
    };

    move || {
        let mut df = model().as_ref().expect("DF model not initialized").clone();
        debug_assert_eq!(df.ch, 1); // Processing for more channels are not implemented yet
        let mut inframe = Array2::zeros((df.ch, df.hop_size));
        let mut outframe = inframe.clone();
        df.process(inframe.view(), outframe.view_mut())
            .expect("Failed to run DeepFilterNet");
        has_init.store(true, Ordering::Relaxed);
        log::info!("Worker init");

        // Input resampler: device sr -> df.sr (flat mono buffer)
        let (mut input_resampler, n_in) = if input_sr != df.sr {
            let r = Fft::<f32>::new(input_sr, df.sr, df.hop_size, 1, FixedSync::Output)
                .expect("Failed to init input resampler");
            let n_in = r.input_frames_max();
            (Some((r, vec![0.0f32; n_in])), n_in)
        } else {
            (None, df.hop_size)
        };

        // Output resampler: df.sr -> device sr
        let (mut output_resampler, n_out) = if output_sr != df.sr {
            let r = Fft::<f32>::new(df.sr, output_sr, df.hop_size, 1, FixedSync::Input)
                .expect("Failed to init output resampler");
            let n_out = r.output_frames_max();
            (Some((r, vec![0.0f32; n_out])), n_out)
        } else {
            (None, df.hop_size)
        };

        while !should_stop.load(Ordering::Relaxed) {
            if rb_in.occupied_len() < n_in {
                // Sleep for half a hop size
                sleep(Duration::from_secs_f32(
                    df.hop_size as f32 / df.sr as f32 / 2.,
                ));
                continue;
            }
            if let Some((ref mut r, ref mut buf)) = input_resampler.as_mut() {
                let need = r.input_frames_next(); // varies by ±1 frame with FixedSync::Output
                let n = rb_in.pop_slice(&mut buf[..need]);
                debug_assert_eq!(n, need);
                let in_adapter = InterleavedSlice::new(&buf[..need], 1, need).unwrap();
                let mut out_adapter =
                    InterleavedSlice::new_mut(inframe.as_slice_mut().unwrap(), 1, df.hop_size)
                        .unwrap();
                let (_read, written) =
                    r.process_into_buffer(&in_adapter, &mut out_adapter, None).unwrap();
                debug_assert_eq!(written, df.hop_size);
            } else {
                let n = rb_in.pop_slice(inframe.as_slice_mut().unwrap());
                debug_assert_eq!(n, n_in);
            }
            let lsnr = df
                .process(inframe.view(), outframe.view_mut())
                .expect("Failed to run DeepFilterNet");
            let mut n = 0;
            if let Some((ref mut r, ref mut buf)) = output_resampler.as_mut() {
                let in_adapter =
                    InterleavedSlice::new(outframe.as_slice().unwrap(), 1, df.hop_size).unwrap();
                let mut out_adapter = InterleavedSlice::new_mut(&mut buf[..], 1, n_out).unwrap();
                let (_read, written) =
                    r.process_into_buffer(&in_adapter, &mut out_adapter, None).unwrap();
                while n < written {
                    n += rb_out.push_slice(&buf[n..written]);
                }
            } else {
                let buf = outframe.as_slice().unwrap();
                while n < n_out {
                    n += rb_out.push_slice(&buf[n..]);
                }
                debug_assert_eq!(n, n_out);
            }
            if let Some(ref mut s_lsnr) = s_lsnr.as_mut() {
                s_lsnr.send(lsnr).expect("Failed to send to LSNR rb");
            }
            if let Some((ref mut s_noisy, ref mut s_enh)) = s_spec.as_mut() {
                push_spec(df.get_spec_noisy(), s_noisy);
                push_spec(df.get_spec_enh(), s_enh);
            }
            if let Some(ref mut r_opt) = r_opt.as_mut() {
                while let Ok((c, v)) = r_opt.try_recv() {
                    match c {
                        DfControl::AttenLim => df.set_atten_lim(v),
                        DfControl::PostFilterBeta => df.set_pf_beta(v),
                        DfControl::MinThreshDb => df.min_db_thresh = v,
                        DfControl::MaxErbThreshDb => df.max_db_erb_thresh = v,
                        DfControl::MaxDfThreshDb => df.max_db_df_thresh = v,
                    }
                }
            }
        }
    }
}

fn push_spec(spec: ArrayView2<Complex32>, sender: &SendSpec) {
    debug_assert_eq!(spec.len_of(Axis(0)), 1); // only single channel for now
    let out = spec.iter().map(|x| x.norm_sqr().max(1e-10).log10() * 10.).collect::<Vec<f32>>();
    sender.send(out.into_boxed_slice()).expect("Failed to send spectrogram")
}

pub fn log_format(buf: &mut env_logger::fmt::Formatter, record: &log::Record) -> io::Result<()> {
    let ts = buf.timestamp_millis();
    let module = record.module_path().unwrap_or("").to_string();
    let level_style = buf.default_level_style(log::Level::Info);

    writeln!(
        buf,
        "{} | {} | {} {}",
        ts,
        level_style.value(record.level()),
        module,
        record.args()
    )
}

pub struct DeepFilterCapture {
    pub sr: usize,
    pub frame_size: usize,
    pub freq_size: usize,
    should_stop: Arc<AtomicBool>,
    worker_handle: Option<JoinHandle<()>>,
    source: AudioSource,
    sink: AudioSink,
}

impl Default for DeepFilterCapture {
    fn default() -> Self {
        DeepFilterCapture::new(None, None, None, None, None)
            .expect("Error during DeepFilterCapture initialization")
    }
}
impl DeepFilterCapture {
    pub fn new(
        model_path: Option<PathBuf>,
        s_lsnr: Option<SendLsnr>,
        s_noisy: Option<SendSpec>,
        s_enh: Option<SendSpec>,
        r_opt: Option<RecvControl>,
    ) -> Result<Self> {
        let ch = 1;
        let (sr, frame_size, freq_size) = init_df(model_path, ch);
        let in_rb = HeapRb::<f32>::new(frame_size * 100);
        let out_rb = HeapRb::<f32>::new(frame_size * 100);
        let (in_prod, in_cons) = in_rb.split();
        let (out_prod, out_cons) = out_rb.split();

        let mut source = AudioSource::new(sr as u32, None)?;
        let mut sink = AudioSink::new(sr as u32, None)?;
        let should_stop = Arc::new(AtomicBool::new(false));
        let has_init = Arc::new(AtomicBool::new(false));
        let s_spec = match (s_noisy, s_enh) {
            (Some(n), Some(e)) => Some((n, e)),
            _ => None,
        };
        let controls = AtomicControls {
            has_init: has_init.clone(),
            should_stop: should_stop.clone(),
        };
        let df_com = GuiCom {
            s_lsnr,
            s_spec,
            r_opt,
        };
        let worker_handle = Some(thread::spawn(get_worker_fn(
            in_cons,
            out_prod,
            source.sr() as usize,
            sink.sr() as usize,
            controls,
            Some(df_com),
        )));
        while !has_init.load(Ordering::Relaxed) {
            sleep(Duration::from_secs_f32(0.01));
        }
        log::info!("DeepFilter Capture init");
        source.start(in_prod)?;
        sink.start(out_cons)?;

        Ok(Self {
            sr,
            frame_size,
            freq_size,
            should_stop,
            worker_handle,
            source,
            sink,
        })
    }

    pub fn should_stop(&mut self) -> Result<()> {
        self.sink.pause()?;
        self.source.pause()?;
        if let Some(h) = self.worker_handle.take() {
            log::info!("Joining DF Worker");
            self.should_stop.swap(true, Ordering::Relaxed);
            h.join().expect("Error during DF worker join");
        }
        Ok(())
    }
}

#[allow(unused)]
#[allow(unknown_lints)]
#[allow(clippy::assigning_clones)]
pub fn main() -> Result<()> {
    INIT_LOGGER.call_once(|| {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
            .filter_module("tract_onnx", log::LevelFilter::Error)
            .filter_module("tract_core", log::LevelFilter::Error)
            .filter_module("tract_hir", log::LevelFilter::Error)
            .filter_module("tract_linalg", log::LevelFilter::Error)
            .format(log_format)
            .init();
    });

    let (lsnr_prod, mut lsnr_cons) = unbounded();
    let mut model_path = env::var("DF_MODEL").ok().map(PathBuf::from);
    if model_path.is_none() {
        model_path = MODEL_PATH.lock().unwrap().clone();
    }
    if let Some(p) = model_path.as_ref() {
        log::info!("Running with model '{:?}'", p);
    }
    let _c = DeepFilterCapture::new(model_path, Some(lsnr_prod), None, None, None);

    loop {
        sleep(Duration::from_millis(200));
        while let Ok(lsnr) = lsnr_cons.try_recv() {
            print!("\rCurrent SNR: {:>5.1} dB\x1b[K", lsnr);
        }
        stdout().flush().unwrap();
    }
}
