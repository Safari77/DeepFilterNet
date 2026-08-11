use std::env;
use std::future::Future;
use std::path::PathBuf;
use std::process::exit;
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::sleep;
use std::time::Duration;

use clap::{Parser, ValueHint};
use crossbeam_channel::unbounded;
use iced::widget::{self, Container, Image, column, container, image, row, slider, text};
use iced::{Alignment, ContentFit, Element, Length, Subscription, Task, alignment};
use image_rs::{Rgba, RgbaImage, imageops};

mod capture;
mod cmap;
use capture::*;

/// Simple program to sample from a hd5 dataset directory
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to model tar.gz
    #[arg(short, long, value_hint = ValueHint::FilePath)]
    model: Option<PathBuf>,
    /// Logging verbosity
    #[arg(
        long,
        short = 'v',
        action = clap::ArgAction::Count,
        global = true,
        help = "Increase logging verbosity with multiple `-vv`",
    )]
    verbose: u8,
}

pub fn main() -> iced::Result {
    let args = Args::parse();
    let level = match args.verbose {
        0 => log::LevelFilter::Warn,
        1 => log::LevelFilter::Info,
        2 => log::LevelFilter::Debug,
        _ => log::LevelFilter::Trace,
    };
    let tract_level = match args.verbose {
        0..=3 => log::LevelFilter::Error,
        4 => log::LevelFilter::Info,
        5 => log::LevelFilter::Debug,
        _ => log::LevelFilter::Trace,
    };
    if let Some(model) = args.model {
        *MODEL_PATH.lock().expect("Failed to lock MODEL_PATH") = Some(model);
    }

    capture::INIT_LOGGER.call_once(|| {
        env_logger::Builder::from_env(env_logger::Env::default())
            .filter_level(level)
            .filter_module("tract_onnx", tract_level)
            .filter_module("tract_hir", tract_level)
            .filter_module("tract_core", tract_level)
            .filter_module("tract_linalg", tract_level)
            .filter_module("iced_winit", log::LevelFilter::Error)
            .filter_module("iced_wgpu", log::LevelFilter::Error)
            .filter_module("wgpu_core", log::LevelFilter::Error)
            .filter_module("wgpu_hal", log::LevelFilter::Error)
            .filter_module("naga", log::LevelFilter::Error)
            .filter_module("crossfont", log::LevelFilter::Error)
            .filter_module("cosmic_text", log::LevelFilter::Error)
            .format(capture::log_format)
            .init();
    });

    iced::application(SpecView::new, SpecView::update, SpecView::view)
        .title("DeepFilterNet Demo")
        .subscription(SpecView::subscription)
        .run()
}

static SPEC_NOISY: OnceLock<Arc<Mutex<SpecImage>>> = OnceLock::new();
static SPEC_ENH: OnceLock<Arc<Mutex<SpecImage>>> = OnceLock::new();

fn spec_noisy() -> Arc<Mutex<SpecImage>> {
    SPEC_NOISY.get().expect("SPEC_NOISY not initialized").clone()
}

fn spec_enh() -> Arc<Mutex<SpecImage>> {
    SPEC_ENH.get().expect("SPEC_ENH not initialized").clone()
}

struct SpecView {
    df_worker: DeepFilterCapture,
    lsnr: f32,
    atten_lim: f32,
    post_filter_beta: f32,
    min_threshdb: f32,
    max_erbthreshdb: f32,
    max_dfthreshdb: f32,
    noisy_img: image::Handle,
    enh_img: image::Handle,
    r_lsnr: RecvLsnr,
    r_noisy: RecvSpec,
    r_enh: RecvSpec,
    s_controls: SendControl,
}

#[derive(Debug, Clone, Copy)]
pub enum Message {
    None,
    Tick,
    LsnrChanged(f32),
    NoisyChanged,
    EnhChanged,
    AttenLimChanged(f32),
    PostFilterChanged(f32),
    MinThreshDbChanged(f32),
    MaxErbThreshDbChanged(f32),
    MaxDfThreshDbChanged(f32),
    Exit,
}

struct SpecImage {
    im: RgbaImage,
    n_frames: u32,
    n_freqs: u32,
    vmin: f32,
    vmax: f32,
}

impl SpecImage {
    fn new(n_frames: u32, n_freqs: u32, vmin: f32, vmax: f32) -> Self {
        Self {
            // Store image transposed so we can iterate over rows quickly
            im: RgbaImage::new(n_freqs, n_frames),
            n_frames,
            n_freqs,
            vmin,
            vmax,
        }
    }
    fn w(&self) -> usize {
        self.n_frames as usize
    }
    fn h(&self) -> usize {
        self.n_freqs as usize
    }
    fn update<I>(&mut self, specs: I, mut n_specs: usize)
    where
        I: Iterator<Item = Box<[f32]>>,
    {
        if n_specs == 0 {
            return;
        }
        if n_specs >= self.n_frames as usize {
            // Just drop a few
            n_specs = self.n_frames as usize - 1;
        }
        for (spec, im_row) in specs.take(n_specs).zip(self.im.rows_mut()) {
            for (s, x) in spec.iter().zip(im_row) {
                // clamp and normalize
                let v = (s.min(self.vmax).max(self.vmin) - self.vmin) / (self.vmax - self.vmin);
                *x = Rgba(cmap::CMAP_INFERNO[(v * 255.) as usize]);
            }
        }
        let (w, h) = (self.w(), self.h());
        self.im.rotate_left((w - n_specs) * 4 * h);
    }
    fn image_handle(&self) -> image::Handle {
        let imt_buf = imageops::rotate270(&self.im).as_raw().to_vec();
        image::Handle::from_rgba(self.n_frames, self.n_freqs, imt_buf)
    }
}

impl SpecView {
    fn new() -> (Self, Task<Message>) {
        let (s_lsnr, r_lsnr) = unbounded();
        let (s_noisy, r_noisy) = unbounded();
        let (s_enh, r_enh) = unbounded();
        let (s_controls, r_controls) = unbounded();

        let model_path = env::var("DF_MODEL").ok().map(PathBuf::from);
        let df_worker = DeepFilterCapture::new(
            model_path,
            Some(s_lsnr),
            Some(s_noisy),
            Some(s_enh),
            Some(r_controls),
        )
        .expect("Failed to initialize DeepFilterNet audio capturing");

        let w = (df_worker.sr / df_worker.frame_size * 10) as u32;
        let freq_res = df_worker.sr / 2 / (df_worker.freq_size - 1);
        let h = (8000 / freq_res) as u32;
        let noisy = Arc::new(Mutex::new(SpecImage::new(w, h, -100., -10.)));
        let enh = Arc::new(Mutex::new(SpecImage::new(w, h, -100., -10.)));
        let (noisy_img, enh_img) = (
            noisy.lock().unwrap().image_handle(),
            enh.lock().unwrap().image_handle(),
        );
        SPEC_NOISY
            .set(noisy)
            .unwrap_or_else(|_| panic!("SPEC_NOISY already initialized"));
        SPEC_ENH.set(enh).unwrap_or_else(|_| panic!("SPEC_ENH already initialized"));
        (
            Self {
                df_worker,
                lsnr: 0.,
                atten_lim: 100.,
                post_filter_beta: 0.,
                min_threshdb: -15.,
                max_erbthreshdb: 35.,
                max_dfthreshdb: 35.,
                r_lsnr,
                r_noisy,
                r_enh,
                s_controls,
                noisy_img,
                enh_img,
            },
            Task::none(),
        )
    }

    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::None => (),
            Message::Exit => {
                self.df_worker.should_stop().expect("Failed to stop DF worker");
                exit(0);
            }
            Message::Tick => {
                let mut commands = Vec::new();
                if let Some(task) = self.update_lsnr() {
                    commands.push(Task::perform(task, move |message| message))
                }
                if let Some(task) = self.update_noisy() {
                    commands.push(Task::perform(task, move |message| message))
                }
                if let Some(task) = self.update_enh() {
                    commands.push(Task::perform(task, move |message| message))
                }
                return Task::batch(commands);
            }
            Message::LsnrChanged(lsnr) => self.lsnr = lsnr,
            Message::NoisyChanged => {
                self.noisy_img =
                    spec_noisy().lock().expect("Failed to lock SPEC_NOISY").image_handle();
            }
            Message::EnhChanged => {
                self.enh_img = spec_enh().lock().expect("Failed to lock SPEC_ENH").image_handle();
            }
            Message::AttenLimChanged(v) => {
                self.atten_lim = v;
                self.s_controls
                    .send((DfControl::AttenLim, v))
                    .expect("Failed to send DfControl")
            }
            Message::PostFilterChanged(v) => {
                self.post_filter_beta = v;
                self.s_controls
                    .send((DfControl::PostFilterBeta, v))
                    .expect("Failed to send DfControl")
            }
            Message::MinThreshDbChanged(v) => {
                self.min_threshdb = v;
                self.s_controls
                    .send((DfControl::MinThreshDb, v))
                    .expect("Failed to send DfControl")
            }
            Message::MaxErbThreshDbChanged(v) => {
                self.max_erbthreshdb = v;
                self.s_controls
                    .send((DfControl::MaxErbThreshDb, v))
                    .expect("Failed to send DfControl")
            }
            Message::MaxDfThreshDbChanged(v) => {
                self.max_dfthreshdb = v;
                self.s_controls
                    .send((DfControl::MaxDfThreshDb, v))
                    .expect("Failed to send DfControl")
            }
        }
        Task::none()
    }

    fn view(&self) -> Element<'_, Message> {
        let content = column![
            row![
                text("DeepFilterNet Demo").size(40).width(Length::Fill),
                button("exit").on_press(Message::Exit)
            ]
            .width(1000.0),
        ];
        #[cfg(feature = "thresholds")]
        let content = {
            content
                .push(slider_view(
                    "Threshold Min [dB]",
                    self.min_threshdb,
                    -15.,
                    35.,
                    Message::MinThreshDbChanged,
                    1000.0,
                    0,
                    3.,
                ))
                .push(slider_view(
                    "Threshold ERB Max [dB]",
                    self.max_erbthreshdb,
                    -15.,
                    35.,
                    Message::MaxErbThreshDbChanged,
                    1000.0,
                    0,
                    3.,
                ))
                .push(slider_view(
                    "Threshold DF Max [dB]",
                    self.max_dfthreshdb,
                    -15.,
                    35.,
                    Message::MaxDfThreshDbChanged,
                    1000.0,
                    0,
                    3.,
                ))
        };
        let content = content
            .push(slider_view(
                "Noise Attenuation [dB]",
                self.atten_lim,
                0.,
                100.,
                Message::AttenLimChanged,
                1000.0,
                0,
                3.,
            ))
            .push(slider_view(
                "Post Filter Beta",
                self.post_filter_beta,
                0.,
                1.,
                Message::PostFilterChanged,
                1000.0,
                3,
                0.001,
            ))
            .push(self.specs())
            .push(
                row![
                    text("Current SNR:").size(18),
                    text(format!("{:>5.1} dB", self.lsnr))
                        .size(18)
                        .width(80.0)
                        .align_x(alignment::Horizontal::Right)
                        .align_y(alignment::Vertical::Top),
                ]
                .spacing(20.0)
                .align_y(Alignment::End),
            );

        container(content)
            .padding(50.0)
            .width(Length::Fill)
            .height(Length::Fill)
            .align_x(alignment::Horizontal::Center)
            .align_y(alignment::Vertical::Center)
            .into()
    }

    fn subscription(&self) -> Subscription<Message> {
        iced::time::every(std::time::Duration::from_millis(20)).map(|_| Message::Tick)
    }

    fn update_lsnr(&mut self) -> Option<impl Future<Output = Message>> {
        if self.r_lsnr.is_empty() {
            return None;
        }
        let recv = self.r_lsnr.clone();
        Some(async move {
            sleep(Duration::from_millis(100));
            let mut lsnr = 0.;
            let mut n = 0;
            while let Ok(v) = recv.try_recv() {
                lsnr += v;
                n += 1;
            }
            if n > 0 {
                lsnr /= n as f32;
                Message::LsnrChanged(lsnr)
            } else {
                Message::None
            }
        })
    }

    fn update_noisy(&mut self) -> Option<impl Future<Output = Message>> {
        if self.r_noisy.is_empty() {
            return None;
        }
        let recv = self.r_noisy.clone();
        let spec = spec_noisy();
        Some(async move {
            let n = recv.len();
            spec.lock().expect("Failed to lock SPEC_NOISY").update(recv.iter().take(n), n);
            Message::NoisyChanged
        })
    }

    fn update_enh(&mut self) -> Option<impl Future<Output = Message>> {
        if self.r_enh.is_empty() {
            return None;
        }
        let recv = self.r_enh.clone();
        let spec = spec_enh();
        Some(async move {
            let n = recv.len();
            spec.lock().expect("Failed to lock SPEC_ENH").update(recv.iter().take(n), n);
            Message::EnhChanged
        })
    }

    fn specs(&self) -> Container<'_, Message> {
        container(column![
            spec_view("Noisy", self.noisy_img.clone(), 1000.0, 250.0),
            spec_view(
                "DeepFilterNet Enhanced",
                self.enh_img.clone(),
                1000.0,
                250.0
            ),
        ])
    }
}

fn spec_view<'a>(
    title: &'a str,
    im: image::Handle,
    width: f32,
    height: f32,
) -> Element<'a, Message> {
    column![
        text(title).size(24).width(Length::Fill),
        spec_raw(im, width, height)
    ]
    .max_width(width)
    .width(Length::Fill)
    .into()
}

fn spec_raw<'a>(im: image::Handle, width: f32, height: f32) -> Container<'a, Message> {
    container(Image::new(im).width(width).height(height).content_fit(ContentFit::Fill))
        .max_width(width)
        .max_height(height)
        .width(Length::Fill)
        .align_x(alignment::Horizontal::Center)
        .align_y(alignment::Vertical::Center)
}

#[allow(clippy::too_many_arguments)]
fn slider_view<'a>(
    title: &'a str,
    value: f32,
    min: f32,
    max: f32,
    message: impl Fn(f32) -> Message + 'a,
    width: f32,
    precision: usize,
    step: f32,
) -> Element<'a, Message> {
    column![
        text(title).size(18).width(Length::Fill),
        row![
            container(slider(min..=max, value, message).step(step)).width(Length::Fill),
            text(format!("{:.precision$}", value))
                .size(18)
                .width(100.0)
                .align_x(alignment::Horizontal::Right)
                .align_y(alignment::Vertical::Top),
        ]
    ]
    .max_width(width)
    .width(Length::Fill)
    .into()
}

fn button(text: &str) -> widget::Button<'_, Message> {
    widget::button(text).padding(10)
}
