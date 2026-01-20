import os
from df.enhance import enhance, init_df, load_audio, save_audio

if __name__ == "__main__":
    model, df_state, *_ = init_df()

    file_path = "noisy_snr0.wav"

    if not os.path.exists(file_path):
        print(f"Error: '{file_path}' not found in the current directory.")
        exit(1)

    print(f"Loading audio from: {file_path}")

    # Load audio (resamples automatically if needed)
    audio, _ = load_audio(file_path, sr=df_state.sr())

    # Run the enhancement
    print("Enhancing...")
    enhanced = enhance(model, df_state, audio)

    # Save the result
    output_file = "enhanced_test.wav"
    save_audio(output_file, enhanced, df_state.sr())
    print(f"Test successful: Saved to {output_file}")
