# Turns a WAV file into a log-mel spectrogram with TensorFlow's built-in audio functions
# Used by both train.py and app.py, so the model always sees audio prepared in the same way

# For decoding audio and computing spectrograms
import tensorflow as tf


# Set key audio configuration
AUDIO_CONFIG = {
    'sample_rate': 44100, # In Hz; the sample rate of the ESC clips
    'clip_seconds': 5, # Every clip is cut or zero-padded to this length
    'frame_length': 2048, # Samples per analysis window (about 46 ms)
    'frame_step': 1024, # Step between windows (50% overlap)
    'n_mels': 64 # Number of mel bands
}

# The 10 classes of ESC-10 in label order
CLASS_NAMES = ['chainsaw', 'clock_tick', 'crackling_fire', 'crying_baby', 'dog', 'helicopter', 'rain', 'rooster', 'sea_waves', 'sneezing']


# Turn the contents of a 16-bit WAV file into a log-mel spectrogram, shape (frames, n_mels, 1)
def wav_to_spectrogram(wav_contents):
    # Decode to a mono waveform, cut or zero-padded to the configured length
    num_samples = AUDIO_CONFIG['sample_rate'] * AUDIO_CONFIG['clip_seconds']
    waveform, sample_rate = tf.audio.decode_wav(wav_contents, desired_channels=1, desired_samples=num_samples)
    if sample_rate != AUDIO_CONFIG['sample_rate']:
        raise ValueError(f"Expected a sample rate of {AUDIO_CONFIG['sample_rate']} Hz.")

    # Compute the power spectrogram, then pool its frequency bins into mel bands
    stft = tf.signal.stft(tf.squeeze(waveform, axis=-1), AUDIO_CONFIG['frame_length'], AUDIO_CONFIG['frame_step'])
    power_spectrogram = tf.abs(stft) ** 2
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        AUDIO_CONFIG['n_mels'], power_spectrogram.shape[-1], AUDIO_CONFIG['sample_rate'], 0.0, AUDIO_CONFIG['sample_rate'] / 2
    ) # From 0 Hz up to half the sample rate
    mel_spectrogram = tf.matmul(power_spectrogram, mel_matrix)

    # Take the log (1e-6 prevents taking the log of zero in silent parts) and add a channel axis, as Conv2D expects
    return tf.math.log(mel_spectrogram + 1e-6)[..., tf.newaxis].numpy()
