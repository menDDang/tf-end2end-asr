import soundfile as sf
import tensorflow as tf


def load_audio(file_path):
    
    audio, sr = sf.read(file_path)
    
    return audio, sr


def compute_spectrum(audio, sr, frame_length_in_sec, step_length_in_sec, num_fft):

    frame_length = tf.cast(tf.round(float(sr) * frame_length_in_sec), tf.int32)
    step_length = tf.cast(tf.round(float(sr) * step_length_in_sec), tf.int32)

    stft = tf.signal.stft(audio, frame_length=frame_length, frame_step=step_length, fft_length=num_fft)
    magnitudes = tf.abs(stft)

    return magnitudes


def compute_mel_bins(sampling_rate, fft_length, num_mels, hertz_low, hertz_high):

    mel_bins = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_mels,
        num_spectrogram_bins=int(fft_length / 2) + 1,
        sample_rate=sampling_rate,
        lower_edge_hertz=hertz_low,
        upper_edge_hertz=hertz_high
    )

    return mel_bins


def compute_mel(spectrum, mel_bins, normalize_mel=True):

    mel = tf.tensordot(spectrum, mel_bins, 1)
    mel.set_shape(spectrum.shape[:-1].concatenate(mel_bins.shape[-1:]))

    log_mel = tf.math.log(mel + 1e-6)
    if normalize_mel:
        log_mel -= tf.reduce_mean(log_mel, axis=0)
    
    return log_mel