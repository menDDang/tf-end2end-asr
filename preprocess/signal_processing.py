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

