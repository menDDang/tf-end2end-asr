import soundfile as sf
import tensorflow as tf


def load_audio(file_path):
    
    audio, sr = sf.read(file_path)
    
    return audio

