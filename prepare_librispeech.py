import os

import tensorflow as tf

import preprocess


def get_transcripts(data_dir, names):

    transcripts = []

    for name in names:
        for speaker_id in os.listdir(os.path.join(data_dir, name)):
            if speaker_id == '.DS_Store': 
                continue

            for chapter_id in os.listdir(os.path.join(data_dir, name, speaker_id)):
                if chapter_id == '.DS_Store': 
                    continue

                transcript_file_name = os.path.join(data_dir, name, speaker_id, chapter_id, speaker_id + '-' + chapter_id + '.trans.txt')
                transcripts.append(transcript_file_name)
    
    return transcripts


def parse_line(line, data_dir, split_names):

    line_split = tf.strings.split(line, ' ')

    # Parse transcripts
    uttid = line_split[0]
    transcript = tf.py_function(
        lambda x: b' '.join(x.numpy()).decode('utf8'),
        inp=[line_split[1:]],
        Tout=tf.string
    )

    # Parse audio file names
    speaker_id, chapter_id, _ = tf.unstack(tf.strings.split(uttid, '-'), 3)
    audio_file_paths = tf.map_fn(
        lambda split_name: tf.strings.join([data_dir, split_name, speaker_id, chapter_id, uttid], '/') + '.flac',
        tf.constant(split_names)
    )

    # Load audio files
    audio, sr = tf.py_function(
        lambda path: preprocess.load_audio(path[0].numpy()),
        inp=[audio_file_paths],
        Tout=[tf.float32, tf.int32]
    )
    
    return audio, sr, transcript
    

def load_librispeech_dataset(data_dir, split_names):

    transcript_file_paths = get_transcripts(data_dir, split_names)

    # dataset : (transcript_file_path)
    dataset = tf.data.TextLineDataset(transcript_file_paths)

    # dataset : (audio, sr, transcript)
    dataset = dataset.map(
        lambda line: parse_line(line, data_dir, split_names),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    return dataset


def preprocess_librispeech(data_dir, split_names, hp):

    # Parse hyper parameters
    sampling_rate = hp["sampling_rate"]
    num_fft = hp["num_fft"]
    frame_length_in_sec = hp["frame_length_in_sec"]
    step_length_in_sec = hp["step_length_in_sec"]
    num_mels = hp["num_mels"]
    hertz_low = hp["hertz_low"]
    hertz_high = hp["hertz_high"]
    normalize_mel = hp["normalize_mel"]
    
    # dataset : (audio, sr, transcript)
    dataset = load_librispeech_dataset(data_dir, split_names)

    # dataset : (audio, sr, tokenized_transcription)
    tokenizer = preprocess.create_char_tokenizer()
    dataset = dataset.map(
        lambda audio, sr, transcript: (audio, sr, preprocess.tokenize_text(transcript, tokenizer)),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    # build spectrum extractor
    spectrum_extractor = lambda audio, sr: (
        preprocess.compute_spectrum(audio, sr, frame_length_in_sec, step_length_in_sec, num_fft))

    # build mel extractor
    mel_bins = preprocess.compute_mel_bins(sampling_rate, num_fft, num_mels, hertz_low, hertz_high)
    mel_extractor = lambda audio, sr: (
        preprocess.compute_mel(spectrum_extractor(audio, sr), mel_bins, normalize_mel))
    
    # dataset : (mel, tokenized_transcription)
    dataset = dataset.map(
        lambda audio, sr, tokens: (mel_extractor(audio, sr), tokens),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    return dataset


if __name__ == "__main__": 

    hp = dict()
    hp["sampling_rate"] = 16000
    hp["num_fft"] = 512
    hp["frame_length_in_sec"] = 0.025
    hp["step_length_in_sec"] = 0.01
    hp["num_mels"] = 80
    hp["hertz_low"] = 125
    hp["hertz_high"] = 7600
    hp["normalize_mel"] = True

    dataset = preprocess_librispeech("data/LibriSpeech", ["dev-clean"], hp)

    for b, inputs in enumerate(dataset.take(1)):

        mel, tokens = inputs
        print(mel.shape)
        print(tokens)