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


