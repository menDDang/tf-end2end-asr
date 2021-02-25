import tensorflow as tf


def serialize_example(mel, tokens):

    def _bytes_features(value):
        if isinstance(value, type(tf.contant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _serialize(_mel, _tokens):
        serialized_mel = tf.io.serialize_tensor(_mel)
        serialized_tokens = tf.io.serialize_tensor(_tokens)

        feature = {
            'mel': _bytes_features(serialized_mel),
            'tokens': _bytes_features(serialized_tokens)
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        example = example.SerializeToString()
        return example

    output = tf.py_function(
        _serialize,
        inp=[mel, tokens],
        Tout=[tf.string])

    return tf.reshape(output, ())


def parse_example(serialized_example):

    parse_dict = {
        'mel': tf.io.FixedLenFeature([], tf.string),
        'tokens': tf.io.FixedLenFeature([], tf.string)
    }

    example = tf.io.parse_single_example(serialized_example, parse_dict)

    mel = tf.io.parse_tensor(example['mel'], out_type=tf.float32)
    tokens = tf.io.parse_tensor(example['tokens'], out_type=tf.int32)

    return (mel, tokens)


def load_dataset(path):

    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(
        parse_example,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
    return dataset

def write_dataset(dataset, path):

    writer = tf.data.experimental.TFRecordWriter(path)
    writer.write(dataset)

    