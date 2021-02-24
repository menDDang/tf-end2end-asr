import os

import tensorflow as tf


def normalize_text(text):

    text = text.lower()
    text = text.replace('"', '')

    return text

def create_char_tokenizer():

    alphabet = "abcdefghijklmnopqrstuvwxyz'"
    end = '</s>'  # this token will be 0. Note that 0 means end of sentence
    keys = [end, ' '] + [c for c in alphabet]

    values = range(len(keys))

    init_key_values = tf.lookup.KeyValueTensorInitializer(keys=keys, values=values)

    hash_table = tf.lookup.StaticHashTable(init_key_values, default_value=0)

    return lambda text: hash_table.lookup(tf.strings.bytes_split(text))


def tokenize_text(text, tokenizer):

    def _normalize(_text):
        return tf.py_function(
        lambda x: normalize_text(x.numpy().decode('utf8')),
        inp=[_text],
        Tout=tf.string)

    normalized_text = _normalize(text)
    tokens = tokenizer(tf.reshape(normalized_text, ()))

    return tokens
