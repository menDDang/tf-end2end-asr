import tensorflow as tf


def build_lookup_table(keys, values=None, default_value=0):
    if values is None:
        values = tf.range(len(keys))

    init_key_values = tf.lookup.KeyValueTensorInitializer(keys=keys, values=values)

    hash_table = tf.lookup.StaticHashTable(init_key_values, default_value=default_value)

    return hash_table


