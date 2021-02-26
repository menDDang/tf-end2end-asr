import tensorflow as tf


class BahdanauAttention(tf.keras.layers.Layer):

    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)


    def call(self, query, values):
        # query : hidden states from before step. shape of [batch_size, hidden_state_dim]
        # value : input tensor. shape of [batch_size, time_length, input_dim]

        # Expand dimension of query
        # [batch_size, hidden_state_dim] -> [batch_size, 1, hidden_state_dim]
        query_with_time_axis = tf.expand_dims(query, axis=1)

        # Compute score : [batch, time_length, 1]
        score = tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values))
        score = self.V(score)
        
        # Compute attention weights : [batch_size, time_length, 1]
        attention_weights = tf.nn.softmax(score, axis=1)

        # Compute context : [batch_size, time_length, hidden_state_dim]
        context = attention_weights * values

        # Add context along time axis
        context = tf.reduce_sum(context, axis=1)
        
        return context, attention_weights


