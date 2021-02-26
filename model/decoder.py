import tensorflow as tf

from .attention import *
from .gru_cell import GRUCell

class Decoder(tf.keras.layers.Layer):

    def __init__(self, 
                attention_unit_num, vocab_size, embedding_dim, gru_unit_num,
                fc_layer_num, fc_unit_num,
                attention_type='Bahdanau',
                gru_layer_norm=False, gru_dropout=False, gru_dropout_prob=0.1,
                fc_activation='relu',
                **kwargs):

        super(Decoder, self).__init__(**kwargs)

        # Attention
        if attention_type == 'Bahdanau':
            self.attention = BahdanauAttention(attention_unit_num)

        # Embedding
        self.embeddnig = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        
        # Fully connected layers
        self.fc = tf.keras.Sequential()
        for n in range(fc_layer_num - 1):
            self.fc.add(tf.keras.layers.Dense(fc_unit_num, activation=fc_activation))
        self.fc.add(tf.keras.layers.Dense(vocab_size))

        # gru cell
        self.cell = GRUCell(gru_unit_num, gru_layer_norm, gru_dropout, gru_dropout_prob, name='gru_cell-'+str(n))


    def call(self, decoder_output_before, hidden_states_before, encoder_outputs):
        # decoder_output_before : [batch_size, 1]
        # hidden_states_before : [batch_size, gru_unit_num]
        # encoder_outputs : [batch_size, time_length, input_dim]

        # Apply attention
        # shape of outputs : [batch_size, attention_unit], [batch_size, time_length, 1]
        context, attention_weights = self.attention(hidden_states_before, encoder_outputs)
        
        # Compute character embedding vectors
        # shape of outputs : [batch_size, 1, embedding_dim]
        embedding_vectors = self.embeddnig(decoder_output_before)
        
        # Concatenate embedding vector and context vector
        # shape of outputs : [batch_size, embedding_dim + attention_unit]
        cell_inputs = tf.concat([tf.squeeze(embedding_vectors, axis=1), context], axis=1)

        # GRU cell
        cell_outputs, [hidden_states_now] = self.cell(cell_inputs, [hidden_states_before])

        # Fully connected layer
        output = self.fc(cell_outputs)

        return output, hidden_states_now, attention_weights

