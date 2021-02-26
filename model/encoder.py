import tensorflow as tf

from .lstm_cell import LSTMCell

class Encoder(tf.keras.layers.Layer):

    def __init__(self, num_layers, num_units, 
                dropout=False, dropout_prob=0.1,
                layer_norm=False,
                **kwargs):
        
        super(Encoder, self).__init__(**kwargs)

        self.nn = tf.keras.Sequential()
        for n in range(num_layers):
            self.nn.add(
                tf.keras.layers.RNN(
                    LSTMCell(num_units, layer_norm, dropout, dropout_prob, name='lstm_cell_'+str(n), **kwargs),
                    return_sequences=True))


    def call(self, x, **kwargs):
        return self.nn(x)