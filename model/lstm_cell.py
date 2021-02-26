import tensorflow as tf


class LSTMCell(tf.keras.layers.Layer):

    def __init__(self, units, 
        layer_norm=False, 
        dropout=False, dropout_prob=0.1,
        weight_initializer=None, bias_initializer=None, **kwargs):

        super(LSTMCell, self).__init__(**kwargs)

        # Set properties
        self.num_units = units
        self.layer_norm = layer_norm
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        self.weight_initializer = 'uniform' if weight_initializer is None else weight_initializer
        self.bias_initializer = 'zeros' if bias_initializer is None else bias_initializer
    
        # Set following properties
        self.state_size = [units, units]


    def build(self, input_shapes):

        # Get input shapes 
        batch_size, input_dim = input_shapes

        # Build weight parameters
        self.W_xf = self.add_weight(shape=[input_dim, self.units], initializer=self.weight_initializer, name='W_xf')
        self.W_hf = self.add_weight(shape=[self.units, self.units], initializer=self.weight_initializer, name='W_hf')
        self.b_f = self.add_weight(shape=[self.units], initializer=self.bias_initializer, name='b_f')
        
        self.W_xi = self.add_weight(shape=[input_dim, self.units], initializer=self.weight_initializer, name='W_xi')
        self.W_hi = self.add_weight(shape=[self.units, self.units], initializer=self.weight_initializer, name='W_hi')
        self.b_i = self.add_weight(shape=[self.units], initializer=self.bias_initializer, name='b_i')
        
        self.W_xo = self.add_weight(shape=[input_dim, self.units], initializer=self.weight_initializer, name='W_xo')
        self.W_ho = self.add_weight(shape=[self.units, self.units], initializer=self.weight_initializer, name='W_ho')
        self.b_o = self.add_weight(shape=[self.units], initializer=self.bias_initializer, name='b_o')
        
        self.W_xg = self.add_weight(shape=[input_dim, self.units], initializer=self.weight_initializer, name='W_xg')
        self.W_hg = self.add_weight(shape=[self.units, self.units], initializer=self.weight_initializer, name='W_hg')
        self.b_g = self.add_weight(shape=[self.units], initializer=self.bias_initializer, name='b_g')
        
        if self.layer_norm:
            self.gamma = self.add_weight(shape=[self.units], initializer=self.weight_initializer, name='gamma')
            self.beta = self.add_weight(shape=[self.units], initializer=self.bias_initializer, name='beta')
            
        self.built = True
    
        
    def call(self, inputs, states, training=True):

        # Get hidden states from before step
        c_prev, h_prev = states[0], states[1]
        
        # Compute outputs of each gate
        f = tf.nn.sigmoid(tf.matmul(inputs, self.W_xf) + tf.matmul(h_prev, self.W_hf) + self.b_f)
        i = tf.nn.sigmoid(tf.matmul(inputs, self.W_xi) + tf.matmul(h_prev, self.W_hi) + self.b_i)
        o = tf.nn.sigmoid(tf.matmul(inputs, self.W_xo) + tf.matmul(h_prev, self.W_ho) + self.b_o)
        g = tf.nn.tanh(tf.matmul(inputs, self.W_xg) + tf.matmul(h_prev, self.W_hg) + self.b_g)
        if self.dropout:
            dropout_mask = tf.cast(
                tf.random.uniform(shape=g.shape, minval=0, maxval=1) > self.dropout_prob,
                dtype=tf.float32)
            g = dropout_mask * g / (1.0 - self.dropout_prob)  # to maintain expectation value

        c = f * c_prev + i * g  # shape of c : [batch_size, self.units]
        
        # Normalize c
        if self.layer_norm:
            mu, sigma = tf.nn.moments(c, axis=[1], keepdims=True)
            c = (c - mu) / sigma
            c = self.gamma * c + self.beta
        
        # Compute hidden state  
        h = o * tf.nn.tanh(c)

        return h, [c, h]  # outputs, states
        
    
    def get_initial_hidden_states(self, batch_size):

        return tf.zeros(shape=[batch_size, self.num_units], dtype=self.dtype)
        
        