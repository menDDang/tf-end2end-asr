import tensorflow as tf


class GRUCell(tf.keras.layers.Layer):

    def __init__(self, num_units,
        layer_norm=False, 
        dropout=False, dropout_prob=0.1,
        weight_initializer=None, bias_initializer=None, **kwargs):

        super(GRUCell, self).__init__(**kwargs)

        # Set properties
        self.num_units = num_units
        self.layer_norm = layer_norm
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        self.weight_initializer = 'uniform' if weight_initializer is None else weight_initializer
        self.bias_initializer = 'zeros' if bias_initializer is None else bias_initializer
    
        # Set following properties
        self.state_size = [num_units]

    def build(self, input_shapes):
        
        input_dim = input_shapes[1]

        # Build weight parameters
        self.W_xr = self.add_weight(shape=[input_dim, self.num_units], initializer=self.weight_initializer, name='W_xr')
        self.W_hr = self.add_weight(shape=[self.num_units, self.num_units], initializer=self.weight_initializer, name='W_hr')
        self.b_r = self.add_weight(shape=[self.num_units], initializer=self.bias_initializer, name='b_r')
        
        self.W_xz = self.add_weight(shape=[input_dim, self.num_units], initializer=self.weight_initializer, name='W_xz')
        self.W_hz = self.add_weight(shape=[self.num_units, self.num_units], initializer=self.weight_initializer, name='W_hz')
        self.b_z = self.add_weight(shape=[self.num_units], initializer=self.bias_initializer, name='b_z')
        
        self.W_xg = self.add_weight(shape=[input_dim, self.num_units], initializer=self.weight_initializer, name='W_xg')
        self.W_hg = self.add_weight(shape=[self.num_units, self.num_units], initializer=self.weight_initializer, name='W_hg')
        self.b_g = self.add_weight(shape=[self.num_units], initializer=self.bias_initializer, name='b_g')
        
        if self.layer_norm:
            self.gamma = self.add_weight(shape=[self.num_units], initializer=self.weight_initializer, name='gamma')
            self.beta = self.add_weight(shape=[self.num_units], initializer=self.bias_initializer, name='beta')
            
        self.built = True


    def call(self, x, states):

        # Get hidden states from before step
        h_prev = states[0]

        # Compute outputs of each gate
        r = tf.nn.sigmoid(tf.matmul(x, self.W_xr) + tf.matmul(h_prev, self.W_hr) + self.b_r)
        z = tf.nn.sigmoid(tf.matmul(x, self.W_xz) + tf.matmul(h_prev, self.W_hz) + self.b_z)
        g = tf.nn.tanh(tf.matmul(x, self.W_xg) + tf.matmul(r * h_prev, self.W_hg) + self.b_g)

        # Apply dropout at g
        if self.dropout:
            dropout_mask = tf.cast(
                tf.random.uniform(shape=g.shape, minval=0, maxval=1) > self.dropout_prob,
                dtype=tf.float32)
            g = dropout_mask * g / (1.0 - self.dropout_prob)  # to maintain expectation value

        # Normalize g
        if self.layer_norm:
            mu, sigma = tf.nn.moments(g, axes=[1], keepdims=True)
            g = (g - mu) / sigma
            g = self.gamma * g + self.beta

        h = z * h_prev + (1 - z) * g
        
        
        return h, [h]