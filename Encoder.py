import tensorflow as tf

debug = False

class Encoder(tf.keras.Model):
    def __init__(self, enc_unit, batch_size, horizon_size, dropout_rate):
        super(Encoder, self).__init__()
        
        self.batch_size = batch_size
        self.enc_unit = enc_unit
        self.horizon_size = horizon_size
        #self.emb_dim = emb_dim
        #self.embedding = tf.keras.layers.Embedding(vocab_size, emb_dim)
        
        self.gru = tf.keras.layers.GRU(self.enc_unit, return_sequences=True, return_state=True, dropout=dropout_rate, recurrent_initializer='glorot_uniform')

        
    def call(self, enc_input, training):
        #enc_emb_input = self.embedding(enc_input)
        #assert enc_input.shape == (self.batch_size, enc_input[1].shape, self.emb_dim)
        
        batch_size = enc_input.shape[0]
        
        if debug:
            print("enc input: {}".format(enc_input.shape))
            print(batch_size, self.horizon_size)
        enc_input = tf.reshape(enc_input, [batch_size, self.horizon_size, 1])
        
        if debug:
            print("enc input: {}".format(enc_input.shape))
            
        output, state = self.gru(enc_input, training=training, initial_state=tf.zeros((batch_size, self.enc_unit)))

        if debug:
            print("output: {}\tstate: {}".format(output.shape, state.shape))
        
        return output, state
    
    

        
    
