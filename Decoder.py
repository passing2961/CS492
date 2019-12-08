import tensorflow as tf
from Attention import *

debug = False

class Decoder(tf.keras.Model):
    def __init__(self, dec_unit, batch_size, horizon_size, dropout_rate, attn):
        super(Decoder, self).__init__()
        
        self.batch_size = batch_size
        self.dec_unit = dec_unit
        #self.embedding = tf.keras.layers.Embedding(vocab_size, emb_dim)
        #self.max_len = max_len
        self.horizon_size = horizon_size
        self.attn = attn
        
        self.gru = tf.keras.layers.GRU(self.dec_unit, return_sequences=True, return_state=True, dropout=dropout_rate, recurrent_initializer='glorot_uniform')

        
        self.fc = tf.keras.layers.Dense(1, activation='relu')
        
        if attn == 'bah':
            self.attention = BahdanauAttention(self.dec_unit)
        elif attn == 'luong':
            self.attention = LuongAttention(self.dec_unit)
            
    def call(self, dec_input, enc_context, enc_output, training):
        
        batch_size = dec_input.shape[0]
        
        assert enc_output.shape == (batch_size, self.horizon_size, self.dec_unit)
        
        if self.attn == 'bah' or self.attn == 'luong':
            context_vector, attention_weights = self.attention(enc_context, enc_output)
        
        if debug:
            print("context_vector: {}\tattention_weights: {}".format(context_vector.shape, attention_weights.shape))
        
        #dec_emb_input = self.embedding(dec_input)
        if debug:
            print("dec input: {}".format(dec_input.shape))
        dec_input = tf.reshape(dec_input, (batch_size, 1, 1))
        #if debug:
        #    print("dec_emb_input: {}".format(dec_emb_input.shape))
        if debug:
            print("dec input: {}".format(dec_input.shape))
        
        if self.attn == 'bah' or self.attn == 'luong':
            dec_concat = tf.concat([tf.expand_dims(context_vector, 1), dec_input], axis=-1)
        if debug:
            print("dec_concat: {}".format(dec_concat.shape))
        
        if self.attn == 'bah' or self.attn == 'luong':
            output, state = self.gru(dec_concat, training=training, initial_state=enc_context)
            #output, state1 = self.gru1(dec_concat, training=training, initial_state=enc_context)

        else:
            output, state = self.gru(dec_input, training=training, initial_state=enc_context)
            
        if debug:
            print("output: {}\tstate: {}".format(output.shape, state.shape))
        
        output = tf.reshape(output, (-1, output.shape[2]))
        if debug:
            print("output: {}".format(output.shape))
        
        prediction = self.fc(output)
        if debug:
            print("logits: {}".format(prediction.shape))
        
        #prediction = tf.reshape(prediction, (-1,))
        
        if self.attn == 'bah' or self.attn == 'luong':
            return prediction, state, attention_weights
        else:
            return prediction, state
        