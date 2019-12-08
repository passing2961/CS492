import tensorflow as tf

from Encoder import *
from Decoder import *
from Attention import *
    
class Seq2Seq(tf.keras.Model):
    def __init__(self, enc_unit, dec_unit, batch_size, horizon_size, dropout_rate):
        super(Seq2Seq, self).__init__()
        
        self.batch_size = batch_size
        
        self.encoder = Encoder(enc_unit, batch_size, horizon_size+1, dropout_rate)
        self.decoder = Decoder(dec_unit, batch_size, horizon_size+1, dropout_rate)
        
    def call(self, enc_inputs, dec_inputs, training, mode):
        
        enc_output, enc_state = self.encoder(enc_inputs, training=training)
        
        dec_hidden = enc_state
        
        if debug:
            print("dec inputs: {}".format(dec_inputs.shape))
            
        #dec_input = dec_inputs([:, 0])
        dec_input = tf.expand_dims([2]*self.batch_size, 1)
        if debug:
            print(dec_input)
            
        if mode == 'train':
            for t in range(dec_inputs.shape[1]):
                prediction, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output, training=training)
                loss +
                dec_input = tf.expand_dims(dec_inputs[:, t+1], 1)
            if debug:
                print("logits: {}".format(prediction.shape))
        
        return prediction
        #for t in range(1, dec_inputs.shape[1]):
        #    logits, dec_hidden, _ = self.decoder(dec_inputs, dec_hidden, enc_output)
        #    
        #    loss += loss_fn(dec_inputs[: , t], 1)
        #    
        #    dec_inputs = tf.expand_dims(dec_targ)
    