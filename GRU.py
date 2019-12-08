import tensorflow as tf
import os
import numpy as np
import pickle as pc

debug = False

class GRU(tf.keras.Model):
    def __init__(self, hidden_unit, keep_rate):
        super(GRU, self).__init__()
        
        self.hidden_unit = hidden_unit
        
        #self.layers = list()
        #for i in range(num_layer):
        #    gru = tf.keras.layers.GRU(self.hidden_unit, return_sequences=True, return_state=True, dropout=keep_rate, recurrent_initializer='glorot_uniform')
        #    self.layers.append(gru)
        self.gru1 = tf.keras.layers.GRU(self.hidden_unit, return_sequences=True, return_state=True, dropout=keep_rate, recurrent_initializer='glorot_uniform')
        #self.gru2 = tf.keras.layers.GRU(self.hidden_unit, return_sequences=True, return_state=True, dropout=keep_rate, recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(1)
        
    def call(self, inp, training):
        
        batch_size = inp.shape[0]
        horizon_size = inp.shape[1]
        
        inp = tf.reshape(inp, [batch_size, horizon_size, 1])
        
        #output = inp
        #for each in self.layers:
        #    output, state = each(output, training=training, initial_state=tf.zeros((batch_size, self.hidden_unit)))
            
        output1, state1 = self.gru1(inp, training=training, initial_state=tf.zeros((batch_size, self.hidden_unit)))
        #output2, state2 = self.gru2(output1, training=training)

        logits = self.fc(output1)
        #logits = self.fc(state2)
        #print(logits.shape)
        logits = tf.reshape(logits, (batch_size, horizon_size))
        return logits