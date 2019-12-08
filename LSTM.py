import tensorflow as tf
import os
import numpy as np
import pickle as pc

debug = False

class LSTM(tf.keras.Model):
    def __init__(self, hidden_unit, keep_rate):
        super(LSTM, self).__init__()
        
        self.hidden_unit = hidden_unit
        
        #3elf.layers = list()
        #for i in range(num_layer):
            #lstm = tf.keras.layers.LSTM(self.hidden_unit, return_sequences=True, return_state=True, dropout=keep_rate, recurrent_initializer='glorot_uniform')
            #elf.layers.append(lstm)
            
        self.lstm1 = tf.keras.layers.LSTM(self.hidden_unit, return_sequences=True, return_state=True, dropout=keep_rate, recurrent_initializer='glorot_uniform')
        self.lstm2 = tf.keras.layers.LSTM(self.hidden_unit, return_sequences=True, return_state=True, dropout=keep_rate, recurrent_initializer='glorot_uniform')
        
        self.fc = tf.keras.layers.Dense(1)
        
    def call(self, inp, training):
        
        batch_size = inp.shape[0]
        horizon_size = inp.shape[1]
        
        if debug:
            print("batch size: {}".format(batch_size))
            print("horizon_size: {}".format(horizon_size))
            print("input: {}".format(inp.shape))
        inp = tf.reshape(inp, [batch_size, horizon_size, 1])
        
        #output = inp
        #for each in self.layers:
        
        output1 = self.lstm1(inp, training=training, initial_state=[tf.zeros((batch_size, self.hidden_unit)),tf.zeros((batch_size, self.hidden_unit))])
        output2 = self.lstm2(output1, training=training)
        
        _output = output2[1]
        
        logits = self.fc(_output)
        
        return logits