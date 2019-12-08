import tensorflow as tf

debug = False

class LuongAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(LuongAttention, self).__init__()
        
        self.W = tf.keras.layers.Dense(units)
        
    def call(self, query, values):
        
        hidden_with_time_axis = tf.expand_dims(query, 1)
        
        score = tf.matmul(hidden_with_time_axis, self.W(values))
        if debug:
            print("score: {}".format(score.shape))
        attention_weights = tf.nn.softmax(score, axis=1)
        
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights        
        
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        
    def call(self, query, values):
        
        if debug:
            print("query: {}\tvalues: {}".format(query.shape, values.shape))
        
        hidden_with_time_axis = tf.expand_dims(query, 1)
        
        if debug:
            print("hidden_query: {}".format(hidden_with_time_axis.shape))
        
        linear_values = self.W1(values)
        linear_query = self.W2(hidden_with_time_axis)
        
        if debug:
            print("linear_value: {}\tlinear_query: {}".format(linear_values.shape, linear_query.shape))
            
        score = self.V(tf.nn.tanh(linear_values + linear_query))
        if debug:
            print("score: {}".format(score.shape))
        
        attention_weights = tf.nn.softmax(score, axis=1)
        if debug:
            print("attention_weights: {}".format(attention_weights.shape))
        
        context_vector = attention_weights * values
        if debug:
            print("context_vector: {}".format(context_vector.shape))
            
        context_vector = tf.reduce_sum(context_vector, axis=1)
        if debug:
            print("context_vector: {}".format(context_vector.shape))
        
        return context_vector, attention_weights
        