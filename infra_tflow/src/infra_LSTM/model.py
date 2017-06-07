# Words
"""Start of model for Time Series Classification"""

"""Construction"""

import numpy as np
import tensorflow as tf

class Model():
    def __init__(self, parameters):
        
        num_layers = parameters['num_layers']
        hidden_size = parameters['hidden_size']
        grad_max_abs = parameters['grad_max_abs']
        batch_size = parameters['batch_size']
        learning_rate = parameters['learning_rate']
        classes = parameters['classes']
        dropout_keep_prob = parameters['dropout_keep_prob']
        
        """Placeholders for input, labels, and dropout"""
        self.input = tf.placeholder(tf.float32, [None, None], name = 'input')
        self.labels = tf.placeholder(tf.int64, [None], name = 'labels')
        seq_length = tf.placeholder(tf.int32, [None])
        
        with tf.name_scope('LSTM Model') as scope:
            def LSTM_cell():
                return tf.contrib.rnn.DropoutWrapper(LSTMCell(hidden_size), keep_prob=dropout_keep_prob)
            
            multi_cell = tf.contrib.rnn.MultiRNNCell([LSTM_Cell() for _ in len(num_layers)])
       
        def return_classification(X):
            outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32, sequence_length=seq_length)
            return outputs[-1]
        
