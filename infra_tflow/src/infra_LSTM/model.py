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
        
        
        # LSTM Cell, with Dropout probability
        with tf.name_scope('LSTM Model') as scope:
            def LSTM_cell():
                return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(hidden_size), keep_prob=dropout_keep_prob)
            
            # LSTM cell network, defined by the combination of cells
            multi_cell = tf.contrib.rnn.MultiRNNCell([LSTM_cell() for _ in len(num_layers)])
       
        # Outputs of the each cell 
        outputs, states = tf.nn.dynamic_rnn(multi_cell, self.input, dtype=tf.float32)
        
        # Output of the final cell
        output = outputs[-1]
            
        #Function to return the final cell output of the LSTM Network 
        def return_classification(X):
            outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32, sequence_length=seq_length)
            return outputs[-1]
        
        """Softmax function for the output"""
        
        with tf.name_scope("Softmax_function") as scope:
            with tf.name_scope("Softmax_parameters"):
                softmax_W = tf.get_variable("softmax_W", [hidden_size, classes])
                softmax_b = tf.get_variable("softmax_b", [classes])
                
            logits = tf.layers.dense(output, softmax_W, softmax_b)
            
            #Loss Funtction, "Softmax Cross Entropy"
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=self.labels,name = 'softmax')
            
            # Cost function, used to compute the gradients later on
            cost = (tf.reduce_sum(loss) / batch_size)
            
            
        # Accuracy, evaluated but not printed anywhere
        with tf.name_scope("Evalutating_accuracy") as scope:
            correct = tf.equal(tf.argmax(logits,1),self.labels)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        
#            cost_summary = tf.summary.scalar('cost', cost)
#            accuracy_summary = tf.summary.scalar('accuracy', accuracy)
        
        """Optimizer"""
        
        with tf.name_scope("Optimizer") as scope:
            
            #Optimizer - GradientDescentOptimizer
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            
            #Gradients, computed by the optimizer
            gradients = optimizer.compute_gradients(cost)
            
            # Capping the gradients using a clipping function
            capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]

            # Training and applying the gradients
            training_op = optimizer.apply_gradients(capped_gradients)
        
        
        
        
        
        
        
        