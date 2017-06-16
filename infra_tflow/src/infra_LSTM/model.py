# Words
"""Start of model for Time Series Classification"""

"""Construction"""

import tensorflow as tf

class Model():
    def __init__(self, parameters):
        
        num_layers = parameters['num_layers']
        hidden_size = parameters['hidden_size']
        grad_max_abs = parameters['grad_max_abs']
        learning_rate = parameters['learning_rate']
        classes = parameters['classes']
        dropout_keep_prob = parameters['dropout_keep_prob']
        sl = parameters['sl']
        logdir = parameters['logdir']
        
        """Placeholders for input, labels, and sequence length"""
        self.input = tf.placeholder(tf.float32, [None, sl, 1], name = 'input') # check this. 1 because this is the number of inputs
        self.labels = tf.placeholder(tf.int64, [None], name = 'labels')
        self.seq_length = tf.placeholder(tf.int32, [None], name = 'sequence_length')
        
        
        # LSTM Cell, with Dropout probability
        with tf.name_scope('LSTM_setup') as scope:
            def LSTM_cell():
                return tf.contrib.rnn.DropoutWrapper(
                        tf.contrib.rnn.LSTMCell(hidden_size), 
                        output_keep_prob=dropout_keep_prob)
            
            # LSTM cell network, defined by the combination of cells
            multi_cell = tf.contrib.rnn.MultiRNNCell([LSTM_cell() for _ in range(num_layers)])
       
        # Outputs of the each cell 
        outputs, states = tf.nn.dynamic_rnn(multi_cell, self.input, sequence_length=self.seq_length, dtype=tf.float32)
        top_layer_h_state = states[-1][1]
        
        """Softmax function for the output"""
        
        with tf.name_scope("Softmax_function") as scope:

            self.logits = tf.layers.dense(top_layer_h_state, classes)
            
            #Cost Funtction, "Softmax Cross Entropy"
            self.cost_confusion = tf.nn.softmax(self.logits)
            self.cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                  labels=self.labels,
                                                                  name = 'softmax')
            # Loss function, used to compute the gradients later on
            self.loss = tf.reduce_mean(self.cost, name='loss')
        
        """Accuracies"""   
        
        with tf.name_scope("Accuracy") as scope:
            self.predictions = tf.arg_max(self.cost_confusion,1)
            self.correct_predictions = tf.equal(self.predictions, self.labels)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, tf.float32))
            

#            cost_summary = tf.summary.scalar('cost', cost)
#            accuracy_summary = tf.summary.scalar('accuracy', accuracy)        

        """Optimizer"""
        
        with tf.name_scope("Optimizer") as scope:
            #Optimizer - GradientDescentOptimizer
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            
            #Gradients, computed by the optimizer
            gradients = optimizer.compute_gradients(self.loss) #returns gradients and vars

#             Capping the gradients using a clipping function
            capped_gradients = [(tf.clip_by_value(grad, 
                                                 clip_value_max=grad_max_abs, 
                                                 clip_value_min=-grad_max_abs), var )for grad, var in gradients]

            # Training and applying the gradients
            # We do not need to use minimize because due to gradient clipping we split into two steps
            self.training_op = optimizer.apply_gradients(capped_gradients)            

        self.trainloss_summary = tf.summary.scalar('Training_Loss', self.loss)
        self.testloss_summary = tf.summary.scalar('Test_Loss', self.loss)
        self.valloss_summary = tf.summary.scalar('Validation_Loss', self.loss)
        self.file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())            
        
        
        
        
        
        
        