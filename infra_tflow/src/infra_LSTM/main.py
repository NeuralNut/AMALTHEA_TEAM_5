#%%
"""
Created on Mon Jun  5 16:44:02 2017

@author: emilyjensen

Main file to test on UCR and infrasound LSTM datasets.
Loads data, defines hyperparameters, calls model build, trains model, and prints
results of training

Does not use GPU
"""

"""
TODO: 	Add TensorBoard functionality as well as checkpoints
	Make confusion matrix rows/cols names of classes
	fix print accuracies 
	Tensorboard 
	confusion matrix 
	log experimental params 
	save/restore scheme
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model import Model

tf.reset_default_graph()

#%%
"""Load and prepare data"""
# Define function for creating sequence length vectors and adding zero buffers
def find_seq_lengths(X):
    N,max_length = X.shape
    seq_lengths = np.zeros(N)
    # Iterate over every sample in dataset
    for i in range(N):
        # start from the last element
        ind = -1
        # find the index of the first non-zero element
        while X[i,ind] == 0:
            ind -= 1
        seq_lengths[i] = max_length + ind + 1
    return seq_lengths.astype(np.int32)

# Define function for loading and splitting data into relevant sets
def load_data(direc,ratio,dataset):
    """Input:
        direc: location of the data archive
        ratio: ratio to split training set into training and validation
        dataset: name of the dataset in the archive"""
    # Define directory of specific dataset
    datadir = direc + '/' + dataset + '/' + dataset
    # Load pre-split training and testing sets
    data_train = np.loadtxt(datadir+'_TRAIN',delimiter=',').astype(np.float32)
    data_seq = find_seq_lengths(data_train[:,1:])
    data_test = np.loadtxt(datadir+'_TEST',delimiter=',').astype(np.float32)
    # Divide training set into training and validation sets
    # First column of data is class number
    # Subtract 1 from classes to index from 0
    N = data_train.shape[0]
    ratio = int(ratio*N)
    np.random.shuffle(data_train)
    X_train = data_train[:ratio,1:]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    train_seq = data_seq[:ratio]
    y_train = (data_train[:ratio,0] - 1).astype(np.int32)
    X_val = data_train[ratio:,1:]
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    val_seq = data_seq[ratio:]
    y_val = (data_train[ratio:,0] - 1).astype(np.int32)
    # Permute testing set
    np.random.shuffle(data_test)
    X_test = data_test[:,1:]
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    test_seq = find_seq_lengths(data_test[:,1:])
    y_test = (data_test[:,0] - 1).astype(np.int32)
    
    return X_train,train_seq,X_val,val_seq,X_test,test_seq,y_train,y_val,y_test

# Load the desired dataset
direc = '/home/likewise-open/FLTECH/msolomon2010/Documents/AMALTHEA_TEAM_5/infra_tflow/src/data/UCR_TS_Archive_2015'
# Splits training set into training and validation sets
ratio = 0.8
X_train,train_seq,X_val,val_seq,X_test,test_seq,y_train,y_val,y_test = load_data(direc,ratio,dataset='ChlorineConcentration')

#%%
"""Define configuration of hyperparameters"""
# Define hyperparameters also used in this file
batch_size = np.array(30, dtype=np.int32)
epochs = 10
dropout = np.array(0.8, dtype=np.float32)
num_classes = (max(y_test) + 1).astype(np.int32)
config = {'num_layers':3, # number of hidden LSTM layers
          'hidden_size':120, # number of units in each layer
          'grad_max_abs':5, # cutoff for gradient clipping
          'batch_size':batch_size,
          'learning_rate':0.1,
          'classes':num_classes,
          'dropout_keep_prob':dropout,
          'sl':X_train.shape[1]
        }

"""Create a new model object"""
model = Model(config)
X = model.input
y = model.labels
#%%
"""Train the model"""
# Define function to divide training data into mini-batches
def create_batches(X,y,seq_lengths,batch_size):
    # Loop over all samples in groups of batch_size
    for i in np.arange(0,X.shape[0],batch_size):
        # Make a list of batches of samples
        yield (X[i:i+batch_size],y[i:i+batch_size],seq_lengths[i:i+batch_size])

# initialize variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    for epoch in range(epochs):
        # Iterate through each mini-batch once per epoch
        for (batch_x,batch_y,batch_seq) in create_batches(X_train,y_train,train_seq,batch_size):
            # Reset accuracy count
            epoch_acc = 0
            # Run training on the batch
            sess.run(model.training_op,feed_dict={X:batch_x, y:batch_y,model.seq_length:batch_seq})
            # Assess and add to accuracy count
            epoch_acc += model.accuracy.eval(feed_dict={X:batch_x, y:batch_y,model.seq_length:batch_seq}) * batch_x.shape[0]
        # After going through each mini-batch, test against validation set
        validation_acc = model.accuracy.eval(feed_dict={X:X_val,y:y_val,model.seq_length:val_seq})
        # Calculate cost of validation set
        validation_cost = model.loss.eval(feed_dict={X:X_val,y:y_val,model.seq_length:val_seq})
        # Print accuracy and cost updates for each epoch
        print('#',epoch,'Epoch train accuracy:',epoch_acc/X_train.shape[0],
              'Validation accuracy:',validation_acc,'Validation cost:',validation_cost)

        # Shuffle samples for next epoch
        np.random.shuffle(X_train)
    # Calculate cost and accuracy for final test set
    test_acc = model.accuracy.eval(feed_dict={X:X_test,y:y_test,model.seq_length:test_seq})
    test_cost = model.loss.eval(feed_dict={X:X_test,y:y_test,model.seq_length:test_seq})
    test_prediction = model.predictions
    
#%%
"""Display results"""
print('Final accuracy:',test_acc,'Final cost:',test_cost)

# create confusion matrix to visualize classification errors
confusion_matrix = tf.confusion_matrix(y_test,test_prediction)
cf_normed = np.array(confusion_matrix,dtype=np.float32)/np.sum(confusion_matrix) * 100
width = 12
height = 12
plt.figure(figsize=(width,height))
plt.imshow(
	cf_normed,
	interpolation='nearest',
	cmap=plt.cm.rainbow
	)

plt.title("Confusion matrix \n(normalised to percent of total test data)")
plt.colorbar()
tick_marks = np.arange(num_classes)
# Make a list of strings of the numbered classes
LABELS = [str(i+1) for i in range(num_classes)] # Can change later once we have name labels if we want
plt.xticks(tick_marks, LABELS, rotation=90)
plt.yticks(tick_marks, LABELS)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
