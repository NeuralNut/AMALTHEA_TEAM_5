#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:04:17 2017

@author: msolomon2010
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import io
from sklearn.metrics import confusion_matrix
import tensorflow as tf

def list_UCR_datasets(direc):
    return [name for name in os.listdir(direc)]
    

"""Load and prepare data"""
# Define function for creating sequence length vectors
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
def load_data(direc,ratio,dataset_name):
    """Input:
        direc: location of the data archive
        ratio: ratio to split training set into training and validation
        dataset: name of the dataset in the archive"""
    # Define directory of specific dataset
    datadir = direc + '/' + dataset_name + '/' + dataset_name
    # Load pre-split training and testing sets
    data_train = np.loadtxt(datadir+'_TRAIN',delimiter=',').astype(np.float32)
    data_test = np.loadtxt(datadir+'_TEST',delimiter=',').astype(np.float32)
    # Divide training set into training and validation sets
    # First column of data is class number
    # Subtract 1 from classes to index from 0
    N = data_train.shape[0]
    ratio = int(ratio*N)
    np.random.shuffle(data_train)
    np.random.shuffle(data_test)
    
    #split labels based on ratio
    y_train = (data_train[:ratio,0]).astype(np.int32)
    y_val = (data_train[ratio:,0]).astype(np.int32)
    y_test = (data_test[:,0]).astype(np.int32)
    
    # 0 index all labels
    nb_classes = len(np.unique(y_test))
    y_train = ((y_train - y_train.min())/(y_train.max()-y_train.min())*(nb_classes-1)).astype(np.int32)
    y_test = ((y_test - y_test.min())/(y_test.max()-y_test.min())*(nb_classes-1)).astype(np.int32)
    y_val = ((y_val - y_val.min())/(y_val.max()-y_val.min())*(nb_classes-1)).astype(np.int32)

    #find the length of each sequence
    data_seq = find_seq_lengths(data_train[:,1:])
    
    #split data based on ratio
    X_train = data_train[:ratio,1:]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    train_seq = data_seq[:ratio]
    X_val = data_train[ratio:,1:]
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    val_seq = data_seq[ratio:]
    X_test = data_test[:,1:]
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    test_seq = find_seq_lengths(data_test[:,1:])
    
    # z-normalize the test/train/val data
    X_train = (X_train - X_train.mean())/X_train.std()
    X_test = (X_test - X_test.mean())/X_test.std()
    X_val = (X_val - X_val.mean())/X_test.std()
    
    return X_train,train_seq,X_val,val_seq,X_test,test_seq,y_train,y_val,y_test



# Define function to divide training data into mini-batches
def create_batches(X,y,seq_lengths,batch_size):
    # Loop over all samples in groups of batch_size
    for i in np.arange(0,X.shape[0],batch_size):
        # Make a list of batches of samples
        B = (X[i:i+batch_size],y[i:i+batch_size],seq_lengths[i:i+batch_size])
        yield B

# Define function to create and save confusion matric
def create_confusion_matrix(y_test,test_prediction,num_classes,sess,model,epoch,dataset):
    # create confusion matrix to visualize classification errors
    confusion_matrix_array = confusion_matrix(y_test,test_prediction)
    cf_normed = np.array(confusion_matrix_array)/np.sum(confusion_matrix_array) * 100
    width = 5
    height = 5
    plt.figure(figsize=(width,height))
    plt.title("%s Confusion Matrix"%(dataset))

    plt.imshow(cf_normed, interpolation='nearest', cmap=plt.cm.Blues)

    tick_marks = np.arange(num_classes)

    # Make a list of strings of the numbered classes
    LABELS = [str(i+1) for i in range(num_classes)] # Can change later once we have name labels if we want
    plt.xticks(tick_marks, LABELS, rotation=90)
    plt.yticks(tick_marks, LABELS)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    plt.savefig('./matrices/matrix%d.png'%(epoch),bbox_inches='tight')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    summary_op = tf.summary.image("Confusion_Matrix", image)
    summary = sess.run(summary_op)
    model.file_writer.add_summary(summary)
    plt.close()
    #print(confusion_matrix_array)