#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:04:17 2017

@author: msolomon2010
"""


#%%

import numpy as np

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
def load_data(direc,ratio,dataset):
    """Input:
        direc: location of the data archive
        ratio: ratio to split training set into training and validation
        dataset: name of the dataset in the archive"""
    # Define directory of specific dataset
    datadir = direc + '/' + dataset + '/' + dataset
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
    
    if np.min(data_train[:,0]) == 1.0:
        y_train = (data_train[:ratio,0] - 1).astype(np.int32)
        y_val = (data_train[ratio:,0] - 1).astype(np.int32)
        y_test = (data_test[:,0] - 1).astype(np.int32)
        print('The raw data targets are %d indexed and were converted to 0 index ' %(np.min(data_train[:,0])))

    elif np.min(data_train[:,0]) == 0.0:
        y_train = (data_train[:ratio,0]).astype(np.int32)
        y_val = (data_train[ratio:,0]).astype(np.int32)
        y_test = (data_test[:,0]).astype(np.int32)
    else:
        print('Something may have gone wrong in load_data')
    data_seq = find_seq_lengths(data_train[:,1:])
    X_train = data_train[:ratio,1:]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    train_seq = data_seq[:ratio]
    X_val = data_train[ratio:,1:]
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    val_seq = data_seq[ratio:]
    X_test = data_test[:,1:]
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    test_seq = find_seq_lengths(data_test[:,1:])
    
    return X_train,train_seq,X_val,val_seq,X_test,test_seq,y_train,y_val,y_test

# Define function to divide training data into mini-batches
def create_batches(X,y,seq_lengths,batch_size):
    # Loop over all samples in groups of batch_size
    for i in np.arange(0,X.shape[0],batch_size):
        # Make a list of batches of samples
        B = (X[i:i+batch_size],y[i:i+batch_size],seq_lengths[i:i+batch_size])
        yield B
