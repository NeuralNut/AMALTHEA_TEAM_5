#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 10:24:15 2017

@author: emilyjensen

functions for loading feature data
"""

import numpy as np
import os
import functions as f
import tensorflow as tf

# Define function for loading and splitting data into relevant sets
def load_data(direc,ratio,dataset_name):
    """Input:
        direc: location of the data archive
        ratio: ratio to split training set into training and validation
        dataset: name of the dataset in the archive"""

    # Load testing data
    # Make an empty array with the correct shape
    datadir = direc + '/' + dataset_name + '/' + dataset_name + '_TEST'
    file_names = [name for name in os.listdir(datadir)]
    num_features = len(file_names)
    file_names.sort()
    temp = np.loadtxt(datadir + '/' + file_names[0],delimiter=',').astype(np.float32)
    N,sl = temp.shape
    data_test = np.zeros([N,sl,num_features])
    data_test[:,:,0] = temp
    for i in range(1,num_features):
        # load the individual feature file
        temp = np.loadtxt(datadir + '/' + file_names[i],delimiter=',').astype(np.float32)
        # paste it into the appropriate layer of the array
        data_test[:,:,i] = temp

    # Load training data
    # Make an empty array with the correct shape
    datadir = direc + '/' + dataset_name + '/' + dataset_name + '_TRAIN'
    file_names = [name for name in os.listdir(datadir)]
    file_names.sort()
    num_features = len(file_names)
    temp = np.loadtxt(datadir + '/' + file_names[0],delimiter=',').astype(np.float32)
    N,sl = temp.shape
    data_train = np.zeros([N,sl,num_features])
    data_train[:,:,0] = temp
    for i in range(1,num_features):
        # load the individual feature file
        temp = np.loadtxt(datadir + '/' + file_names[i],delimiter=',').astype(np.float32)
        # paste it into the appropriate layer of the array
        data_train[:,:,i] = temp

    # Divide training set into training and validation sets
    # First column of data is class number
    N = data_train.shape[0]
    ratio = int(ratio*N)
    np.random.shuffle(data_train)
    np.random.shuffle(data_test)
    
    #split labels based on ratio
    y_train = (data_train[:ratio,0,0]).astype(np.int32)
    y_val = (data_train[ratio:,0,0]).astype(np.int32)
    y_test = (data_test[:,0,0]).astype(np.int32)
    
    # 0 index all labels
    nb_classes = len(np.unique(y_test))
    y_train = ((y_train - y_train.min())/(y_train.max()-y_train.min())*(nb_classes-1)).astype(np.int32)
    y_test = ((y_test - y_test.min())/(y_test.max()-y_test.min())*(nb_classes-1)).astype(np.int32)
    y_val = ((y_val - y_val.min())/(y_val.max()-y_val.min())*(nb_classes-1)).astype(np.int32)
    
    #find the length of each sequence
    data_seq = f.find_seq_lengths(data_train[:,1:,0])
    
    #split data based on ratio
    X_train = data_train[:ratio,1:,:]
    train_seq = data_seq[:ratio]
    X_val = data_train[ratio:,1:,:]
    val_seq = data_seq[ratio:]
    X_test = data_test[:,1:,:]
    test_seq = f.find_seq_lengths(data_test[:,1:,0])
    
    # z-normalize the test/train/val data
    X_train = (X_train - X_train.mean())/X_train.std()
    X_test = (X_test - X_test.mean())/X_test.std()
    X_val = (X_val - X_val.mean())/X_test.std()
    
    return X_train,train_seq,X_val,val_seq,X_test,test_seq,y_train,y_val,y_test