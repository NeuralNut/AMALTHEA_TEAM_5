#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 11:52:00 2017

@author: emilyjensen

Extracts classes 2 and 4 of Alaska dataset

Run this file and then run the main.py file from the second cell to the end (starting at hyperparameters)
"""

#import packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from datetime import datetime
import io


# import user-defined functions
from model import Model
from functions import load_data
from functions import create_batches


tf.reset_default_graph()


now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs_Emily"
logdir = "{}/run-{}/".format(root_logdir, now)

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

# set directory for Alaska data
direc = '/home/emilyjensen/repos/project/shared_repo/AMALTHEA_TEAM_5/infra_tflow/src/data/'
dataset = 'alaska_data'
datadir = direc + '/' + dataset + '/' + dataset
# load data
data_train = np.loadtxt(datadir+'_TRAIN',delimiter=',').astype(np.float32)
data_test = np.loadtxt(datadir+'_TEST',delimiter=',').astype(np.float32)

# only keep classes 2 and 4
data_train = np.append(data_train[data_train[:,0] == 2],data_train[data_train[:,0] == 4],axis=0)
data_test = np.append(data_test[data_test[:,0] == 2],data_test[data_test[:,0] == 4],axis=0)


N = data_train.shape[0]
ratio = 0.8
ratio = int(ratio*N)
np.random.shuffle(data_train)
np.random.shuffle(data_test)
y_train = (data_train[:ratio,0]).astype(np.int32)
y_val = (data_train[ratio:,0]).astype(np.int32)
y_test = (data_test[:,0]).astype(np.int32)
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

# re-index classes 2 and 4 to 0 and 1, respectively
for i in [y_test,y_train,y_val]:
    i[i == 2] = 0
    i[i == 4] = 1
# now run main.py until the end (starting from second cell)