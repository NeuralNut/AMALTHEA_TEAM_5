# -*- coding: utf-8 -*-
"""
SETUP
"""
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import numpy.random as rnd
import os

# to make this notebook's output stable across runs
rnd.seed(42)

# To plot pretty figures
#%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "tensorflow"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

"""
Creating Your First Graph and Running It
"""

import tensorflow as tf

# running this code does not actually perform the computation. Instead a
# computational graph is created. In order to execute the computation, we must
# tell Tensorflow to decide which device will be used (Session()) initialize 
# the variables (x.initializer()) and run the computation (run())
x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y+y+2

# A clean way to start a session, initialize variables, and evaluate the graph
with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
    
# to avoid initializing variables, we could use tf.global_variables_initializer()
init = tf.global_variables_initializer() # prepare an init node

with tf.Session() as sess:
    init.run() #initialize variables
    result =f.eval()
    
"""
Managing Graphs
"""
# experimentation usually involves funning a command more than once. This leads
# to creating a default computation graph with duplicate nodes. We can resolve
# this by restering the kernel in Ipython or by tf.reset_default_graph()

x1 = tf.Variable(1)
x1.graph is tf.get_default_graph() #True

# temporarily make the new Graph the default graph using a with block
graph = tf.Graph()
with graph.as_default():
    x2= tf.Variable(2)
    
x2.graph is graph #True becuase temporary default graph
x2.graph is tf.get_default_graph() #False

"""
Lifecycle of a Node Value
    -how tf handles node dependencies
    -node values dropped between graph runs (except Variables)
    -variables start at initializer and end when session is closed
    -tf vs distributed tf
"""
# this model has x(w), y(x(w)), and z(x(w)) dependencies and the way we treat
# these dependencies in tf is important to our codes efficiency

w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

# calling the eval() method onto the variables y and z separately means that x 
# and w are evaluated in two graph runs - inefficient
with tf.Session() as sess:
    print(y.eval())
    print(z.eval())

# a better practice is to group y and z in one grpah run as follows
with tf.Session() as sess:
    y_val, z_val = sess.run([y, z])
    print(y_val)
    print(z_val)
    
"""
Linear Regression with Tensorflow
    -ops vs source ops
    -i/o are mutlidimensional arrays call tensors
    -performing computations with tensors
    -an example involving linear regression on Cali housing data
    - numpy and tf working compatibly
"""
import numpy as np
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing() # shorten tag for the data
m, n = housing.data.shape # declare data shape into variables
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data] # add bias unit


X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name="y")
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session() as sess:
    theta_value = theta.eval()
    y_value = y.eval()

"""
Batch Gradient Descent
    -manual computation of gradients vs tf's autodiff
    -important to normalize input feature vector else training slows
    -LOF: random_uniform(), assign(), StandardScaler [scikitlearn]
    -GD vs Momentum optimizers (tf makes easy to switch between them)
"""
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing() # shorten tag for the data
m, n = housing.data.shape # declare data shape into variables

# housing data is scaled using sklearn
housing_scaler = StandardScaler()
scaled_housing_data = housing_scaler.fit_transform(housing.data) #mean-centered then scaled by feature
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]# add bias unit

#define hyperparameters
n_epochs = 10000
learning_rate = 0.001

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name="y")

theta = tf.Variable(tf.random_uniform([n + 1, n], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")

#manual gradient computation
#gradients = 2/m * tf.matmul(tf.transpose(X), error) 

# gradients computed using autodiff
#gradients = tf.gradients(mse, [theta])[0]
#training_op = tf.assign(theta, theta - learning_rate * gradients)

# gradients computed using a tf optimizer method (GD or Momentum)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
#optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)

training_op = optimizer.minimize(mse)



init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range (n_epochs):
        if epoch % 100 == 0:
            print("Epoch ", epoch, "MSE=", mse.eval())
        sess.run(training_op)
        
    best_theta = theta.eval()

"""
Mini-Batch Gradient Descent (alternate ways to feed data to the optimzer)
    -relpace X an y at every iteration with the next mini-batch
    -simplest method with use of placeholder nodes
    -saving models and creating chekcpoints; Saver() declared after Variables()
   
"""
# illustration of placeholders
A = tf.placeholder(tf.float32, shape=(None, 3))
B = A + 5
with tf.Session() as sess:
    B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})
    B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8, 9]]})
    

# 
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import numpy.random as rnd


###############################################################################
#                                                                             #
#                          construction phase                                 #
#                                                                             #
###############################################################################
tf.reset_default_graph()

housing = fetch_california_housing() # shorten tag for the data
m, n = housing.data.shape # declare data shape into variables

# housing data is scaled using sklearn
housing_scaler = StandardScaler()
scaled_housing_data = housing_scaler.fit_transform(housing.data) #mean-centered then scaled by feature
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]# add bias unit

#define hyperparameters
learning_rate = 0.001
n_epochs = 100


X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n + 1, n], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
# optimizer = tf.train.AdamOptimizer()
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
saver = tf.train.Saver({"weights": theta})

###############################################################################
#                                                                             #
#                          execution phase                                    #
#                                                                             #
###############################################################################

def fetch_batch(epoch, batch_index, batch_size):
    rnd.seed(epoch * n_batches + batch_index) 
    indices = rnd.randint(m, size=batch_size)
    X_batch = scaled_housing_data_plus_bias[indices]
    y_batch = housing.target.reshape(-1, 1)[indices]
    return X_batch, y_batch

batch_size = 100
n_batches = int(np.ceil(m / batch_size))

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            save_path = saver.save(sess, "/tmp/my_model.ckpt")
        for batch_index in range(n_batches):           
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
#            if epoch % 100 == 0:
#                print("Epoch ", epoch, "MSE=", mse.eval())
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
    
    best_theta = theta.eval()
    save_path = saver.save(sess, "/tmp/my_model_final.ckpt")
print("Best theta: \n" , best_theta)

"""
Visualizing Graph and Training Curves Using TensorBoard
    -linear regression model trained by mini-batch gradient descent 
    -saving checkpoints at regular intervals
    -now time to plot our training curve
"""
tf.reset_default_graph()

from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

n_epochs = 1000
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

mse_summary = tf.summary.scalar('MSE', mse)
summary_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                summary_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()

summary_writer.flush()
summary_writer.close()
print("Best theta:")
print(best_theta)

"""
Name Scopes
    -used to clean up graphs with many nodes
"""

tf.reset_default_graph()

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

n_epochs = 1000
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
with tf.name_scope('loss') as scope:
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

mse_summary = tf.summary.scalar('MSE', mse)
summary_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())


n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                summary_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()

summary_writer.flush()
summary_writer.close()
print("Best theta:")
print(best_theta)


"""
Modularity p. 246
    -stay DRY (don't repeat yourself)
"""
# create 5 ReLU functions and output their sum by building a relu() function
def relu(X):
    w_shape = (int(X.get_shape()[1]),1) # return the shape of the weights
    w = tf.Variable(tf.random_normal(w_shape), name="weights") #init weights as random normal
    b = tf.Variable(0.0, name="bias") # make bias unit zero
    z = tf.add(tf.matmul(X,w), b, name="z") # compute the argument of the relu
    return tf.maximum(z, 0, name="relu") 
n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name="output")
summary_writer = tf.summary.FileWriter("tf_logs/relu1", tf.get_default_graph())
summary_writer.close()








