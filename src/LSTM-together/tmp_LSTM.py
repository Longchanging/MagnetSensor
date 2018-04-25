# coding:utf-8
'''
@time:    Created on  2018-04-14 01:27:13
@author:  Lanqing
@Func:    realData_LSTM.LSTM
'''
"""
LSTM for time series classification
This model takes in time series and class labels.
The LSTM models the time series. A fully-connected layer
generates an output to be classified with Softmax
"""
import time

from Model import Model, sample_batch, load_data
from config import *
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf  # TF 1.1.0rc1

tf.logging.set_verbosity(tf.logging.ERROR)
# Random Seed for reproducibility
np.random.seed(1) 
tf.set_random_seed(1234)
a = tf.random_uniform([1])

# Set these directories

def train_test_evalation_split(data, label):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=test_ratio, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=evaluation_ratio, random_state=0)
    return X_train, X_val, X_test, y_train, y_val, y_test

"""Load the data"""
data = np.loadtxt(after_pca_data + 'After_pca_data.txt')
label = np.loadtxt(after_pca_data + 'After_pca_label.txt')
X_train, X_val, X_test, y_train, y_val, y_test = train_test_evalation_split(data, label)

N, sl = X_train.shape
num_classes = len(np.unique(y_train))

"""Hyperparamaters"""
batch_size = batch_size
max_iterations = max_iterations  # 2000
dropout = dropout
config = {    'num_layers' :    num_layers,  # number of layers of stacked RNN's
              'hidden_size' :   hidden_size,  # memory cells in a layer
              'max_grad_norm' : max_grad_norm,  # maximum gradient norm during training
              'batch_size' :    batch_size,
              'learning_rate' : learning_rate,  # 0.005
              'sl':             sl,
              'num_classes':    num_classes}

epochs = np.floor(batch_size * max_iterations / N)
print('Train %.0f samples in approximately %d epochs and %d classes' % (N, epochs, num_classes))

# Instantiate a model
model = Model(config)

"""Session time"""
sess = tf.Session()  # Depending on your use, do not forget to close the session
writer = tf.summary.FileWriter(summaries_dir, sess.graph)  # writer for Tensorboard
saver = tf.train.Saver()
sess.run(model.init_op)
sess.run(a)

cost_train_ma = -np.log(1 / float(num_classes) + 1e-9)  # Moving average training cost
acc_train_ma = 0.0
try:
    for i in range(max_iterations):
        X_batch, y_batch = sample_batch(X_train, y_train, batch_size)
    
        # Next line does the actual training
        cost_train, acc_train, _ = sess.run([model.cost, model.accuracy, model.train_op], feed_dict={model.input: X_batch, model.labels: y_batch, model.keep_prob:dropout})
        cost_train_ma = cost_train_ma * 0.99 + cost_train * 0.01
        acc_train_ma = acc_train_ma * 0.99 + acc_train * 0.01
        if i % 30 == 1:
        # Evaluate validation performance
            X_batch, y_batch = sample_batch(X_val, y_val, batch_size)
            cost_val, summ, acc_val = sess.run([model.cost, model.merged, model.accuracy], feed_dict={model.input: X_batch, model.labels: y_batch, model.keep_prob:1.0})
            print('At %5.0f/%5.0f: COST %5.3f/%5.3f(%5.3f) -- Acc %5.3f/%5.3f(%5.3f)' % (i, max_iterations, cost_train, cost_val, cost_train_ma, acc_train, acc_val, acc_train_ma))
            # Write information to TensorBoard
            writer.add_summary(summ, i)
            writer.flush()
            
except KeyboardInterrupt:
    pass

  
epoch = float(i) * batch_size / N
print('Trained %.1f epochs, accuracy is %5.3f and cost is %5.3f' % (epoch, acc_val, cost_val))

cost_test, summ, acc_test = sess.run([model.cost, model.merged, model.accuracy], feed_dict={model.input: X_test, model.labels: y_test, model.keep_prob:1.0})
print('Test COST  %5.3f -- Test Acc %5.3f' % (cost_test, acc_test))

# Save the final model
saver.save(sess, Model_folder + 'model_final.ckpt')
