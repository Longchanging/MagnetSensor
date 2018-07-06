# coding:utf-8
'''
@time:    Created on  2018-07-03 17:05:39
@author:  Lanqing
@Func:    src.o7_baseline_LSTM_Org
'''

"""
LSTM for time series classification
This model takes in time series and class labels.
The LSTM models the time series. A fully-connected layer
generates an output to be classified with Softmax
"""

##### 加载参数，全局变量
import pickle
with open('config.pkl','rb') as f:  # Python 3: open(..., 'rb')
    train_batch_size, MAX_NB_VARIABLES, batch_size, NB_CLASS, \
    model_folder, train_tmp, epochs, units, n_splits = pickle.load(f)

import time

from sklearn.model_selection import ShuffleSplit, StratifiedKFold, train_test_split
from src.utils.lstm import Model, sample_batch

import numpy as np
import tensorflow as tf  # TF 1.1.0rc1

# train ,test,evaluation
test_ratio = 0.2
evaluation_ratio = 0.1

# train detail 
batch_size = 500
learning_rate = .005
max_iterations = 500  # 2000
hidden_size = 100  # memory
# other parameter
dropout = 0.8
num_layers = 3  # number of layers of stacked RNN's
max_grad_norm = 5  # maximum gradient norm during training
n_splits = 50

tf.logging.set_verbosity(tf.logging.ERROR)
# Random Seed for reproducibility
np.random.seed(1) 
tf.set_random_seed(1234)
a = tf.random_uniform([1])

after_pca_data = train_tmp
direc = train_tmp + '/keep-point/'
summaries_dir = train_tmp + '/keep-point/'
Model_folder = model_folder

# Set these directories

def train_test_evalation_split(data, label):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=test_ratio, random_state=0)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=evaluation_ratio, random_state=0)
    return X_train, X_test, y_train, y_test 

"""   Load the data   """
data = np.loadtxt(after_pca_data + 'After_pca_data.txt')
label = np.loadtxt(after_pca_data + 'After_pca_label.txt')
X_train, X_test_left, y_train, y_test_left = train_test_evalation_split(data, label)
X = X_train
y = y_train

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
# print(tf.summary)
saver = tf.train.Saver()
sess.run(model.init_op)
sess.run(a)

cost_train_ma = -np.log(1 / float(num_classes) + 1e-9)  # Moving average training cost
acc_train_ma = 0.0
try:         
    scores = []
    skf_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=10)
    # print(skf_cv) 
    i = 0
    for train_index, test_index in skf_cv.split(X, y):
    
        i += 1                                               
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Next line does the actual training
        for _ in range(4):
            X_train, y_train = sample_batch(X_train, y_train, batch_size)
            print(sess.run(next_batch))

        cost_train, acc_train, _ = sess.run([model.cost, model.accuracy, model.train_op], feed_dict={model.input: X_train, model.labels: y_train, model.keep_prob:dropout})
        cost_train_ma = cost_train_ma * 0.99 + cost_train * 0.01
        acc_train_ma = acc_train_ma * 0.99 + acc_train * 0.01
        
        # Evaluate validation performance
        cost_val, summ, acc_val = sess.run([model.cost, model.merged, model.accuracy], feed_dict={model.input: X_test, model.labels: y_test, model.keep_prob:1.0})
        print('Iter %d: COST %5.3f/%5.3f(%5.3f) -- Acc %5.3f/%5.3f(%5.3f)' % (i, cost_train, cost_val, cost_train_ma, acc_train, acc_val, acc_train_ma))
            
except KeyboardInterrupt:
    pass

epoch = float(i) * batch_size / N
print('Trained %.1f epochs, accuracy is %5.3f and cost is %5.3f' % (epoch, acc_val, cost_val))

cost_test, summ, acc_test = sess.run([model.cost, model.merged, model.accuracy], feed_dict={model.input: X_test_left, model.labels: y_test_left, model.keep_prob:1.0})
print('Test COST  %5.3f -- Test Acc %5.3f' % (cost_test, acc_test))

# Save the final model
saver.save(sess, Model_folder + 'model_final.ckpt')
