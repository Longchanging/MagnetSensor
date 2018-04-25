# coding:utf-8
'''
@time:    Created on  2018-04-18 23:16:08
@author:  Lanqing
@Func:    testFCN.prepare_data
'''
from config import *
import numpy as np

def train_test_evalation_split(data, label):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=test_ratio, random_state=0, shuffle=True)
    return X_train, X_test, y_train, y_test 

def reshape_X(X, batch_size):
    from math import floor
    r, c = X.shape
    batch_num = floor(r / batch_size)
    # print(batch_num, batch_size * batch_num)
    if r > 1 and c > 1:
        X_reshaped = X[:(batch_size * batch_num)].reshape([batch_num, c, batch_size])
    else:
        X_reshaped = X[:batch_num]
    # print(X_reshaped.shape)
    return X_reshaped, batch_num

def fetch_batch_data(data, n_classes, batch_size):
    r, c = data.shape
    data_list = []
    label_list = []
    for i in range(n_classes):  # excluded label
        print('Processing class %d' % i)
        loc = np.where(data[:, -1] == i)
        data_ = data[loc][:, :-1]  # extract data of class i
        data_class_i, batch_num = reshape_X(data_, batch_size)
        label_class_i = np.ones([batch_num, 1]) * int(data[loc][0, -1])
        print('label shape: \t', label_class_i.shape, '\t label shape: \t', data_class_i.shape)
        data_list.append(data_class_i) 
        label_list.append(label_class_i)
    return data_list, label_list

def vstack_list(tmp):
    data = np.vstack((tmp[0], tmp[1]))
    for i in range(2, len(tmp)):
        data = np.vstack((data, tmp[i]))
    return data

# read data
data = np.loadtxt(after_pca_data + 'After_pca_data.txt', skiprows=M)
label = np.loadtxt(after_pca_data + 'After_pca_label.txt', skiprows=M)
print(data.shape,label.shape)

#data = data.reshape([len(data), 1])
label = label.reshape([len(label), 1])

n_classes = len(np.unique(label))
data = np.hstack((data, label))
print(data.shape)

data_list, label_list = fetch_batch_data(data, n_classes, batch_size)
data = vstack_list(data_list) 
label = vstack_list(label_list) 
print(label)
print('label shape: \t', label.shape, '\t data shape: \t', data.shape)

# split
X_train, X_test, y_train, y_test = train_test_evalation_split(data, label)

# ''' Save the datasets '''
print("Train dataset : ", X_train.shape, y_train.shape)
print("Test dataset : ", X_test.shape, y_test.shape)
print("Train dataset metrics : ", X_train.mean(), X_train.std())
print("Test dataset : ", X_test.mean(), X_test.std())
print("Nb classes : ", len(np.unique(y_train)))

# print(X_train, X_train.shape)
np.save(processed_folder + 'X_train.npy', X_train)
np.save(processed_folder + 'y_train.npy', y_train)
np.save(processed_folder + 'X_test.npy', X_test)
np.save(processed_folder + 'y_test.npy', y_test)
