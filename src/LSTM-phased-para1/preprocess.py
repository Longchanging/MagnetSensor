# coding:utf-8
'''
@time:    Created on  2018-04-13 20:19:41
@author:  Lanqing
@Func:    realData_LSTM.algorithm
'''
from sklearn.model_selection import train_test_split
from config import *   
import numpy as np

def train_test_evalation_split(data, label):
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=test_ratio, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=evaluation_ratio, random_state=0)
    return X_train, X_val, X_test, y_train, y_val, y_test

def one_hot_coding(data):
    from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, LabelEncoder
    enc = LabelEncoder()
    out = enc.fit_transform(data)  
    return  out, enc

def min_max_scaler(train_data):
    # Min max scaler
    from sklearn import preprocessing
    XX = preprocessing.MinMaxScaler().fit(train_data)  
    train_data = XX.transform(train_data) 
    return train_data

def PCA(X):
    # reduce the input dimension
    if use_pca:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=saved_dimension_after_pca)
        X = pca.fit_transform(X)
    return X

def preprocess():
    
    # read concentrate data
    tmp = []
    label = []
    for category in different_category:
        file_ = np.loadtxt(after_fft_data + str(category) + '.txt', skiprows=M)
        # print('ok')
        tmp_label = [category] * len(file_)
        tmp.append(file_) 
        label += tmp_label
    data = np.vstack((tmp[0], tmp[1]))
    for i in range(2, len(different_category)):
        data = np.vstack((data, tmp[i]))
    del file_, tmp 
    print(data.shape)
    
    # One hot label
    print(len(label))
    label, enc = one_hot_coding(label)
    print(label)
    
    # min-max scaler
    data = min_max_scaler(data)
    
    # PCA , the location of these function should be reconsidered
    data = PCA(data)
    print(data[0:2], data.shape)
    
    np.savetxt(after_pca_data + 'After_pca_data.txt', data)
    np.savetxt(after_pca_data + 'After_pca_label.txt', label)
    return  

preprocess()
