# coding:utf-8
'''
@time:    Created on  2018-04-13 20:19:41
@author:  Lanqing
@Func:    realData_LSTM.algorithm
'''
from config import *   
import numpy as np

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
    tmp, tmp_feature = [], []
    label = []

    # load each category data and feature data
    file_ = np.loadtxt(after_fft_data + str(category) + '.txt', skiprows=M)
    file_numeric_feature = np.loadtxt(after_fft_data + str(category) + '_numeric_feature.txt', skiprows=M)
    tmp_label = [category] * len(file_)
        
    # One hot label
    label, _ = one_hot_coding(tmp_label)  # not really 'one-hot',haha
    print('Label categories all: %d' % len(np.unique(label)))

    # merge data and numeric feature
    data = np.hstack((file_, file_numeric_feature))
    print('Merge feature and data shape: \t', data.shape)
    
    # PCA , the location of these function should be reconsidered
    data = PCA(data)
    print('Data after pca shape: \t', data.shape)
    
    # min-max scaler
    data = min_max_scaler(data)
    
    print('\nShape info: \nfinal data: %d rows * %d cols' % (data.shape[0], data.shape[1]))
    print('final label: %d rows' % len(label))
    
    print('all data after pca saved to folder: %s' % after_pca_data)
    np.savetxt(after_pca_data + 'After_pca_data.txt', data)
    np.savetxt(after_pca_data + 'After_pca_label.txt', label)
    
    return  

preprocess()
