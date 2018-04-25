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
    tmp, tmp_feature = [], []
    label = []

    # only read selected rows to process 

    for category in different_category:
        
        # load each category data and feature data
        file_ = np.loadtxt(after_fft_data + str(category) + '.txt', skiprows=M)
        file_numeric_feature = np.loadtxt(after_fft_data + str(category) + '_numeric_feature.txt', skiprows=M)
        tmp_label = [category] * len(file_)
        
        # generate label part and merge all
        tmp.append(file_) 
        tmp_feature.append(file_numeric_feature) 
        label += tmp_label
    
    # fetch all data array after fft
    data = np.vstack((tmp[0], tmp[1]))
    for i in range(2, len(different_category)):
        data = np.vstack((data, tmp[i]))
    del file_, tmp 
    print('All data after fft shape: \t', data.shape)
    
    # fetch all feature array no fft but feature_extraction
    data_feature = np.vstack((tmp_feature[0], tmp_feature[1]))
    for i in range(2, len(different_category)):
        data_feature = np.vstack((data_feature, tmp_feature[i]))
    print('All extracted feature shape: \t', data_feature.shape)
    
    # One hot label
    label, _ = one_hot_coding(label)  # not really 'one-hot',haha
    print('Label categories all: %d' % len(np.unique(label)))

    # merge data and numeric feature
    data = np.hstack((data, data_feature))
    print('Merge feature and data shape: \t', data.shape)
    
    # PCA , the location of these function should be reconsidered
    data = PCA(data)
    print('Data after pca shape: \t', data.shape)
    
    # min-max scaler
    data = min_max_scaler(data)
    #data_feature = min_max_scaler(data_feature)
    
    print('\n Shape info: \n final data: %d rows * %d cols' % (data.shape[0], data.shape[1]))
    print('final label: %d rows' % len(label))
    
    print('all data after pca saved to folder: %s'%after_pca_data)
    np.savetxt(after_pca_data + 'After_pca_data.txt', data)
    np.savetxt(after_pca_data + 'After_pca_label.txt', label)
    
    return  

preprocess()
