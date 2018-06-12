# coding:utf-8
'''
@time:    Created on  2018-04-18 23:16:08
@author:  Lanqing
@Func:    testFCN.prepare_data
'''
from o2_config import train_tmp, test_tmp, predict_tmp, batch_size, test_ratio,\
     evaluation_ratio,whether_shuffle_train_and_test,train_tmp_test
import numpy as np

def train_test_evalation_split(data, label): 
    '''
    split train and test
    :param data: train data
    :param label: train label
    '''
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=test_ratio, random_state=0, shuffle=whether_shuffle_train_and_test)
    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=evaluation_ratio, random_state=0, shuffle=whether_shuffle_train_and_test)
    return X_train, X_test, X_validate, y_train, y_test , y_validate

def reshape_X(X, batch_size):
    '''
    Reshape array X  to  units * batch_size
    :param X: array
    :param batch_size: user defined batch size
    '''
    from math import floor
    r, c = X.shape
    batch_num = floor(r / batch_size)

    if r > 1 and c > 1:
        X_reshaped = X[:(batch_size * batch_num)].reshape([batch_num, c, batch_size])
    else:
        X_reshaped = X[:batch_num]
    return X_reshaped, batch_num

def fetch_batch_data(data, n_classes, batch_size):
    '''
    Use the reshape function and process data 
    :param data: only data not including labels
    :param n_classes: num of classes
    :param batch_size: user defined batch_size 
    '''
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
    '''
    vstack a list of several array, and concentrate the arrays
    :param tmp: a list of arrays
    '''
    if len(tmp) > 1:
        data = np.vstack((tmp[0], tmp[1]))
        for i in range(2, len(tmp)):
            data = np.vstack((data, tmp[i]))
    else:
        data = tmp[0]    
    return data

def main_prepare(after_pca_data):
    '''
    read a folder and prepare the data in a folder
    :param after_pca_data: folder name (not including the file name)
    '''
    # read data
    data = np.loadtxt(after_pca_data + 'After_pca_data.txt')
    label = np.loadtxt(after_pca_data + 'After_pca_label.txt')
    
    label = label.reshape([len(label), 1])
    n_classes = len(np.unique(label))
    data = np.hstack((data, label))
    
    data_list, label_list = fetch_batch_data(data, n_classes, batch_size)
    data = vstack_list(data_list) 
    label = vstack_list(label_list) 
    print('Prepared label shape: \t', label.shape, '\t data shape: \t', data.shape)
    
    return data, label

def train_test():
    '''
    main function to prepare data used in train and evaluate and test the model
    :param train_test_data_folder: data folder that stores the processed data
    '''
    # read data from train_test_data_folder + 'after_pca_data.txt' / 'after_pca_label.txt'
    data, label = main_prepare(train_tmp)
    
    # split
    X_train, X_test, X_validate, y_train, y_test , y_validate = train_test_evalation_split(data, label)
    
    # Save the datasets 
    print("Train dataset : ", X_train.shape, y_train.shape)
    print("Test dataset : ", X_test.shape, y_test.shape)
    print("Train dataset metrics : ", X_train.mean(), X_train.std())
    print("Test dataset : ", X_test.mean(), X_test.std())
    print("Nb classes : ", len(np.unique(y_train)))
    
    # print(X_train, X_train.shape)
    np.save(train_tmp + 'X_train.npy', X_train)
    np.save(train_tmp + 'y_train.npy', y_train)
    np.save(train_tmp + 'X_test.npy', X_validate)
    np.save(train_tmp + 'y_test.npy', y_validate)
    np.save(train_tmp_test + 'X_test.npy', X_test)
    np.save(train_tmp_test + 'y_test.npy', y_test)

    return

def test():
    '''
    main function to prepare data used in prediction
    :param predict_data_folder: data folder that stores the prediction processes data
    '''
    # read data from train_test_data_folder + 'after_pca_data.txt' / 'after_pca_label.txt'
    data, label = main_prepare(test_tmp)
    np.save(test_tmp + 'X_test.npy', data)
    np.save(test_tmp + 'y_test.npy', label)

    return

def predict():
    '''
    main function to prepare data used in prediction
    :param predict_data_folder: data folder that stores the prediction processes data
    '''
    # read data from train_test_data_folder + 'after_pca_data.txt' / 'after_pca_label.txt'
    data, label = main_prepare(predict_tmp)
    np.save(predict_tmp + 'X_test.npy', data)
    np.save(predict_tmp + 'y_test.npy', label)    
    return

if __name__ == '__main__':
    train_test() 
    test()
    predict()
