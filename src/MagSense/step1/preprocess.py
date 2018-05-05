# coding:utf-8
'''
@time:    Created on  2018-04-13 20:19:41
@author:  Lanqing
@Func:    realData_LSTM.algorithm
'''
from sklearn.externals import joblib
import numpy as np
from step1.config import use_pca, saved_dimension_after_pca, \
    train_Model_folder, \
    train_after_fft_data, trainTestdifferent_category, train_after_pca_data, \
    test_after_fft_data, test_after_pca_data, \
    predict_fft_data, predict_different_category, predict_pca_data


def one_hot_coding(data):
    from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, LabelEncoder
    enc = LabelEncoder()
    out = enc.fit_transform(data)  
    joblib.dump(enc, train_Model_folder + "Label_Encoder.m")
    return  out, enc

def min_max_scaler(train_data):
    # Min max scaler
    from sklearn import preprocessing
    XX = preprocessing.MinMaxScaler().fit(train_data)  
    train_data = XX.transform(train_data) 
    joblib.dump(XX, train_Model_folder + "Min_Max.m")
    return train_data

def PCA(X):
    # reduce the input dimension
    if use_pca:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=saved_dimension_after_pca)
        X = pca.fit_transform(X)
        joblib.dump(pca, train_Model_folder + "PCA.m")
    return X

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

def get_data_label(after_fft_data, different_category):
    ''' 
        Get all data and label for train,test or prediction
    '''
    tmp, tmp_feature, label = [], [], []
    for category in different_category:  
        
        # load each category data and feature data
        file_ = np.loadtxt(after_fft_data + str(category) + '.txt')
        file_numeric_feature = np.loadtxt(after_fft_data + str(category) + '_numeric_feature.txt')
        tmp_label = [category] * len(file_)
        
        # generate label part and merge all
        tmp.append(file_) 
        tmp_feature.append(file_numeric_feature) 
        label += tmp_label
    
    # fetch all data array after fft
    data = vstack_list(tmp)
    data_feature = vstack_list(tmp_feature)
    data = np.hstack((data, data_feature))
    print('All data after fft shape: \t', data.shape, 'Label categories: %d' % len(np.unique(label)))
    
    return data, label

def preprocess(train_test_validation_flag, after_fft_data, different_category, after_pca_data):
    
    data, label = get_data_label(after_fft_data, different_category) 
        
    if train_test_validation_flag == 'train' :
        data = PCA(data)
        data = min_max_scaler(data) 
        label, _ = one_hot_coding(label)  # not really 'one-hot',haha
        
    elif train_test_validation_flag == 'test' or train_test_validation_flag == 'predict':
        
        from sklearn.externals import joblib
        label_encoder = joblib.load(train_Model_folder + "Label_Encoder.m")
        min_max = joblib.load(train_Model_folder + "Min_Max.m")
        pca = joblib.load(train_Model_folder + "PCA.m")
        
        data = pca.transform(data)
        data = min_max.transform(data) 
        
        if train_test_validation_flag == 'test':
            label = label_encoder.transform(label)  # not really 'one-hot',haha
        elif train_test_validation_flag == 'predict' :
            label, _ = one_hot_coding(label)  # not really 'one-hot',haha
    
    print('\n Shape info: \n final data: %d rows * %d cols' % (data.shape[0], data.shape[1]))
    print('final label: %d rows' % len(label))
    
    print('all data after pca saved to folder: %s' % after_pca_data)
    np.savetxt(after_pca_data + 'After_pca_data.txt', data)
    np.savetxt(after_pca_data + 'After_pca_label.txt', label)
    
    return  

if __name__ == '__main__':
    
    # train
    preprocess('train', train_after_fft_data, trainTestdifferent_category, train_after_pca_data)
    
    # test
    # preprocess('test', test_after_fft_data, trainTestdifferent_category, test_after_pca_data)
    
    # predict
    preprocess('predict', predict_fft_data, predict_different_category, predict_pca_data)
