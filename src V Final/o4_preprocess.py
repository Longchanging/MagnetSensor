# coding:utf-8
'''
@time:    Created on  2018-04-13 20:19:41
@author:  Lanqing
@Func:    realData_LSTM.algorithm
'''

import pickle

##### 加载参数，全局变量
with open('config.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    
    dict_all_parameters = pickle.load(f)

    train_keyword = dict_all_parameters['train_keyword']
    train_folder = dict_all_parameters['train_folder']
    test_folder = dict_all_parameters['test_folder']
    predict_folder = dict_all_parameters['predict_folder']
    train_tmp = dict_all_parameters['train_tmp']
    train_keyword = dict_all_parameters['train_keyword'] 
    model_folder = dict_all_parameters['model_folder'] 
    NB_CLASS = dict_all_parameters['NB_CLASS'] 

    saved_dimension_after_pca = dict_all_parameters['saved_dimension_after_pca'] 
    sigma = dict_all_parameters['sigma'] 
    use_gauss = dict_all_parameters['use_gauss'] 
    use_pca = dict_all_parameters['use_pca'] 
    use_fft = dict_all_parameters['use_fft'] 
    whether_shuffle_train_and_test = dict_all_parameters['whether_shuffle_train_and_test'] 
    
from sklearn.externals import joblib
import numpy as np

def one_hot_coding(data, train_test_flag):
    from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, LabelEncoder
    enc = LabelEncoder()
    out = enc.fit_transform(data)  
    if train_test_flag == 'train':
        joblib.dump(enc, model_folder + "Label_Encoder.m")
    return  out, enc

def min_max_scaler(train_data):

    # Min max scaler
    from sklearn import preprocessing
    r, c = train_data.shape
    train_data = train_data.reshape([r * c, 1])

    XX = preprocessing.MinMaxScaler().fit(train_data)  
    train_data = XX.transform(train_data) 

    train_data = train_data.reshape([r, c])

    joblib.dump(XX, model_folder + "Min_Max.m")

    return train_data

def PCA(X):
    # reduce the input dimension
    if use_pca:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=saved_dimension_after_pca)
        X = pca.fit_transform(X)
        joblib.dump(pca, model_folder + "PCA.m")
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
    fft_data = vstack_list(tmp)
    data_feature = vstack_list(tmp_feature)
    print('All data after fft shape:', fft_data.shape, '\tLabel categories: %d' % len(np.unique(label)))
    return fft_data, data_feature, label

def preprocess(train_test_validation_flag , different_category, after_pca_data):
    
    fft_data, data_feature, label = get_data_label(after_pca_data, different_category) 
        
    if train_test_validation_flag == 'train' :
        
        fft_data = PCA(fft_data)
        fft_data = min_max_scaler(fft_data) 
        
        data_feature = min_max_scaler(data_feature)         
        
        #### 暂时苟合在一起， 最后处理、划分完训练、预测集再分开； 先FFT，后numberic
        data = np.hstack((data_feature, fft_data))
        
        label, _ = one_hot_coding(label, 'train')  # not really 'one-hot',haha
        
        
    print('final data: %d rows * %d cols,including fft %d cols, numberic %d cols ' % (data.shape[0], data.shape[1], fft_data.shape[1], data_feature.shape[1]))    
    print('all data after pca saved to folder: %s\n' % after_pca_data)
    np.savetxt(after_pca_data + 'After_pca_data.txt', data)
    np.savetxt(after_pca_data + 'After_pca_label.txt', label)

    return  

if __name__ == '__main__':
    
    # train
    preprocess('train', train_keyword, train_tmp)
