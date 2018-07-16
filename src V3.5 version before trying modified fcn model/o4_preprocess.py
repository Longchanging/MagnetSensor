# coding:utf-8
'''
@time:    Created on  2018-04-13 20:19:41
@author:  Lanqing
@Func:    realData_LSTM.algorithm
'''
from sklearn.externals import joblib
import numpy as np
from imp import reload
import src.o2_config
from src.o2_config import use_pca, saved_dimension_after_pca, train_keyword, predict_keyword , \
    model_folder, train_tmp, test_tmp, predict_tmp, test_keyword
reload(src.o2_config)

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
    data = vstack_list(tmp)
    data_feature = vstack_list(tmp_feature)
    data = np.hstack((data, data_feature))
    print('All data after fft shape:', data.shape, '\tLabel categories: %d' % len(np.unique(label)))
    return data, label

def preprocess(train_test_validation_flag , different_category, after_pca_data):
    
    data, label = get_data_label(after_pca_data, different_category) 
        
    if train_test_validation_flag == 'train' :
        data = PCA(data)
        data = min_max_scaler(data) 
        label, _ = one_hot_coding(label, 'train')  # not really 'one-hot',haha
        
    elif train_test_validation_flag == 'test' or train_test_validation_flag == 'predict':
        
        from sklearn.externals import joblib
        label_encoder = joblib.load(model_folder + "Label_Encoder.m")
        min_max = joblib.load(model_folder + "Min_Max.m")
        # pca = joblib.load(model_folder + "PCA.m")
        
        if use_pca:
            pca = joblib.load(model_folder + "PCA.m")
            data = pca.transform(data)
        
        # reshape and min-max
        train_max = float(min_max.data_max_)
        train_min = float(min_max.data_min_)

        # test or new data
        r, c = data.shape
        data = data.reshape([r * c, 1])

        # find the max and min of train and test data
        test_max = np.max(data)
        test_min = np.min(data)
        all_max = train_max if train_max >= test_max else test_max
        all_min = train_min if train_min <= test_min else test_min

        print('Check max and min: train_max,train_min,test_max,test_min,all_max,all_min')

        print(train_max, train_min, '\t', test_max, test_min, '\t', all_max, all_min)
        data = (data - all_min) / (all_max - all_min)
        data = data.reshape([r, c])
        
        if train_test_validation_flag == 'test':
            label = label_encoder.transform(label)  # not really 'one-hot',haha
        elif train_test_validation_flag == 'predict' :
            label, _ = one_hot_coding(label, 'predict')  # not really 'one-hot',haha
    
    print('final data: %d rows * %d cols' % (data.shape[0], data.shape[1]))    
    print('all data after pca saved to folder: %s\n' % after_pca_data)
    np.savetxt(after_pca_data + 'After_pca_data.txt', data)
    np.savetxt(after_pca_data + 'After_pca_label.txt', label)

    return  

if __name__ == '__main__':
    
    # train
    preprocess('train', train_keyword, train_tmp)

    # test
    preprocess('test', test_keyword, test_tmp)
        
    # predict
    preprocess('predict', predict_keyword, predict_tmp)
