# coding:utf-8
'''
@time:    Created on  2018-04-13 18:18:44
@author:  Lanqing
@Func:    read data and preprocess
'''
from config import *
import numpy as np

def read_single_txt_file(single_file_name):
    '''
    input data format:
        1. each TXT is a sample file with label
        2. each file include sample time 
        3. each line in a file is a sample 
    '''    
    file_list = []
    fid = open(single_file_name, 'r')
    for line in fid:
        line = line.strip('\n')
        if ',' in line:
            line = line.split(',')[:-1]  # exclude the last comma
            file_list.append(line)
    return np.array(file_list).astype(int)

def fft_transform(vector):
    if use_fft:
        transformed = np.fft.fft(vector)  # FFT
        transformed = transformed.reshape([transformed.shape[0] * transformed.shape[1], 1])  # reshape 
        return transformed.real
    else:
        return vector

def gauss_filter(X, sigma):
    if use_gauss:
        import scipy.ndimage
        gaussian_X = scipy.ndimage.filters.gaussian_filter(X, sigma)
        return gaussian_X
    else:
        return X
    
def numericalFeather(singleColumn):
    ''' 
        function : process numerical data
        usage:  singleColumn -> list or list/1-d array like
                return -> various statistical feathers
                
        Attention:
                In order to distinguish when One-Hot coding ,between ctgry and numeric,must * 1.0 here
               
    '''
    # input a number list
    # return all feather such as max,min,square error,mean,total,every month 
    
    N1 = 1.0
    
    countL = len(singleColumn) * N1
    singleColumn = np.array(singleColumn) * N1
    maxL = np.max(singleColumn) * N1
    minL = np.min(singleColumn) * N1
    medianL = np.median(singleColumn) * N1
    varL = np.var(singleColumn) * N1
    totalL = np.sum(singleColumn) * N1
    meanL = np.mean(singleColumn) * N1
    static = [medianL, varL, meanL]
    return np.array(static)

def preprocess_One_Array(array_, category):
    '''
        1. Process after "vstack" 
        2. receive an file array and corresponding category
        3. return a clean array
    '''    
    # split window
    # In order to reduce calculate cost, consider just using array operations.
    final_list = []
    numerical_feature_list = []
    rows, cols = array_.shape
    i = 0
    while(i * overlap_window + window_length < rows):  # attention here
        tmp_window = array_[(i * overlap_window) : (i * overlap_window + window_length)]  # # main
        tmp_window = tmp_window.reshape([window_length * cols, 1])  # reshape
         
        # gauss filter
        tmp_window = gauss_filter(tmp_window, sigma)
        
        # numerical feature
        numerical_feature_tmp = numericalFeather(tmp_window)
        
        # fft process
        tmp_window = fft_transform(tmp_window)
        
        final_list.append(tmp_window)
        numerical_feature_list.append(numerical_feature_tmp)
        i += 1
        
    final_array = np.array(final_list)
    numerical_feature_list = np.array(numerical_feature_list)

    final_array = final_array.reshape([final_array.shape[0], final_array.shape[1]])
    numerical_feature_array = numerical_feature_list.reshape([numerical_feature_list.shape[0], numerical_feature_list.shape[1]])
    print('%s final shape: \t' % category, final_array.shape)
    
    np.savetxt(after_fft_data + str(category) + '.txt', final_array) 
    np.savetxt(after_fft_data + str(category) + '_numeric_feature.txt', numerical_feature_array) 
    
    return

def read_data():
    '''
        1. loop all different files , read all into numpy array
        2. label all data
        3. construct train and test
    '''

    file_array = read_single_txt_file(file_test)     
    read_samples = int(percent2read_afterPCA_data * file_array.shape[0])
    
    print('only read %d samples ' % read_samples)
    file_array = file_array[:read_samples]
    preprocess_One_Array(file_array, category)
    
    return 

read_data() 
