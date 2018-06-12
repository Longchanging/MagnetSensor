# coding:utf-8
'''
@time:    Created on  2018-04-13 18:18:44
@author:  Lanqing
@Func:    read data and preprocess
'''
import numpy as np
import pandas as pd
from o2_config import train_folder, test_folder, predict_folder, \
    train_tmp, test_tmp, predict_tmp, \
    train_keyword, predict_keyword, \
    window_length, overlap_window, \
    train_data_rate, \
    use_gauss, use_fft, sigma

def divide_files_by_name(folder_name, different_category):
    '''
        read all txt files and divide files into different parts
    '''
    import os
    dict_file = dict(zip(different_category, [[]] * len(different_category)))  # initial
    print('Processing files in folder: %s' % folder_name)
    for category in different_category:
        dict_file[category] = []  # Essential here
        for (root, _, files) in os.walk(folder_name):  # List all file names
            for filename in files:  # 这里很容易出问题，会读大目录下全部文件
                file_ = os.path.join(root, filename)
                if category in filename:
                    dict_file[category].append(file_)
    return dict_file

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
        if ',' in line and '2018' not in line:
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
        gaussian_X = scipy.ndimage.filters.gaussian_filter1d(X, sigma)
        return gaussian_X
    else:
        return X
    
def numericalFeather(singleColumn):
    # input a number list
    # return all feather such as max,min,square error,mean,total,every month 
    N1 = 1.0
    singleColumn = np.array(singleColumn) * N1
    medianL = np.median(singleColumn) * N1
    varL = np.var(singleColumn) * N1
    meanL = np.mean(singleColumn) * N1
    static = [medianL, varL, meanL]
    return np.array(static)

def preprocess_One_Array(array_, category, after_fft_data):
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

    # ##
    # Add information of original data 
    # ##
    while(i * overlap_window + window_length < rows):  # attention here

        tmp_window = array_[(i * overlap_window) : (i * overlap_window + window_length)]  # # main
        tmp_window = tmp_window.reshape([window_length * cols, 1])  # reshape
         
        # gauss filter
        tmp_window = gauss_filter(tmp_window, sigma)

        # numerical feature
        # numerical_feature_tmp = numericalFeather(tmp_window)
        numerical_feature_tmp = tmp_window

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

def read__data(input_folder, different_category, percent2read_afterPCA_data, after_fft_data_folder):
    '''
        1. loop all different files , read all into numpy array
        2. label all data
        3. construct train and test
    '''

    file_dict = divide_files_by_name(input_folder, different_category)
    a_sample_array = read_single_txt_file(file_dict[list(file_dict.keys())[0]][0]) 
    _, cols = a_sample_array.shape
    describe_tmp, len_list = [], []

    for category in different_category:
        
        file_array_one_category = np.array([[0]] * cols).T  # Initial, skill here
        for one_category_single_file in file_dict[category]:  # No worry "Ordered",for it is list

            file_array = read_single_txt_file(one_category_single_file)
            read_samples = int(percent2read_afterPCA_data * file_array.shape[0])
            print('only read %d samples ' % read_samples)
            file_array = file_array[:read_samples]            
            file_array_one_category = np.vstack((file_array_one_category, file_array))  
            
        print('%s category all: \t' % category, file_array_one_category[1], file_array_one_category.shape)   
        
        file_array_one_category = file_array_one_category[1:]  # exclude first line
        
        # 收集统计信息
        len_array = file_array_one_category.shape[0] * file_array_one_category.shape[1]
        len_list.append(len_array)
        describe_tmp.append(file_array_one_category.reshape([len_array, 1]))       
        
        # 预处理
        preprocess_One_Array(file_array_one_category, category, after_fft_data_folder)
    
    print('不同类型数据的长度', len_list)
    new_list = []
    min_len = min(len_list)
    for i in range(len(describe_tmp)):
        new_list.append(describe_tmp[i][:min_len])
    new_list = np.array(new_list).reshape([min_len, len(describe_tmp)])
    describe_tmp = pd.DataFrame(new_list, columns=different_category)
    print(describe_tmp.describe())
    
    return 

if __name__ == '__main__':
    
    # train and test
    read__data(train_folder, train_keyword, train_data_rate, train_tmp)

    # test
    # read__data(test_folder, test_keyword, 1, test_tmp)
    
    # predict
    read__data(predict_folder, predict_keyword, 1, predict_tmp)
