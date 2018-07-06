# coding:utf-8
'''
@time:    Created on  2018-04-13 18:18:44
@author:  Lanqing
@Func:    read data and preprocess
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
    sample_rate = dict_all_parameters['sample_rate'] 
    saved_dimension_after_pca = dict_all_parameters['saved_dimension_after_pca'] 
    sigma = dict_all_parameters['sigma'] 
    use_gauss = dict_all_parameters['use_gauss'] 
    use_pca = dict_all_parameters['use_pca'] 
    use_fft = dict_all_parameters['use_fft'] 
    overlap_window = dict_all_parameters['overlap_window'] 
    window_length = dict_all_parameters['window_length'] 
    whether_shuffle_train_and_test = dict_all_parameters['whether_shuffle_train_and_test'] 
    
import numpy as np
import pandas as pd

def divide_files_by_name(folder_name, different_category):
    '''
        read all txt files and divide files into different parts
    '''
    import os
    dict_file = dict(zip(different_category, [[]] * len(different_category)))  # initial
    print('Processing files in folder: %s' % folder_name)
    
    ##### 加入异常处理逻辑，即： 如果关键词找不到匹配的文件，舍弃这个category
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
        4. 加入 采样控制
    '''    
    file_list = []
    fid = open(single_file_name, 'r')
    
    ###### 读取数据同时进行采样
    
    count_line = 0
    for line in fid:
        line = line.strip('\n')
        count_line += 1
        if ',' in line and '2018' not in line and (count_line % sample_rate == 0):
            line = line.split(',')[0]  # exclude the last comma
            file_list.append(line)
            
    read_array = np.array(file_list).astype(int)
            
    file_name = single_file_name.split('/')[-1]
    print('%s total %d seconds, sample %d points,sample rate: per %d ms' % (file_name, int(count_line / 1000), int(len(file_list)), sample_rate))
    
    return read_array

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
    
    print('%s final shape: %d * %d' % (category, final_array.shape[0], final_array.shape[1] * 2))
    
    np.savetxt(after_fft_data + str(category) + '.txt', final_array) 
    np.savetxt(after_fft_data + str(category) + '_numeric_feature.txt', numerical_feature_array) 
    
    return

def read__data(input_folder, different_category, percent2read_afterPCA_data, after_fft_data_folder):
    '''
        1. loop all different files , read all into numpy array
        2. label all data
        3. construct train and test
        4. 完成修改，默认读入一个一维的矩阵，降低采样率
    '''

    file_dict = divide_files_by_name(input_folder, different_category)
    # a_sample_array = read_single_txt_file(file_dict[list(file_dict.keys())[0]][0]) 
    # _, cols = a_sample_array.shape
    describe_tmp, len_list = [], []

    for category in different_category:
        
        # file_array_one_category = np.array([[0]] * cols).T  # Initial, skill here
        file_array_one_category = np.array([0])
        
        for one_category_single_file in file_dict[category]:  # No worry "Ordered",for it is list

            file_array = read_single_txt_file(one_category_single_file)
            read_samples = int(percent2read_afterPCA_data * file_array.shape[0])
            # print('only read %d samples ' % read_samples)
            file_array = file_array[:read_samples]
            file_array = file_array.reshape([len(file_array), 1])    
            # print(file_array.shape)
                    
            file_array_one_category = np.vstack((file_array_one_category, file_array))  
            
        file_array_one_category = file_array_one_category[1:]  # exclude first line
        print('%s category all: \t' % category, file_array_one_category[1], file_array_one_category.shape)   
        
        
        # 收集统计信息
        len_array = file_array_one_category.shape[0] * file_array_one_category.shape[1]
        len_list.append(len_array)
        describe_tmp.append(file_array_one_category.reshape([len_array, 1]))       
        
        # 预处理
        preprocess_One_Array(file_array_one_category, category, after_fft_data_folder)
    
    return 

if __name__ == '__main__':
    
    # train and test
    read__data(train_folder, train_keyword, train_data_rate, train_tmp)
