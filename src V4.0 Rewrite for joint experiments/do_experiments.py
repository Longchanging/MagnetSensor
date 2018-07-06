# coding:utf-8

########  DO EXPERIMENTS

import pickle
import time
from src.o1_setup import generate_configs

####### 存储所有参数并写入文件
model_folder_ = '../data/model/'
train_info_file = model_folder_ + 'train_info.txt'
dict_all_parameters = {}
train_info = {}

####### 定义需要变更的参数

train_folders = {   'platform':['mac_', 'shenzhou_', 'hp_', 'windows_'],
                    'apps':['05_work_word', '06_work_excel', '07_work_ppt', \
                            '08_social_wechat', '09_social_qq', \
                            '13_game_zuma', '14_game_candy', '15_game_minecraft', \
                            '16_picture_win3d', '17_chrome_surfing', '18_firefox_surfing', \
                            '19_chrome_gmail_work', '20_chrome_twitter', \
                            '22_chrome_amazon', '23_chrome_agar'],
                    'users':['lanqing', 'panhao', 'wangzhong', 'fangliang', 'zhoujie', 'weilun', 'yeqi']}

# 采样
sample_rate = 10  # 单位是毫秒 ，>=1
epochs, n_splits = 2 , 10  # 10折交叉验证和epoch数量固定
train_data_rate = 0.03  # 使用处理后数据比例，用于减小训练数据使用的样本数(训练预测阶段)
train_batch_size = 500

dict_all_parameters['sample_rate'] = sample_rate
dict_all_parameters['epochs'] = epochs
dict_all_parameters['n_splits'] = n_splits
dict_all_parameters['train_data_rate'] = train_data_rate
dict_all_parameters['train_batch_size'] = train_batch_size


# 处理
saved_dimension_after_pca, sigma = 20, 500 
use_gauss, use_pca, use_fft = True, False, True  # True
whether_shuffle_train_and_test = True

dict_all_parameters['saved_dimension_after_pca'] = saved_dimension_after_pca

dict_all_parameters['sigma'] = sigma
dict_all_parameters['use_gauss'] = use_gauss
dict_all_parameters['use_pca'] = use_pca
dict_all_parameters['use_fft'] = use_fft
dict_all_parameters['whether_shuffle_train_and_test'] = whether_shuffle_train_and_test

# 训练
test_ratio, evaluation_ratio = 0.2, 0.1  # 划分训练、测试、验证集

dict_all_parameters['test_ratio'] = test_ratio
dict_all_parameters['evaluation_ratio'] = evaluation_ratio

window_length_list = [2, 5, 10, 20, 50, 100, 500]  # 窗口大小
batch_size_list = [2, 5, 10, 20, 50]  # 训练 batch大小
units_list = [2, 5, 10, 20, 50, 200]  # int(MAX_NB_VARIABLES / 2)

i = 0

####### 存储相关信息
import os
if os.path.exists(train_info_file):
    os.remove(train_info_file)
    
fid = open(train_info_file, 'a')
fid.write('Index,totalRunTime,dataSet,CLASS,sample_rate,train_data_rate,window_length,batch_size,units,MAX_NB_VARIABLES,\
    knn_acc,rf_acc,time_tr,lstm_acc,time_lstm,fcn_acc,time_fcn')
fid.write('\n')
fid.close()
                
###### 循环遍历
for train_folder in train_folders:
    
    ##### 生成固定参数
    train_keyword, train_folder, test_folder, predict_folder, train_tmp, test_tmp, predict_tmp, \
    train_tmp_test, model_folder, NB_CLASS = generate_configs(train_folders, train_folder)

    dict_all_parameters['train_keyword'] = train_keyword
    dict_all_parameters['train_folder'] = train_folder
    dict_all_parameters['test_folder'] = test_folder
    dict_all_parameters['predict_folder'] = predict_folder
    dict_all_parameters['train_tmp'] = train_tmp
    dict_all_parameters['train_keyword'] = train_keyword
    dict_all_parameters['NB_CLASS'] = NB_CLASS
    dict_all_parameters['train_tmp_test'] = train_tmp_test
    dict_all_parameters['model_folder'] = model_folder_

    for window_length in window_length_list:
        for batch_size in batch_size_list:
            for units in units_list:

                i += 1
                overlap_window = window_length  # 窗口和滑动大小
                MAX_NB_VARIABLES = window_length * 2

                dict_all_parameters['window_length'] = window_length
                dict_all_parameters['batch_size'] = batch_size
                dict_all_parameters['units'] = units
                dict_all_parameters['overlap_window'] = overlap_window
                dict_all_parameters['MAX_NB_VARIABLES'] = MAX_NB_VARIABLES

                # Saving the objects:
                with open('config.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                    pickle.dump(dict_all_parameters, f)

                print(dict_all_parameters)
                
                from src.o8_main import read_commands

                ####### RUN
                
                start__time = time.time()

                accuracy_all_list, t1, s1, t2, accuracy_all, t3 = read_commands()
                
                end__time = time.time()
                run_time = end__time - start__time

                fid = open(train_info_file, 'a')
                import numpy as np
                
                str_ = '%s,%s,%.3f,%s,%d,%.2f,%d,%d,%d,%d,' % (i, train_folder, run_time, NB_CLASS, sample_rate, \
                                         train_data_rate, window_length, batch_size, units, MAX_NB_VARIABLES)
                
                fid.write(str_)

                # fid.write('Index,dataSet,CLASS,sample_rate,train_data_rate,window_length,batch_size,units,MAX_NB_VARIABLES')
                metrix = '%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f' % (accuracy_all_list[0], accuracy_all_list[1], t1, \
                                                            np.mean(s1), t2, \
                                                            np.mean(accuracy_all), t3)
                fid.write(metrix)


                fid.write('\n')
                fid.close()
