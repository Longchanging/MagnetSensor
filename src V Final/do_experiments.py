# coding:utf-8

########  DO EXPERIMENTS

import pickle
import time
from src.o1_setup import generate_configs

####### 存储所有参数并写入文件
model_folder_ = '../data/model/'
train_info_file = model_folder_ + 'train_info_all.txt'
dict_all_parameters = {}
train_info = {}

####### 定义需要变更的参数

train_folders = {     
                    'platform':['mac_', 'shenzhou_', 'hp_', 'windows_'],
                    'apps':['05_work_word', '08_social_wechat', '14_game_candy', '06_work_excel', '17_chrome_surfing', '07_work_ppt', \
                             '09_social_qq', \
                            '13_game_zuma' , '15_game_minecraft', \
                             '16_picture_win3d', '18_firefox_surfing', \
                             '19_chrome_gmail_work', '20_chrome_twitter', \
                             '22_chrome_amazon', '23_chrome_agar'],
                    'users':['lanqing', 'panhao', 'wangzhong', 'fangliang', 'zhoujie', 'weilun', 'yeqi'],
                    'all':['fangliang__05_work_word', 'fangliang__06_work_excel', 'fangliang__07_work_ppt',
                            'fangliang__08_social_wechat', 'fangliang__09_social_qq', 'fangliang__13_game_zuma',
                            'fangliang__14_game_candy', 'fangliang__15_game_minecraft', 'fangliang__16_picture_win3d',
                            'fangliang__17_chrome_surfing', 'fangliang__18_firefox_surfing', 'fangliang__19_chrome_gmail_work',
                            'fangliang__20_chrome_twitter', 'fangliang__22_chrome_amazon', 'fangliang__23_chrome_agar',
                            'lanqing__05_work_word', 'lanqing__06_work_excel', 'lanqing__07_work_ppt', 'lanqing__08_social_wechat',
                            'lanqing__09_social_qq', 'lanqing__13_game_zuma', 'lanqing__14_game_candy', 'lanqing__15_game_minecraft',
                            'lanqing__16_picture_win3d', 'lanqing__17_chrome_surfing', 'lanqing__18_firefox_surfing',
                            'lanqing__19_chrome_gmail_work', 'lanqing__20_chrome_twitter', 'lanqing__22_chrome_amazon',
                            'lanqing__23_chrome_agar', 'panhao__05_work_word', 'panhao__06_work_excel', 'panhao__07_work_ppt',
                            'panhao__08_social_wechat', 'panhao__09_social_qq', 'panhao__13_game_zuma', 'panhao__14_game_candy',
                            'panhao__15_game_minecraft', 'panhao__16_picture_win3d', 'panhao__17_chrome_surfing', 'panhao__18_firefox_surfing',
                            'panhao__19_chrome_gmail_work', 'panhao__20_chrome_twitter', 'panhao__22_chrome_amazon', 'panhao__23_chrome_agar',
                            'wangzhong__05_work_word', 'wangzhong__06_work_excel', 'wangzhong__07_work_ppt', 'wangzhong__08_social_wechat',
                            'wangzhong__09_social_qq', 'wangzhong__13_game_zuma', 'wangzhong__14_game_candy', 'wangzhong__15_game_minecraft',
                            'wangzhong__16_picture_win3d', 'wangzhong__17_chrome_surfing', 'wangzhong__18_firefox_surfing',
                            'wangzhong__19_chrome_gmail_work', \
                            'wangzhong__20_chrome_twitter', 'wangzhong__22_chrome_amazon', 'wangzhong__23_chrome_agar', 'weilun__05_work_word',
                            'weilun__06_work_excel', 'weilun__07_work_ppt', 'weilun__08_social_wechat', 'weilun__09_social_qq',
                            'weilun__13_game_zuma', \
                            'weilun__14_game_candy', 'weilun__15_game_minecraft', 'weilun__16_picture_win3d', 'weilun__17_chrome_surfing',
                            'weilun__18_firefox_surfing', 'weilun__19_chrome_gmail_work', 'weilun__20_chrome_twitter', 'weilun__22_chrome_amazon',
                            'weilun__23_chrome_agar', 'yeqi__05_work_word', 'yeqi__06_work_excel', 'yeqi__07_work_ppt', 'yeqi__08_social_wechat',
                            'yeqi__09_social_qq', 'yeqi__13_game_zuma', 'yeqi__14_game_candy', 'yeqi__15_game_minecraft', 'yeqi__16_picture_win3d',
                            'yeqi__17_chrome_surfing', 'yeqi__18_firefox_surfing', 'yeqi__19_chrome_gmail_work', 'yeqi__20_chrome_twitter',
                            'yeqi__22_chrome_amazon', 'yeqi__23_chrome_agar', 'zhoujie__05_work_word', 'zhoujie__06_work_excel',
                            'zhoujie__07_work_ppt', 'zhoujie__08_social_wechat', 'zhoujie__09_social_qq', 'zhoujie__13_game_zuma',
                            'zhoujie__14_game_candy', 'zhoujie__15_game_minecraft', 'zhoujie__16_picture_win3d', 'zhoujie__17_chrome_surfing',
                            'zhoujie__18_firefox_surfing', 'zhoujie__19_chrome_gmail_work',
                            'zhoujie__20_chrome_twitter', 'zhoujie__22_chrome_amazon', 'zhoujie__23_chrome_agar'],
                 'os':[ 'mac', 'freebsd', 'win8', 'win10', ],
                 'lathe':['a', 'b', '30', '50', '70', '80', '90', '100'],
                 '0806':['z0.csv', 'z2.csv', 'z10.csv', 'z30.csv', 'z50.csv', 'z70.csv', 'z85.csv', 'z100.csv'],
                 '08061':['z0', 'z2', 'z10', 'z30', 'z50', 'z70', 'z85', 'z100']
                 }

xiaopiliang = {   'platform':['mac_', 'shenzhou_', 'hp_', 'windows_'],
                    'apps':['05_work_word',
                            '08_social_wechat',
                            '13_game_zuma',
                            '16_picture_win3d',
                            '17_chrome_surfing'],
                    'users':['lanqing', 'panhao']}

train_folders = train_folders

# 采样
sample_rate = 1  # 单位是毫秒 ，>=1
epochs, n_splits = 2 , 10  # 10折交叉验证和epoch数量固定
train_data_rate = 0.01  # 使用处理后数据比例，用于减小训练数据使用的样本数(训练预测阶段)
train_batch_size = 100

dict_all_parameters['sample_rate'] = sample_rate
dict_all_parameters['epochs'] = epochs
dict_all_parameters['n_splits'] = n_splits
dict_all_parameters['train_data_rate'] = train_data_rate
dict_all_parameters['train_batch_size'] = train_batch_size


# 处理
saved_dimension_after_pca, sigma = 2, 500 
use_gauss, use_pca, use_fft = True, True, True  # True
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

window_length_list = [2]  # [2, 5, 10, 20, 100]  # 窗口大小
batch_size_list = [20]  # [2, 5, 10]  # 训练 batch大小
units_list = [2]  # [20, 10, 50, 200]  # int(MAX_NB_VARIABLES / 2)

i = 0

####### 存储相关信息
import os
if os.path.exists(train_info_file):
    os.remove(train_info_file)
    
fid = open(train_info_file, 'a')
fid.write('Index,dataSet,totalRunTime,CLASS,sample_rate,train_data_rate,window_length,batch_size,units,MAX_NB_VARIABLES,\
    knn_acc,rf_acc,time_tr,lstm_acc,time_lstm,fcn_acc,time_fcn')
fid.write('\n')
fid.close()
                
###### 循环遍历
for train_folder in train_folders:
    
    train_folder = '08061'
    
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
                overlap_window = int(0.5 * window_length)  # 窗口和滑动大小
                MAX_NB_VARIABLES = window_length * 2
                MAX_NB_VARIABLES = (window_length + saved_dimension_after_pca) if use_pca else window_length * 2

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
                                                            np.max(s1), t2, \
                                                            np.max(accuracy_all), t3)
                fid.write(metrix)


                fid.write('\n')
                fid.close()
