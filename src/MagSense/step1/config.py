# coding:utf-8
'''
@time:    Created on  2018-04-13 20:19:26
@author:  Lanqing
@Func:    realData_LSTM.config
'''
# About folder
input_folder, test_folder, predict_folder = 'data/mac/', 'data/test/', 'data/real_time_testing/'  # 机型
trainTestdifferent_category, predict_different_category = ['music', 'off', 'web', 'online', 'surfing', 'word'], ['realtime']
train_after_fft_data = train_after_pca_data = train_Model_folder = 'data/result/'  # 读取文件后的数据
test_after_fft_data = test_after_pca_data = 'data/test/'  # 读取文件后的数据
predict_fft_data = predict_pca_data = 'data/predict/'

# About sampling
window_length, overlap_window = 50, 25  # 窗口和滑动大小
train_percent2read_afterPCA_data = 0.1  # 使用处理后数据比例，用于减小训练数据使用的样本数(训练预测阶段)

# Preprocessing: PCA Gaussian FFT
saved_dimension_after_pca, sigma = 0.9999, 15
use_gauss, use_pca, use_fft = True, True, True  # True
