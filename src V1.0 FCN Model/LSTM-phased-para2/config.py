'''
@time:    Created on  2018-04-13 20:19:26
@author:  Lanqing
@Func:    realData_LSTM.config
'''
# input data
input_folder = 'test/'
file_test = input_folder + 'realtimetesting.txt'
category = 'test'

# temp data
after_pca_data = 'test/'
after_fft_data = 'test/'
Model_folder = 'test/'
M = 0

# About sampling
window_length = 50
overlap_window = 25  
percent2read_afterPCA_data = 1

# PCA Gaussian FFT
saved_dimension_after_pca = 22
sigma = 15
use_gauss = True  # True
use_pca = True
use_fft = True
