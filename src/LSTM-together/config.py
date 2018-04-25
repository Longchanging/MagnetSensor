# coding:utf-8
'''
@time:    Created on  2018-04-13 20:19:26
@author:  Lanqing
@Func:    realData_LSTM.config
'''
# input data
input_folder = '../../data/input/5/' 
#different_category = ['game', 'music', 'video'] 
different_category = ['chrome_surfing', 'chrome_video', 'offvideo',
						'game','music',
						'word',
						'powerpoint']

# temp data
after_fft_data = '../../data/tmp/para1/'

after_pca_data = '../../data/tmp/para1/'
Model_folder = '../../data/model/together/'
direc = '../../data/keep-point/together/'
summaries_dir = '../../data/keep-point/together/'

# About sampling
window_length = 500
overlap_window = 250  # 定义为：每次向前滑动的大小。例如：每天向前移动1，定义为1.

# train ,test,evaluation
test_ratio = 0.2
evaluation_ratio = 0.1
M = 0  # for test part data

# PCA Gaussian FFT
saved_dimension_after_pca = 0.99
sigma = 15
use_gauss = True  # True
use_pca = True
use_fft = True

# train detail 
batch_size = 500
learning_rate = .00000005
max_iterations = 100000  # 2000
hidden_size = 100  # memory
# other parameter
dropout = 0.8
num_layers = 3  # number of layers of stacked RNN's
max_grad_norm = 5  # maximum gradient norm during training
