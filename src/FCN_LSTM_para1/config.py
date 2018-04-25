# coding:utf-8
'''
@time:    Created on  2018-04-13 20:19:26
@author:  Lanqing
@Func:    realData_LSTM.config
'''
different_category = ['chrome_surfing', 'chrome_video', 'offvideo',
						'game', 'music',
						'word',
						'powerpoint']
different_category = ['word_1','word_2']#,'word_3']
different_category = ['powerpoint_1', 'powerpoint_2','powerpoint_3']
different_category = ['music', 'off','web','online']
different_category = ['music', 'off','web','online','surfing']
different_category = ['shenzhou_net_music','mac__netmusic_app','hp_netmusic_playingmusic']#, 'off','web','online','surfing']


# input data
input_folder = 'data/input/'
after_fft_data = '../../data/tmp/para2/'
after_pca_data = '../../data/tmp/para2/'
processed_folder = 'data/input/'
input_folder = processed_folder
Model_folder = 'data/model/'

# About sampling
window_length = 100
overlap_window = 50  # 定义为：每次向前滑动的大小。例如：每天向前移动1，定义为1.

# train ,test,evaluation
M = 0  # read data scale
#batch_size = 20
test_ratio = 0.2

# PCA Gaussian FFT
saved_dimension_after_pca = 0.9999
sigma = 15
use_gauss = True  # True
use_pca = True
use_fft = True

# train detail 
epochs = 200
batch_size = 16
train_batch_size = 16
learning_rate = .005
DATASET_INDEX = 48

MAX_TIMESTEPS = batch_size  # MAX_TIMESTEPS_LIST[DATASET_INDEX]
MAX_NB_VARIABLES = 12  # MAX_NB_VARIABLES[DATASET_INDEX]
NB_CLASS = len(different_category)  # NB_CLASSES_LIST[DATASET_INDEX]
TRAINABLE = True
