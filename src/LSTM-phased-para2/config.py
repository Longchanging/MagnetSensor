'''
@time:    Created on  2018-04-13 20:19:26
@author:  Lanqing
@Func:    realData_LSTM.config
'''
# input data
#input_folder = '../../data/input/multi_pc_data/' 
input_folder = '../../data/input/hp/'
#input_folder = '../../data/input/shenzhou/'
input_folder = '../../data/input/mac/'
#input_folder = '../../data/input/mac_shenzhou/'
#input_folder = '../../data/input/hp_shenzhou/'
#input_folder = '../../data/input/hp_mac/'
#input_folder = '../../data/input/hp_shenzhou_mac/'
#input_folder = '../../data/input/mac/'
input_folder = '../../data/input/real_time_testing/realtimetesting.txt'


#different_category = ['game', 'music', 'video'] 
#different_category = ['chrome_surfing', 'chrome_video', 'offvideo',
#						'game','music',
#						'word',
#						'powerpoint']
different_category = ['powerpoint_1', 'powerpoint_2','powerpoint_3']
different_category = ['music', 'off','web','online','surfing','powerpoint']
different_category = ['music', 'off','web','online','surfing','word']
different_category = ['realtimetesting']
#different_category = ['music']#, 'off','web','online','surfing']
#different_category = ['shenzhou_net_music','mac__netmusic_app','hp_netmusic_playingmusic']#, 'off','web','online','surfing']

# temp data
after_fft_data = '../../data/tmp/para2/'
after_pca_data = '../../data/tmp/para2/'

Model_folder = '../../data/model/para2/'
direc = '../../data/keep-point/para2/'
summaries_dir = '../../data/keep-point/para2/'

# About sampling
window_length = 50
overlap_window = 25  # 定义为：每次向前滑动的大小。例如：每天向前移动1，定义为1.

percent2read_afterPCA_data = 1

# train ,test,evaluation
test_ratio = 0.2
evaluation_ratio = 0.1
M = 0  # for test part data

# PCA Gaussian FFT
saved_dimension_after_pca = 0.9999
sigma = 15
use_gauss = True  # True
use_pca = True
use_fft = True

# train detail 
batch_size = 32
learning_rate = .005
max_iterations = 100000  # 2000
hidden_size = 100  # memory
# other parameter
dropout = 0.8
num_layers = 3  # number of layers of stacked RNN's
max_grad_norm = 5  # maximum gradient norm during training
