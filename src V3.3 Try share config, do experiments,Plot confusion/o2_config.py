# coding:utf-8
# input and preprocess。Train 包括完整的train evaluation test, test 指的是完全相同的数据类型代入计算，predict指的是没有标签。
test_keyword = ['aha']
predict_keyword = ['testdata']
base = '../data/' 

# 采样
sample_rate = 5  # 单位是毫秒 ，>=1

# Preprocessing: PCA Gaussian FFT
window_length, overlap_window = 100, 50  # 窗口和滑动大小
train_data_rate = 1  # 使用处理后数据比例，用于减小训练数据使用的样本数(训练预测阶段)
saved_dimension_after_pca, sigma = 20, 500 
use_gauss, use_pca, use_fft = True, False, True  # True
whether_shuffle_train_and_test = True

# Model detail 
epochs = 50 
batch_size, train_batch_size = 2, 2  # essential

test_ratio, evaluation_ratio = 0.2, 0.1  # 划分训练、测试、验证集
TRAINABLE = True
MAX_NB_VARIABLES = saved_dimension_after_pca if use_pca else window_length * 2

### 此处添加文件相关信息 ###
train_keyword = ['win_lanqing__05_work_word', 'win_lanqing__07_work_ppt', 'win_lanqing__08_social_wechat', 'win_lanqing__09_social_qq', 'win_lanqing__12_game_plants', 'win_lanqing__13_game_zuma', 'win_lanqing__14_game_candy', 'win_lanqing__15_game_minecraft', 'win_lanqing__16_picture_win3d', 'win_lanqing__17_chrome_surfing', 'win_lanqing__19_chrome_gmail_work', 'win_lanqing__20_chrome_twitter', 'win_lanqing__22_chrome_amazon']
train_folder = '../data//input//lanqing_20180523/'
test_folder = '../data//input//lanqing_20180523/'
predict_folder = '../data//input//lanqing_20180523/'
train_tmp = '../data//tmp/lanqing_20180523//tmp/train/'
test_tmp = '../data//tmp/lanqing_20180523//tmp/test/'
predict_tmp = '../data//tmp/lanqing_20180523//tmp/predict/'
train_tmp_test = '../data//tmp/lanqing_20180523//tmp/train/test/'
model_folder = '../data//tmp/lanqing_20180523//model/'
NB_CLASS = 13
