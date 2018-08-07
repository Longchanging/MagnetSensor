# coding:utf-8
# input and preprocess。Train 包括完整的train evaluation test, test 指的是完全相同的数据类型代入计算，predict指的是没有标签。
test_keyword = ['aha']
predict_keyword = ['testdata']
base = '../data/' 

# 采样
sample_rate = 25  # 单位是毫秒 ，>=1

# Preprocessing: PCA Gaussian FFT
window_length, overlap_window = 20, 20  # 窗口和滑动大小
train_data_rate = 0.01  # 使用处理后数据比例，用于减小训练数据使用的样本数(训练预测阶段)
saved_dimension_after_pca, sigma = 5, 500 
use_gauss, use_pca, use_fft = True, True, True  # True
whether_shuffle_train_and_test = True

# Model detail 
epochs, n_splits = 50 , 10
batch_size, train_batch_size = 2, 2000  # essential，batch size 其实是LSTM的N_step

test_ratio, evaluation_ratio = 0.2, 0.1  # 划分训练、测试、验证集
TRAINABLE = True
MAX_NB_VARIABLES = (window_length + saved_dimension_after_pca) if use_pca else window_length * 2
units = 50  # int(MAX_NB_VARIABLES / 2)

### 此处添加文件相关信息 ###
train_keyword = ['win_yuhui__05_work_word', 'win_yuhui__06_work_excel', 'win_yuhui__07_work_ppt', 'win_yuhui__08_social_wechat', 'win_yuhui__09_social_qq', 'win_yuhui__12_game_plants', 'win_yuhui__13_game_zuma', 'win_yuhui__14_game_candy', 'win_yuhui__15_game_minecraft', 'win_yuhui__16_picture_win3d', 'win_yuhui__17_chrome_surfing', 'win_yuhui__18_firefox_surfing', 'win_yuhui__19_chrome_gmail_work', 'win_yuhui__20_chrome_twitter', 'win_yuhui__21_chrome_youtube', 'win_yuhui__22_chrome_amazon', 'win_yuhui__23_chrome_agar']
train_folder = '../data//input//yuhui_20180527/'
test_folder = '../data//input//yuhui_20180527/'
predict_folder = '../data//input//yuhui_20180527/'
train_tmp = '../data//tmp/../data//input//yuhui_20180527///tmp/train/'
test_tmp = '../data//tmp/../data//input//yuhui_20180527///tmp/test/'
predict_tmp = '../data//tmp/../data//input//yuhui_20180527///tmp/predict/'
train_tmp_test = '../data//tmp/../data//input//yuhui_20180527///tmp/train/test/'
model_folder = '../data//tmp/../data//input//yuhui_20180527///model/'
NB_CLASS = 17
