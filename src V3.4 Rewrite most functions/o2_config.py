# coding:utf-8
# input and preprocess。Train 包括完整的train evaluation test, test 指的是完全相同的数据类型代入计算，predict指的是没有标签。
test_keyword = ['aha']
predict_keyword = ['testdata']
base = '../data/' 

# 采样
sample_rate = 10  # 单位是毫秒 ，>=1

# Preprocessing: PCA Gaussian FFT
window_length, overlap_window = 5, 5  # 窗口和滑动大小
train_data_rate = 0.1  # 使用处理后数据比例，用于减小训练数据使用的样本数(训练预测阶段)
saved_dimension_after_pca, sigma = 20, 500 
use_gauss, use_pca, use_fft = True, False, True  # True
whether_shuffle_train_and_test = True

# Model detail 
epochs, n_splits = 50 , 10
batch_size, train_batch_size = 2, 2  # essential，batch size 其实是LSTM的N_step

test_ratio, evaluation_ratio = 0.2, 0.1  # 划分训练、测试、验证集
TRAINABLE = True
MAX_NB_VARIABLES = saved_dimension_after_pca if use_pca else window_length * 2
units = 20  # int(MAX_NB_VARIABLES / 2)

### 此处添加文件相关信息 ###
train_keyword = ['05_work_word', '06_work_excel', '07_work_ppt', '08_social_wechat', '09_social_qq', '13_game_zuma', '14_game_candy', '15_game_minecraft', '16_picture_win3d', '17_chrome_surfing', '18_firefox_surfing', '19_chrome_gmail_work', '20_chrome_twitter', '22_chrome_amazon', '23_chrome_agar']
train_folder = '../data//input//apps/'
test_folder = '../data//input//apps/'
predict_folder = '../data//input//apps/'
train_tmp = '../data//tmp/apps//tmp/train/'
test_tmp = '../data//tmp/apps//tmp/test/'
predict_tmp = '../data//tmp/apps//tmp/predict/'
train_tmp_test = '../data//tmp/apps//tmp/train/test/'
model_folder = '../data//tmp/apps//model/'
NB_CLASS = 15
