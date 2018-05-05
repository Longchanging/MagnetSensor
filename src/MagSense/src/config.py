# coding:utf-8
# input and preprocess。Train 包括完整的train evaluation test, test 指的是完全相同的数据类型代入计算，predict指的是没有标签。
base = '../data/' 
train_folder, test_folder, predict_folder = base + '/input/mac/', base + '/input/test/', base + '/input/real_time_testing/'  # 机型
train_keyword, predict_keyword = ['music', 'off', 'web', 'online', 'surfing', 'word'], ['realtime']
train_tmp, test_tmp, predict_tmp = base + '/tmp/train/', base + '/tmp/test/', base + '/tmp/predict/'  # 读取文件后的数据
model_folder = base + '/model/'

# Preprocessing: PCA Gaussian FFT
window_length, overlap_window = 50, 25  # 窗口和滑动大小
train_data_rate = 0.1  # 使用处理后数据比例，用于减小训练数据使用的样本数(训练预测阶段)
saved_dimension_after_pca, sigma = 30, 15
use_gauss, use_pca, use_fft = True, True, True  # True

# Model detail 
epochs, MAX_NB_VARIABLES = 10, 30  # essential
batch_size, train_batch_size = 32, 32  # essential
test_ratio, evaluation_ratio = 0.2, 0.1  # 划分训练、测试、验证集
TRAINABLE , NB_CLASS = True, 6

# Baseline 