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
saved_dimension_after_pca, sigma = 5, 500 
use_gauss, use_pca, use_fft = True, True, True  # True
whether_shuffle_train_and_test = True

# Model detail 
epochs, n_splits = 50 , 10
batch_size, train_batch_size = 2, 2  # essential，batch size 其实是LSTM的N_step

test_ratio, evaluation_ratio = 0.2, 0.1  # 划分训练、测试、验证集
TRAINABLE = True
MAX_NB_VARIABLES = saved_dimension_after_pca if use_pca else window_length
units = 20  # int(MAX_NB_VARIABLES / 2)

### 此处添加文件相关信息 ###
train_keyword = ['mac_', 'shenzhou_', 'hp_', 'windows_']
train_folder = '../data//input//platform/'
test_folder = '../data//input//platform/'
predict_folder = '../data//input//platform/'
train_tmp = '../data//tmp/platform//tmp/train/'
test_tmp = '../data//tmp/platform//tmp/test/'
predict_tmp = '../data//tmp/platform//tmp/predict/'
train_tmp_test = '../data//tmp/platform//tmp/train/test/'
model_folder = '../data//tmp/platform//model/'
NB_CLASS = 4
