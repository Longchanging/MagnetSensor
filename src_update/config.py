# coding:utf-8
# input and preprocess。Train 包括完整的train evaluation test, test 指的是完全相同的数据类型代入计算，predict指的是没有标签。
test_keyword = ['aha']
predict_keyword = ['test']

# Preprocessing: PCA Gaussian FFT
window_length, overlap_window = 2,1   # 窗口和滑动大小
train_data_rate = 0.1  # 使用处理后数据比例，用于减小训练数据使用的样本数(训练预测阶段)
saved_dimension_after_pca, sigma = 20, 500 
use_gauss, use_pca, use_fft = True, False, True  # True
whether_shuffle_train_and_test = True

# Model detail 
epochs = 20 
batch_size, train_batch_size = 2, 2  # essential

test_ratio, evaluation_ratio = 0.2, 0.1  # 划分训练、测试、验证集
TRAINABLE , NB_CLASS = True, len(train_keyword)
MAX_NB_VARIABLES = saved_dimension_after_pca if use_pca else window_length*20
