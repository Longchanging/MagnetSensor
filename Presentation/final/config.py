# coding:utf-8
'''  
    input and preprocess。
    Train 包括完整的train evaluation test, 
    test 指的是完全相同的数据类型代入计算，
    predict指的是没有标签。 
'''

dict_all_parameters = {}
train_info = {}
train_folders = {
            'apps':['offline_video', 'iqiyi', 'word', 'netmusic', 'surfing', 'live'],
            'devices':['hp', 'mac', 'shenzhou', 'windows'],
            'users':['word_1', 'word_2', 'word_3'],
            'History_data':['safari_surfing', 'word_edit', 'safari_youku_video', 'word_scan'],
            '0912':['safari_surfing', 'safari_youku', 'word', 'zuma', 'netmusic'],
            '0913':['netmusic', 'NoLoad', 'safari_surfing', 'tencent_video', 'word_edit', 'Zuma'],
            '0915':['NoLoad', 'safari_surfing', 'tencent_video', 'zuma_game'],
            '0916':['lanqing', 'panhao'],
            }

# 采样
sample_rate = 1  # 单位是毫秒 ，>=1
epochs, n_splits = 2 , 10  # 10折交叉验证和epoch数量固定
train_batch_size = 100
# 处理
saved_dimension_after_pca, sigma = 1000 , 5 
use_gauss, use_pca, use_fft = True, True, True  # True
whether_shuffle_train_and_test = True
# 训练
test_ratio, evaluation_ratio = 0.2, 0.1  # 划分训练、测试、验证集
batch_size = 20  # [2, 5, 10]  # 训练 batch大小
units = 1  # [20, 10, 50, 200]  # int(MAX_NB_VARIABLES / 2)
# 循环遍历
window_length = 50  # [2, 5, 10, 20, 100]  # 窗口大小
train_data_rate = 0.1  # 使用处理后数据比例，用于减小训练数据使用的样本数(训练预测阶段)
i = 0
# 参数
overlap_window = int(0.25 * window_length)  # 窗口和滑动大小
MAX_NB_VARIABLES = window_length * 2
MAX_NB_VARIABLES = (window_length + saved_dimension_after_pca) if use_pca else window_length * 2

### 此处添加文件相关信息 ###
train_info_file = 'train_info_all.txt'
train_keyword = ['NoLoad', 'safari_surfing', 'tencent_video', 'zuma_game']
train_folder = '../data//input//0915/'
test_folder = '../data//input//0915/'
predict_folder = '../data//input//0915/'
train_tmp = '../data//tmp/0915//tmp/train/'
test_tmp = '../data//tmp/0915//tmp/test/'
predict_tmp = '../data//tmp/0915//tmp/predict/'
train_tmp_test = '../data//tmp/0915//tmp/train/test/'
model_folder = '../data//tmp/0915//model/'
NB_CLASS = 4
