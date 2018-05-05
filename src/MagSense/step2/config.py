# coding:utf-8
'''
@time:  Lanqing Created on  2018-04-13 20:19:26
'''
# input data
train_test_data_folder, predict_data_folder = 'data/result/', 'data/predict/'  # 预处理后数据
processed_train_folder, processed_test_folder, processed_predict_folder, Model_folder = \
                'step2/data/train/', 'step2/data/test/', 'step2/data/predict/', 'step2/data/model/'

# train detail 
epochs, MAX_NB_VARIABLES = 200, 19  # essential
batch_size, train_batch_size = 32, 32  # essential
test_ratio, evaluation_ratio = 0.2, 0.1  # 划分训练、测试、验证集
TRAINABLE , NB_CLASS = True, 6
