'''
@time:    Created on  2018-04-13 20:19:26
@author:  Lanqing
@Func:    realData_LSTM.config
'''
# input data
category = 'test'

# input data
input_folder = 'data/input/'
after_pca_data = input_folder
processed_folder = 'data/input/'
input_folder = processed_folder
Model_folder = 'data/model/'

# random seed
random_number = 1

# train ,test,evaluation
M = 0  # read data scale
test_ratio = 1

# train detail 
epochs = 200
batch_size = 32
#train_batch_size = 32
learning_rate = .005
DATASET_INDEX = 48

MAX_TIMESTEPS = batch_size  # MAX_TIMESTEPS_LIST[DATASET_INDEX]
MAX_NB_VARIABLES = 22  # MAX_NB_VARIABLES[DATASET_INDEX]
NB_CLASS = 1  # NB_CLASSES_LIST[DATASET_INDEX]
TRAINABLE = True
