# coding:utf-8
'''
@time:    Created on  2018-06-30 14:35:08
@author:  Lanqing
@Func:    src.o7_hierarchical_fcn
'''

##### 加载参数，全局变量
import pickle
with open('config.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    
    dict_all_parameters = pickle.load(f)

    train_batch_size = dict_all_parameters['train_batch_size']
    MAX_NB_VARIABLES = dict_all_parameters['MAX_NB_VARIABLES']
    batch_size = dict_all_parameters['batch_size']
    train_tmp = dict_all_parameters['train_tmp']
    train_keyword = dict_all_parameters['train_keyword'] 
    train_tmp_test = dict_all_parameters['train_tmp_test'] 
    epochs = dict_all_parameters['epochs'] 
    window_length = dict_all_parameters['window_length']
    saved_dimension_after_pca = dict_all_parameters['saved_dimension_after_pca']

    n_splits = dict_all_parameters['n_splits'] 
    model_folder = dict_all_parameters['model_folder'] 
    whether_shuffle_train_and_test = dict_all_parameters['whether_shuffle_train_and_test'] 
    evaluation_ratio = dict_all_parameters['evaluation_ratio'] 
    NB_CLASS = dict_all_parameters['NB_CLASS'] 

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # remove warnings

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from keras.models import Model
from keras.optimizers import Adam
from numpy.random import seed

from src.o7_baseline_traditional import validatePR
from src.o7_baseline_LSTM import get_full_dataset, get_weight, oneHot2List

import keras
import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from sklearn import metrics

seed(1)    

##### FCN超参数

epochs = 50
batch_size = batch_size
train_batch_size = 800
learning_rate = 1e-2
monitor = 'val_acc'
optimization_mode = 'max'
compile_model = True
factor = 1. / np.sqrt(2)  # not time series 1. / np.sqrt(2)

def generate_model():
    
    ip1 = Input(shape=(window_length, batch_size))
    ip2 = Input(shape=(saved_dimension_after_pca, batch_size))

    x1 = Masking()(ip1)
    x1 = LSTM(8)(x1)
    x1 = Dropout(0.8)(x1)

    y1 = Permute((2, 1))(ip1)
    y1 = Conv1D(16, 8, padding='same', kernel_initializer='he_uniform')(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = squeeze_excite_block(y1)

    y1 = Conv1D(32, 5, padding='same', kernel_initializer='he_uniform')(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = squeeze_excite_block(y1)

    y1 = Conv1D(16, 3, padding='same', kernel_initializer='he_uniform')(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)

    y1 = GlobalAveragePooling1D()(y1)

    x1 = concatenate([x1, y1])
    
    #### 第二部分
    
    x2 = Masking()(ip2)
    x2 = LSTM(8)(x2)
    x2 = Dropout(0.8)(x2)

    y2 = Permute((2, 1))(ip2)
    y2 = Conv1D(16, 8, padding='same', kernel_initializer='he_uniform')(y2)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)
    y2 = squeeze_excite_block(y2)

    y2 = Conv1D(32, 5, padding='same', kernel_initializer='he_uniform')(y2)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)
    y2 = squeeze_excite_block(y2)

    y2 = Conv1D(16, 3, padding='same', kernel_initializer='he_uniform')(y2)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)
    
    y2 = GlobalAveragePooling1D()(y2)
    x2 = concatenate([x2, y2])
    
    x = concatenate([x1, x2])

    out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(inputs=[ip1, ip2], outputs=out)
    model.summary()

    # add load model code here to fine-tune

    return model

def squeeze_excite_block(inputs):
    ''' Create a squeeze-excite block
    Args:
        inputs: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    filters = inputs._keras_shape[-1]  # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(inputs)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([inputs, se])
    return se

def result_saver(name, actual_y_list, prediction_y_list, accuracy, loss, re, conf_matrix):
    result_saver_dict = {}
    result_saver_dict['actual_y_list'] = actual_y_list
    result_saver_dict['prediction_y_list'] = prediction_y_list
    result_saver_dict['accuracy'] = accuracy
    result_saver_dict['loss'] = loss
    result_saver_dict['conf_matrix'] = conf_matrix
    print('conf_matrix：\n', conf_matrix)
    f = open(model_folder + name + '_final_result.txt', "w")
    for (key, value) in result_saver_dict.items():
        f.write(str(key) + '\t' + str(value) + '\n')
    f.close()
    return result_saver_dict

def train_evalation_split(data, label): 
    '''
    split train and test
    :param data: train data
    :param label: train label
    '''
    from sklearn.model_selection import train_test_split
    X_train, X_validate, y_train, y_validate = train_test_split(data, label, \
        test_size=evaluation_ratio, random_state=0, shuffle=whether_shuffle_train_and_test)
    return X_train, X_validate, y_train, y_validate

def train_fcn():
    
    X_train, y_train, X_test_left, y_test_left = get_full_dataset()
    X, y = X_train, y_train
    model = generate_model()
    
    # y_oneHot = keras.utils.to_categorical(y)
    print('after one hot,y shape:', y.shape)
    
    n_splits = 2
    epochs = 1000
    skf_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=10)
    scores_accu, scores_f1 = [], []

    ############# 十折交叉验证
    i = 0
    for train_index, test_index in skf_cv.split(X, y):
        
        i += 1
        print('============    第  %d 折交叉验证            =====================' % i)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test_org = y[train_index], y[test_index]
        
        print(dict(Counter(y_train[:, 0])))
        weight_dict = get_weight(list(y_train[:, 0]))
        weight_fn = "%s/%s_weights.h5" % (model_folder, train_tmp.split('/')[-2])
        print(weight_fn)
        
        y_train = keras.utils.to_categorical(y_train)
        
        X_train, X_validate, y_train, y_validate = train_evalation_split(X_train, y_train)
        print(X_train.shape)
        
        ######## 细粒度定义模型
        model_checkpoint = ModelCheckpoint(weight_fn, verbose=1, mode=optimization_mode,
                                           monitor=monitor, save_best_only=True, save_weights_only=True)
        reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=100, mode=optimization_mode,
                                      factor=factor, cooldown=0, min_lr=1e-4, verbose=2)
        callback_list = [model_checkpoint, reduce_lr]
        optm = Adam(lr=learning_rate)
        if compile_model:
            model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])
    
        #### 到这一步的时候，终于可以兄弟分家了
        model.fit(x=[X_train[:, :window_length], X_train[:, window_length:]], y=y_train, batch_size=train_batch_size, epochs=epochs, callbacks=callback_list,
                  class_weight=weight_dict, verbose=1, validation_data=([X_validate[:, :window_length], X_validate[:, window_length:]], y_validate))
        
        predict_y = model.predict([X_test[:, :window_length], X_test[:, window_length:]]) 
        predict_y = oneHot2List(predict_y)
         
        # precise = metrics.average_precision_score(y_test_org, predict_y)
        # report = metrics.classification_report()
        _, _, F1Score, _, accuracy_all = validatePR(predict_y, y_test_org[:, 0])
        print (' \n accuracy_all: \n', accuracy_all, '\F1Score:  \n', F1Score)  # judge model,get score
        # print('report:', report)
        
        scores_accu.append(accuracy_all)
        scores_f1.append(F1Score)

    print (' \n accuracy_all: \n', scores_accu, '\nMicro_average:  \n', scores_f1)  # judge model,get score
    
    # ## 使用全部数据，使用保存的，模型进行实验
    predict_y_left = model.predict([X_test_left[:, :window_length], X_test_left[:, window_length:]])  # now do the final test
    predict_y_left = oneHot2List(predict_y_left)
    confusion = metrics.confusion_matrix(y_test_left, predict_y_left)
    np.savetxt(model_folder + 'fcn_train_test_confusion_matrix.csv', confusion.astype(int), delimiter=',', fmt='%d')
    print ('\tfinal confusion matrix:\n', confusion)

    return

if __name__ == "__main__":
    train_fcn()
