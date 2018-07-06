# coding:utf-8
'''
@time:    Created on  2018-04-19 10:52:07
@author:  Lanqing
@Func:    testFCN.mydata_model
'''

import pickle

##### 加载参数，全局变量
with open('config.pkl', 'rb') as f:  # Python 3: open(..., 'rb')

    dict_all_parameters = pickle.load(f)

    train_batch_size = dict_all_parameters['train_batch_size']
    MAX_NB_VARIABLES = dict_all_parameters['MAX_NB_VARIABLES']
    batch_size = dict_all_parameters['batch_size']
    train_tmp = dict_all_parameters['train_tmp']
    train_keyword = dict_all_parameters['train_keyword'] 
    train_tmp_test = dict_all_parameters['train_tmp_test'] 
    epochs = dict_all_parameters['epochs'] 
    n_splits = dict_all_parameters['n_splits'] 
    model_folder = dict_all_parameters['model_folder'] 
    whether_shuffle_train_and_test = dict_all_parameters['whether_shuffle_train_and_test'] 
    NB_CLASS = dict_all_parameters['NB_CLASS']
    evaluation_ratio = dict_all_parameters['evaluation_ratio']


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # remove warnings

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from keras.models import Model
from keras.optimizers import Adam
from numpy.random import seed

from src.o7_baseline_traditional import validatePR
from src.o7_baseline_LSTM import get_data, get_weight, oneHot2List

import keras
import numpy as np
from src.utils.layer_utils import AttentionLSTM
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from sklearn import metrics

seed(1)    

##### FCN超参数

batch_size = batch_size
learning_rate = 1e-3
monitor = 'val_acc'  # val_loss
optimization_mode = 'max'
compile_model = True
factor = 1. / np.sqrt(2)  # not time series 1. / np.sqrt(2)

def generate_model():
    ip = Input(shape=(MAX_NB_VARIABLES, batch_size))

    x = Masking()(ip)
    x = LSTM(8)(x)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(16, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(32, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(16, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out)
    model.summary()

    # add load model code here to fine-tune

    return model

def squeeze_excite_block(input):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    filters = input._keras_shape[-1]  # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
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
    
    X_train, y_train, X_test_left, y_test_left = get_data()
    X, y = X_train, y_train
    model = generate_model()
    
    # y_oneHot = keras.utils.to_categorical(y)
    print('after one hot,y shape:', y.shape)
    
    skf_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=10)
    scores_accu, scores_f1 = [], []

    ############# 十折交叉验证
    i = 0

    ######## 细粒度定义模型
    weight_fn = "%s/%s_weights.h5" % (model_folder, train_tmp.split('/')[-2])
    print(weight_fn)

    model_checkpoint = ModelCheckpoint(weight_fn, verbose=1, mode=optimization_mode,
                                       monitor=monitor, save_best_only=True, save_weights_only=True)


    ##### 注意到十折交叉验证好像不可行，集群上已经改为训epoch
    for train_index, test_index in skf_cv.split(X, y):
        
        i += 1
        print('============    第  %d 折交叉验证            =====================' % i)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test_org = y[train_index], y[test_index]
        
        print(dict(Counter(y_train[:, 0])))
        weight_dict = get_weight(list(y_train[:, 0]))
        
        y_train = keras.utils.to_categorical(y_train)
        
        X_train, X_validate, y_train, y_validate = train_evalation_split(X_train, y_train)
        
        reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=100, mode=optimization_mode,
                                      factor=factor, cooldown=0, min_lr=1e-4, verbose=2)
        callback_list = [model_checkpoint, reduce_lr]
        optm = Adam(lr=learning_rate)
        if compile_model:
            model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])
    
        model.fit(X_train, y_train, batch_size=train_batch_size, epochs=epochs, callbacks=callback_list, class_weight=weight_dict, \
                   verbose=1, validation_data=(X_validate, y_validate))
        
        predict_y = model.predict(X_test) 
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
    predict_y_left = model.predict(X_test_left)  # now do the final test
    predict_y_left = oneHot2List(predict_y_left)
    confusion = metrics.confusion_matrix(y_test_left, predict_y_left)
    np.savetxt(model_folder + 'fcn_train_test_confusion_matrix.csv', confusion.astype(int), delimiter=',', fmt='%d')
    print ('\tfinal confusion matrix:\n', confusion)

    return accuracy_all

if __name__ == "__main__":
    train_fcn()
