# coding:utf-8
'''
@time:    Created on  2018-04-19 10:52:07
@author:  Lanqing
@Func:    testFCN.mydata_model
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # remove warnings
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from keras.models import Model
from numpy.random import seed

import numpy as np
from step2.config import *
from step2.utils.keras_utils import train_model, evaluate_model, predict_model, set_trainable
from step2.utils.layer_utils import AttentionLSTM

seed(1)    

def generate_model():
    ip = Input(shape=(MAX_NB_VARIABLES, batch_size))

    x = Masking()(ip)
    x = LSTM(8)(x)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
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
    import json
    result_saver_dict = {}
    result_saver_dict['actual_y_list'] = actual_y_list
    result_saver_dict['prediction_y_list'] = prediction_y_list
    result_saver_dict['accuracy'] = accuracy
    result_saver_dict['loss'] = loss
    result_saver_dict['conf_matrix'] = conf_matrix
    print('conf_matrix：\n', conf_matrix)
    f = open(Model_folder + name + '_final_result.txt', "w")
    for (key, value) in result_saver_dict.items():
        f.write(str(key) + '\t' + str(value) + '\n')
    f.close()
    return result_saver_dict

def train_():  
    model = generate_model() 
    train_model_folder = Model_folder + processed_train_folder.split('/')[-2] + "_weights.h5"
    train_model(model, folder_path=processed_train_folder, epochs=epochs, batch_size=train_batch_size)  # , monitor='val_loss',optimization_mode='min')
    return

def test_():
    model = generate_model()
    train_model_folder = Model_folder + processed_train_folder.split('/')[-2] + "_weights.h5"
    test_model_folder = Model_folder + processed_test_folder.split('/')[-2] + "_weights.h5"
    if os.path.exists(test_model_folder):
        os.remove(test_model_folder)
    os.rename(train_model_folder, test_model_folder)
    actual_y_list, prediction_y_list, accuracy, loss, re, conf_matrix = evaluate_model(model, folder_path=processed_test_folder, batch_size=train_batch_size)
    result_saver_dict = result_saver('evaluate', actual_y_list, prediction_y_list, accuracy, loss, re, conf_matrix)
    return

def predict_():
    model = generate_model()
    train_model_folder = Model_folder + processed_train_folder.split('/')[-2] + "_weights.h5"
    test_model_folder = Model_folder + processed_test_folder.split('/')[-2] + "_weights.h5"
    predict_model_folder = Model_folder + processed_predict_folder.split('/')[-2] + "_weights.h5"
    if os.path.exists(predict_model_folder):
        os.remove(predict_model_folder)                               
    os.rename(test_model_folder, predict_model_folder)
    re = predict_model(model, folder_path=processed_predict_folder, batch_size=train_batch_size)
    np.savetxt(Model_folder + 'Predict' + '_final_result.txt', np.array(re))
    return

if __name__ == "__main__":
    train_()
    test_()
    predict_()
