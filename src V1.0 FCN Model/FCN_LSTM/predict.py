# coding:utf-8
'''
@time:    Created on  2018-04-25 15:13:38
@author:  Lanqing
@Func:    testFCN.predict
'''
# coding:utf-8
'''
@time:    Created on  2018-04-19 10:52:07
@author:  Lanqing
@Func:    testFCN.mydata_model
'''

from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from keras.models import Model
from keras.models import load_model

from config import *
from config import Model_folder, processed_folder
import numpy as np
from utils.keras_utils import train_model, evaluate_model, set_trainable
from utils.layer_utils import AttentionLSTM


def generate_model():
    ip = Input(shape=(MAX_NB_VARIABLES, MAX_TIMESTEPS))

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

if __name__ == '__main__':
    

    # load model
    model = generate_model()
    model.load_weights(Model_folder + 'input_weights.h5')

    X_train = np.load(processed_folder + 'X_train.npy')
    y_train = np.load(processed_folder + 'y_train.npy')
    X_test = np.load(processed_folder + 'X_test.npy')
    y_test = np.load(processed_folder + 'y_test.npy')
    
    # predict
    print('test after load: \n', X_train)
    predict_result = model.predict(X_train)
    print(predict_result.shape)
    loc = np.argmax(predict_result, axis=1)
    print(loc,loc.shape)
    print('predict_result: \n', list(loc))
