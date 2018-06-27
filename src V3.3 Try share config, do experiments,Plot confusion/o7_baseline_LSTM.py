# coding:utf-8
'''
@time:    Created on  2018-06-26 21:15:24
@author:  Lanqing
@Func:    
'''

import keras
from keras.layers import LSTM
from keras.layers.core import Dropout, Dense, Activation
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.preprocessing import MinMaxScaler
from algorithms import *
import numpy as np


# Hyper Paramters
TimeStep = 10
HiddenLayers = 64
EPOCHS = 1000
BatchSize = 24
validate_ratio = 0.1

# Load data from files
def npzload(data):
    l = []
    for key in data.keys():
        d = data[key]
        d = np.transpose(d, (1, 0, 2))
        d = np.reshape(d, (d.shape[0], d.shape[1] * d.shape[2]))
        l.append(d)
        # print(d.shape)
    return l


def load_all_data():
    data = {}
    data1 = np.load('./data/01.npz')
    data2 = np.load('./data/02.npz')
    data3 = np.load('./data/03.npz')
    label = np.load('./data/label.npy')
    data[1] = npzload(data1)
    data[2] = npzload(data2)
    data[3] = npzload(data3)
    data['label'] = keras.utils.to_categorical(label)
    print(data.keys())
    return data

def compress(data, label):
    x = []
    y = []
    for i in range(len(data)):
        clip = data[i]
        tmp = int(clip.shape[0] / TimeStep)
        for step in range(tmp):
            l = step * TimeStep
            r = l + TimeStep
            x.append(np.reshape(clip[l:r], (TimeStep, clip.shape[1])))
            y.append(label[i])
    return np.array(x), np.array(y)


def preprocess(data, label):
    tmpall = []
    for clip in data:
        tmpall.extend(clip)
    scaler = MinMaxScaler()
    scaler.fit(tmpall)
    re = []
    for clip in data:
        re.append(scaler.transform(clip))
    train, test = (compress(re[:9], label[:9]), compress(re[9:], label[9:]))
    
    np.savetxt('FL_train.txt', train[0].reshape([ train[0].shape[0] * train[0].shape[1], train[0].shape[2]]))
    
    # Shuffle
    index = list(range(train[0].shape[0]))
    np.random.shuffle(index)
    shuffle_train = (train[0][index], train[1][index])
    index = list(range(test[0].shape[0]))
    np.random.shuffle(index)
    shuffle_test = (test[0][index], test[1][index])
    return (shuffle_train, shuffle_test)

def get_data(index):
    data = load_all_data()
    curdata = preprocess(data[index], data['label'])
    return curdata
    return (compress(curdata[:9], data['label'][:9]), compress(curdata[9:], data['label'][9:]))

def concentrate(data):
    return

def get_model():
    model = Sequential()
    model.add(LSTM(input_shape=(TimeStep, 310), units=16))
    model.add(Dense(3, activation='softmax'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def train_all():
    
    train_datas, train_labels, test_datas, test_labels = [], [], [], []
    
    for i in range(1, 4):
        train, test = get_data(i)
        train_data, train_label, test_data, test_label = train[0], train[1], test[0], test[1]
        print(train_data.shape)
        train_datas.append(train_data) 
        train_labels.append(train_label) 
        test_datas.append(test_data) 
        test_labels.append(test_label) 

    train_data = vstack_list(train_datas)  # concentrate data
    train_label = vstack_list(train_labels)
    test_data = vstack_list(test_datas)
    test_label = vstack_list(test_labels)
    
    print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)

    model = get_model()
    
    X_train, X_validate, y_train, y_validate = train_test_evalation_split(train[0], train[1], validate_ratio=validate_ratio)
    print(train[0].shape, test[0].shape)

    model.fit(X_train, y_train, epochs=300, validation_data=(X_validate, y_validate), verbose=1)
    re = model.predict(test[0])
    
    import numpy as np
    actual_y_list, prediction_y_list = [], []
    for item in test[1]:
        actual_y_list.append(np.argmax(item))
    print(actual_y_list)
        
    for item in re:
        prediction_y_list.append(np.argmax(item))
    print(prediction_y_list)
    
    acc = get_acc(re, test[1])
    Precise, Recall, F1Score, Micro_average, accuracy_all = validatePR(prediction_y_list, actual_y_list) 
    print(acc)
    print(Precise, '\n', Recall, '\n', F1Score, '\n', Micro_average, accuracy_all)
    
    model.save(str('all') + '.h5')
    with open(str('all') + '.txt', 'w') as f:
        f.write('Accuracy:' + str(acc))

    return

def train(index):
    model = get_model()
    train, test = get_data(index)
    X_train, X_validate, y_train, y_validate = train_test_evalation_split(train[0], train[1], validate_ratio=validate_ratio)
    print(train[0].shape, test[0].shape)

    model.fit(X_train, y_train, epochs=300, validation_data=(X_validate, y_validate), verbose=1)
    re = model.predict(test[0])
    
    import numpy as np
    actual_y_list, prediction_y_list = [], []
    for item in test[1]:
        actual_y_list.append(np.argmax(item))
    print(actual_y_list)
        
    for item in re:
        prediction_y_list.append(np.argmax(item))
    print(prediction_y_list)
    
    acc = get_acc(re, test[1])
    Precise, Recall, F1Score, Micro_average, accuracy_all = validatePR(prediction_y_list, actual_y_list) 
    print(acc)
    print(Precise, '\n', Recall, '\n', F1Score, '\n', Micro_average, accuracy_all)
    
    model.save(str(index) + '.h5')
    with open(str(index) + '.txt', 'w') as f:
        f.write('Accuracy:' + str(acc))
        
    return
        
if __name__ == '__main__':
    # for i in range(1, 4):
    #    train(i)
        
    # problem2
    train_all()