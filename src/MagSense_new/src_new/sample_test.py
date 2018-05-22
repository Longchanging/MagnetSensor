# coding:utf-8
from collections import Counter
import os

from pylab import *
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.utils import shuffle  

from config import train_batch_size, MAX_NB_VARIABLES, batch_size, NB_CLASS, \
    model_folder, train_tmp, train_tmp_test, test_tmp, predict_tmp, epochs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from train_test import generate_model, train_model, evaluate_model, predict_model


# config 
# folder = '../data/input/data_0509/'
# folder = '../data/input/windows/'
# folder = '../data/input/20180511/'
# config 
# folder = '../data/input/data_0509/'
# folder = '../data/input/windows/'
# folder = '../data/input/20180511/'
# folder = '../data/input/20180512/'
# folder = '../data/input/20180513/'
# folder = '../data/input/20180514/'
# folder = '../data/input/20180515/'
# folder = '../data/input/20180516/'
folder = '../data/input/20180517/'

# files_train = ['0-netmusic.txt', '1-youkuapp.txt', '2-tencentweb.txt', '3-iinavideo.txt', '4-surfingweb.txt']
# files_train = ['0-offlinevideo_vlc.txt', '1-webvideo_tencent.txt', '2-appvideo_aiqiyi.txt', '3-netmusic.txt']
# files_train = ['0-netmusic.txt', '1-offlinevideo.txt', '2-surfing.txt']
# files_train = ['1_netmusic.txt', '2_chrome_surfing.txt', '3_aqiyi.txt', '4_offline_video_potplayer.txt']
# files_train = ['01_music.txt', '02_aqiyi.txt', '03_chrome.txt']
# files_train = ['01_aqiyi.txt', '02_offline_video.txt', '03_chrome_surfing.txt', '04_chrome_agar.txt', '05_word.txt', '06_ppt.txt', '07_wechat.txt']
# files_train = ['01_aqiyi.txt', '02_offline_video.txt', '03_chrome_surfing.txt', '04_word.txt', '05_ppt.txt']
# files_train = ['01_word.txt', '02_ppt.txt', '03_offline_video.txt', '04_aqiyi.txt']
files_train = ['01_chrome_surfing.txt', '02_ppt.txt', '03_offline_video.txt', '04_aqiyi.txt']

# files_test = '13204-testdata.txt'
# files_test = '3201-testdata.txt'
# files_test = 'test_201.txt'
# files_test = '1234_newtest.txt'
# files_test = 'test_123.txt'
# files_test = 'test_123567.txt'
# files_test = 'new_new_test_123567.txt'
# files_test = 'test_12345.txt'
# files_test = 'test_12345.txt'
files_test = 'main_test_1234.txt'
# files_test = 'test_4_aqiyi.txt'
# files_test = 'test_2_ppt.txt'

model_folder = '../data/model/'

T = 3000000  # 剔除异常值
sigma = 500  # 高斯滤波
use_pca , saved_dimension_after_pca = False, 30
window_length, overlap_window = 250, 250  # 窗口和滑动大小 默认200
batch_size = 5
Max_, Min_ = 198, 189

N = 100000  # 每个文件读的最大行数
N_test_samples = 500000
M = 20  # 绘图时降采样步长
test_ratio, evaluation_ratio = 0.2, 0.1  # 切分数据集
whether_shuffle_train_and_test = True

def read_single_txt_file(single_file_name, train_test_flag):
    file_list, i = [], 0    
    fid = open(single_file_name, 'r')
    print('正在处理文件：', single_file_name, '读%d行' % N)
    
    for line in fid :
        
        if train_test_flag == 'train':
            if  i < N :
                line = line.strip('\n')
                if ',' in line and '2018' not in line:
                    i += 1
                    line = line.split(',')[:-1]  # exclude the last comma
                    file_list.append(line)
            else:
                break
            
        elif train_test_flag == 'test':
            line = line.strip('\n')
            if ',' in line and '2018' not in line:  # and i < N_test_samples:
                i += 1
                line = line.split(',')[:-1]  # exclude the last comma
                file_list.append(line)

    file_list = np.array(file_list).astype(int)
    file_list = file_list.reshape([1, file_list.shape[0] * file_list.shape[1]])
    mean_sample = np.mean(file_list[1:100])
    
    print('max after', np.max(file_list), 'min after:', np.min(file_list))
        
    file_list = gauss_filter(file_list, sigma)  # 高斯滤波，对象是一个List
    
    return file_list

def vstack_list(tmp):
    # 对一个List所有Array按行合并
    if len(tmp) > 1:  
        data = np.vstack((tmp[0], tmp[1]))
        for i in range(2, len(tmp)):
            data = np.vstack((data, tmp[i]))
    else:
        data = tmp[0]    
    return data

def read_data(folder_, files_, train_test_flag):
    # # 主函数一，控制操作流程
    list_fl, list_nm, sample_list = [], [], []
    
    for item in files_:  # 读文件
        
        fid = folder_ + item
        file_one_class = read_single_txt_file(fid, train_test_flag)  # 滤波
        print(np.array(file_one_class).shape)
        list_fl.append(file_one_class)

        file_one_class = file_one_class.T
        r, c = file_one_class.shape
        i = range(0, r, M) 
        file_one_class = file_one_class[i]  # 降采样
        file_one_class = file_one_class.T
        
        file_name = item.split('.')[0] 
        sample_list.append(file_one_class)
        list_nm.append(file_name) 
    sample_list = vstack_list(sample_list)
    list_fl = vstack_list(list_fl)
    return list_nm, sample_list.T, list_fl.T

def LabelEncoder(data):
    from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, LabelEncoder
    enc = LabelEncoder()
    out = enc.fit_transform(data)  
    joblib.dump(enc, model_folder + "Label_Encoder.m")
    return  out, enc

def min_max_scaler(train_data):
    # 归一化,改动，一起归一化
    r, c = train_data.shape
    train_data = train_data.reshape([r * c, 1])
    min_ = np.min(train_data) 
    max_ = np.max(train_data)  
    train_data = (train_data - min_) / (max_ - min_)
    print('min and max: \n', min_, max_)
    train_data = train_data.reshape([r, c])
    return train_data, min_, max_

def standard_data(X):
    import sklearn.preprocessing
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)
    return X, scaler

def min_max_test(c):
    min_ = (np.min(c, axis=1)).reshape([c.shape[0], 1])
    max_ = (np.max(c, axis=1)).reshape([c.shape[0], 1])
    return (c - min_) / (max_ - min_)

def gauss_filter(X, sigma):
    
    import matplotlib.pyplot as plt  # 高斯滤波
    import scipy.ndimage
    # X = (X - Min_) / (Max_ - Min_)  # 预归一化
    X = X / 200  # 预归一化
    gaussian_X = scipy.ndimage.filters.gaussian_filter1d(X, sigma)
    N = 10000
    #     print(X, '\n', gaussian_X) # 绘制归一化前后的图
    #     plt.plot(range(N), gaussian_X[0][:N])
    #     plt.show()
    return gaussian_X

def split_window(Array_):
    # 划分窗口，可以有overlap
    r = len(Array_)
    win_list = []
    i = 0
    while(i * overlap_window + window_length < r):  # attention here
        tmp_window = Array_[(i * overlap_window) : (i * overlap_window + window_length)]  # # main
        tmp_window = tmp_window.reshape([window_length , 1])  # reshape
        win_list.append(tmp_window)
        i += 1
    return np.array(win_list)

def plot_pdf(array_, save_file):
    # 绘图，给定array绘图
    import matplotlib
    matplotlib.rcParams['agg.path.chunksize'] = 10000   
    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages(save_file)
    yDown = min(array_)
    yUp = max(array_)
    fig = plt.figure()      
    ax = fig.add_subplot(111) 
    fig.set_size_inches(240, 20)  # 18.5, 10.5
    plt.axis([0, len(array_), yDown, yUp])  # 0.240 0.2455 50ms #0.2375, 0.245 100ms
    plt.title("MagnValue vs. Sampling", size=80)
    plt.xlabel("Sample Time ", size=50)
    plt.ylabel("MagnSignal(V)", size=40)   
    ax.plot(range(len(array_)), array_, 'b-')
    ax.legend() 
    plt.savefig(pp, format='pdf')
    pp.close()
    return

def main_read_data():

    # # 主函数
    list_nm, files_sample, list_fl = read_data(folder, files_train, 'train')  # 读文件夹，生成训练数据
    
    files_ = list_fl
    files_, min_, max_ = min_max_scaler(list_fl)  # 归一化
    
    df_train = pd.DataFrame(files_, columns=list_nm)
    df_train_sample = pd.DataFrame(files_sample, columns=list_nm)
    
    list_nm, files_sample, list_fl = read_data(folder, [files_test], 'test')  # 读文件夹，生成测试数据

    # 改变策略 重新归一化
    file_test = (list_fl - min_) / (max_ - min_)
    files_sample = (files_sample - min_) / (max_ - min_)
    file_test, min_, max_ = min_max_scaler(list_fl)  # 归一化
    files_sample, min_, max_ = min_max_scaler(files_sample)  # 归一化
    
    # 新归一化策略 将test 和 train合并重新归一化
    
    
    
    # file_test = XX.transform(list_fl)  # 归一化
    # files_sample = XX.transform(files_sample)  # 归一化

    df_test = pd.DataFrame(file_test, columns=list_nm)
    df_test_sample = pd.DataFrame(files_sample, columns=list_nm)
    
    data = df_test_sample.values  # 绘测试数据图
    plot_pdf(data, 'aa.pdf')
    
    return df_train, df_test, df_train_sample, df_test_sample

def rewrite_main_read_data():

    # # 主函数
    list_nm, files_sample, list_fl = read_data(folder, files_train, 'train')  # 读文件夹，生成训练数据
    df_train = pd.DataFrame(list_fl, columns=list_nm)
    df_train_sample = pd.DataFrame(files_sample, columns=list_nm)
    
    list_nm, files_sample, list_fl = read_data(folder, [files_test], 'test')  # 读文件夹，生成测试数据
    df_test = pd.DataFrame(list_fl, columns=list_nm)
    df_test_sample = pd.DataFrame(files_sample, columns=list_nm)
    
    return df_train, df_test, df_train_sample, df_test_sample

def process(dataframe):
    name_list, array_list = [], []
    # 预处理 
    for item_name in dataframe.columns:
        data = dataframe[item_name].values
        data = split_window(data)  # 分割窗口
        label = len(data) * [item_name]
        array_list.append(data)
        name_list.extend(label)
    data = vstack_list(array_list)
    data = data.reshape([data.shape[0], data.shape[1]])
    return data, name_list

# 准备数据

def frequency_field_feature(vector):
    transformed = np.fft.fft(vector)  # FFT
    # transformed = transformed.reshape([1, len(transformed)])  # reshape 
    return transformed.real

def time_field_feature(singleColumn):
    from collections import Counter
    medianL = np.median(singleColumn) 
    varL = np.var(singleColumn) 
    meanL = np.mean(singleColumn) 
    minL = np.min(singleColumn) 
    maxL = np.max(singleColumn) 
    modeL = Counter(singleColumn).most_common(1)[0][1]
    rangeL = maxL - minL
    aboveMeanL = len(np.where(singleColumn > meanL))
    static = [medianL, varL, meanL, modeL, rangeL, aboveMeanL]
    return np.array(static)

def feature_enfgineering(array_of_window):
    listL = []
    for i in range(len(array_of_window)):
        data = array_of_window[i]
        time_feature = time_field_feature(data)
        frequency_feature = frequency_field_feature(data)
        # print(frequency_feature)
        # listL.append(np.hstack((time_feature, data, frequency_feature)))
        listL.append(np.hstack((data, frequency_feature)))
        # listL.append(frequency_feature)  # 只使用频域特征
    listL = vstack_list(listL)
    return listL

def PCA(X):
    # reduce the input dimension
    from sklearn.decomposition import PCA
    pca = PCA(n_components=saved_dimension_after_pca)
    X = pca.fit_transform(X)
    return X, pca

def reshape_X(X, batch_size):
    from math import floor
    r, c = X.shape
    batch_num = floor(r / batch_size)

    if r > 1 and c > 1:
        X_reshaped = X[:(batch_size * batch_num)].reshape([batch_num, c, batch_size])
    else:
        X_reshaped = X[:batch_num]
    return X_reshaped, batch_num

def fetch_batch_data(data, n_classes, batch_size):
    r, c = data.shape
    data_list = []
    label_list = []
    for i in range(n_classes):  # excluded label
        print('Processing class %d' % i)
        loc = np.where(data[:, -1] == i)
        data_ = data[loc][:, :-1]  # extract data of class i
        data_class_i, batch_num = reshape_X(data_, batch_size)
        label_class_i = np.ones([batch_num, 1]) * int(data[loc][0, -1])
        print('label shape: \t', label_class_i.shape, '\t label shape: \t', data_class_i.shape)
        data_list.append(data_class_i) 
        label_list.append(label_class_i)
    return data_list, label_list

def main_prepare(data, label, batch_size):
    
    label = label.reshape([len(label), 1])
    n_classes = len(np.unique(label))
    data = np.hstack((data, label))
    
    data_list, label_list = fetch_batch_data(data, n_classes, batch_size)
    data = vstack_list(data_list) 
    label = vstack_list(label_list) 
    
    print('Prepared label shape: \t', label.shape, '\t data shape: \t', data.shape)
    return data, label

def train_test_evalation_split(data, label): 
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=test_ratio, random_state=0, shuffle=whether_shuffle_train_and_test)
    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=evaluation_ratio, random_state=0, shuffle=whether_shuffle_train_and_test)
    return X_train, X_test, X_validate, y_train, y_test , y_validate

def one_hot(label):
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder()
    label = enc.fit_transform(label)  
    return label, enc

def validatePR(prediction_y_list, actual_y_list):

    right_num_dict = {}
    prediction_num_dict = {}
    actual_num_dict = {}

    Precise = {}
    Recall = {}
    F1Score = {}
    
    if len(prediction_y_list) != len(actual_y_list):
        raise(ValueError)    
    
    for (p_y, a_y) in zip(prediction_y_list, actual_y_list):
        
        if p_y not in prediction_num_dict:
            prediction_num_dict[p_y] = 0
        prediction_num_dict[p_y] += 1

        if a_y not in actual_num_dict:  # here mainly for plot 
            actual_num_dict[a_y] = 0
        actual_num_dict[a_y] += 1

        if p_y == a_y:  # basis operation,to calculate P,R,F1
            if p_y not in right_num_dict:
                right_num_dict[p_y] = 0
            right_num_dict[p_y] += 1
    
    for i in  np.sort(list(actual_num_dict.keys()))  : 
                
        count_Pi = 0  # range from a to b,not 'set(list)',because we hope i is sorted 
        count_Py = 0
        count_Ri = 0
        count_Ry = 0

        for (p_y, a_y) in zip(prediction_y_list, actual_y_list):
            
            
            if p_y == i:
                count_Pi += 1
                
                if p_y == a_y:                              
                    count_Py += 1
                    
            if a_y == i :
                count_Ri += 1
                
                if a_y == p_y:
                    count_Ry += 1    
        
        Precise[i] = count_Py / count_Pi if count_Pi else 0               
        Recall[i] = count_Ry / count_Ri if count_Ri else 0
        F1Score[i] = 2 * Precise[i] * Recall[i] / (Precise[i] + Recall[i]) if Precise[i] + Recall[i] else 0
    
    Micro_average = np.mean(list(F1Score.values()))
    
    lenL = len(prediction_y_list)
    sumL = np.sum(list(right_num_dict.values()))
    accuracy_all = sumL / lenL
        
    return Precise, Recall, F1Score, Micro_average, accuracy_all

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

def knn_classifier(trainX, trainY, testX, testY):  # KNN Classifier
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()  # metric='cosine')
    model.fit(trainX, trainY)
    predict_ = model.predict(testX)
    # print(list(predict_.astype(int)), list(testY.astype(int)))
    Precise, Recall, F1Score, Micro_average, accuracy_all = validatePR(predict_, testY)
    print('KNN result:\n', Precise, Recall, '\n', F1Score, '\n', Micro_average, '\n', accuracy_all)
    return model

def LSTM_classifier(n_step, n_input, n_hidden, n_classes, X_train, X_test, X_validate, y_train, y_test, y_validate):
    #     from keras.layers import LSTM, Dense, Dropout, Flatten
    #     from keras.models import Sequential
    #     from keras.optimizers import Adam
    y_train = y_train.reshape(-1, 1)
    y_validate = y_validate.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    y_train, enc = one_hot(y_train)  # 独热编码处理
    y_validate = enc.transform(y_validate)
    y_test = enc.transform(y_test)

    model = Sequential()
    model.add(LSTM(input_shape=(n_input, n_step), units=n_hidden))
    #     model.add(Dropout(0.3))
    #     model.add(Flatten())

    model.add(Dense(n_classes, activation='softmax'))
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    # train
    model.fit(X_train, y_train, epochs=300, validation_data=(X_validate, y_validate), verbose=1)
    # test
    re = model.predict(X_test)  # # in  One hot format
    accu = accuracy(re, y_test)
    print('Accuracy of LSTM on test data:\t', accu)
    return model

df_train, df_test, df_train_sample, df_test_sample = rewrite_main_read_data()
print(df_train.describe(), '\n', df_test.describe())

# 绘图
# df = pd.concat([df_train_sample, df_test_sample])
# df.plot(subplots=True)
# plt.show()

# 测试集只取采样后的数据
# df_train = df_train_sample 
# df_test = df_test_sample 

# 初步预处理（独热编码 测试数据归一化）
train_data, train_label_list = process(df_train)  # 训练数据
train_label, enc = LabelEncoder(train_label_list)

test_data, test_label_list = process(df_test)  # 测试数据

train_data, a_, b_ = min_max_scaler(train_data)  # 归一化
print('train min and max,', a_, ' ', b_)
test_data, a_, b_ = min_max_scaler(test_data)  # 归一化
print('test min and max,', a_, ' ', b_)

# train_data, scaler_ = standard_data(train_data)  # 标准化
# test_data, scaler_ = standard_data(test_data)  # 标准化

test_label = (np.ones([test_data.shape[0], 1]) * 0).astype(int)
print(train_data.shape, test_data.shape, train_label[1:5], test_label[1:5].T)

# 特征工程
train_data = feature_enfgineering(train_data)
test_data = feature_enfgineering(test_data)

train_data, pca = PCA(train_data)  # 降维
test_data = pca.transform(test_data)  # 降维

# 切分数据集
X_train, X_test, X_validate, y_train, y_test, y_validate = train_test_evalation_split(train_data, train_label)
print(X_train.shape, X_test.shape, X_validate.shape)

# 训练和预测  传统模型
knn_model = knn_classifier(X_train, y_train, X_test, y_test)  # KNN
pre = knn_model.predict(test_data)
print('KNN 预测结果:\n', Counter(pre))
np.savetxt('predict_by_KNN.txt', pre)

# 处理成深度学习需要的格式
train_data, train_label = main_prepare(train_data, train_label, batch_size)
test_data, test_label = main_prepare(test_data, test_label, batch_size)

X_train, X_test, X_validate, y_train, y_test, y_validate = train_test_evalation_split(train_data, train_label)
# print(X_train, X_train.shape)
np.save(train_tmp + 'X_train.npy', X_train)
np.save(train_tmp + 'y_train.npy', y_train)
np.save(train_tmp + 'X_test.npy', X_validate)
np.save(train_tmp + 'y_test.npy', y_validate)
np.save(train_tmp_test + 'X_test.npy', X_test)
np.save(train_tmp_test + 'y_test.npy', y_test)
np.save(predict_tmp + 'X_test.npy', test_data)
np.save(predict_tmp + 'y_test.npy', test_label)

# FCN 学习 train-evaluate
model = generate_model() 
train_model_folder = model_folder + train_tmp.split('/')[-2] + "_weights.h5"
train_model(model, folder_path=train_tmp, epochs=epochs, batch_size=train_batch_size)  # , monitor='val_loss',optimization_mode='min')

# test
test_model_folder = model_folder + train_tmp_test.split('/')[-2] + "_weights.h5"
if os.path.exists(test_model_folder):
    os.remove(test_model_folder)
os.rename(train_model_folder, test_model_folder)
actual_y_list, prediction_y_list, accuracy, loss, re, conf_matrix = evaluate_model(model, folder_path=train_tmp_test, batch_size=train_batch_size)

# predict
predict_model_folder = model_folder + predict_tmp.split('/')[-2] + "_weights.h5"
if os.path.exists(predict_model_folder):
    os.remove(predict_model_folder) 
if os.path.exists(test_model_folder):                              
    os.rename(test_model_folder, predict_model_folder)
re = predict_model(model, folder_path=predict_tmp, batch_size=train_batch_size)
np.savetxt('Predict_by_fcn.txt', np.array(re))
