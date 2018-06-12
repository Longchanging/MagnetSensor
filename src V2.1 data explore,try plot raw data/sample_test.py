# coding:utf-8
from collections import Counter
from pylab import *
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# config 
folder = '../data/input/data_0509/'
folder = '../data/input/windows/'
folder = '../data/input/20180511/'

files_train = ['0-netmusic.txt', '1-youkuapp.txt', '2-tencentweb.txt', '3-iinavideo.txt', '4-surfingweb.txt']
files_train = ['0-offlinevideo_vlc.txt', '1-webvideo_tencent.txt', '2-appvideo_aiqiyi.txt', '3-netmusic.txt']
files_train = ['0-netmusic.txt', '1-offlinevideo.txt', '2-surfing.txt']

files_test = '13204-testdata.txt'
files_test = '3201-testdata.txt'
files_test = 'test_201.txt'

model_folder = '../data/model/'

sigma = 100  # 高斯滤波
window_length, overlap_window = 50, 50  # 窗口和滑动大小

N = 50000  # 每个文件读的最大行数
M = 100  # 绘图时降采样步长
test_ratio, evaluation_ratio = 0.2, 0.1  # 切分数据集
whether_shuffle_train_and_test = True

def read_single_txt_file(single_file_name):
    file_list, i = [], 0    
    fid = open(single_file_name, 'r')
    print('正在处理文件：', single_file_name, '读%d行' % N)
    for line in fid :
        if  i < N:
            line = line.strip('\n')
            if ',' in line:
                i += 1
                line = line.split(',')[:-1]  # exclude the last comma
                file_list.append(line)
        else:
            break
    file_list = np.array(file_list).astype(int)
    file_list = file_list.reshape([1, file_list.shape[0] * file_list.shape[1]])
    
    loc = np.where(file_list < 100)  # 过滤异常值
    file_list = file_list[loc]
    file_list = file_list.reshape([1, len(file_list)])    
    
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

def read_data(folder_, files_):
    # # 主函数一，控制操作流程
    list_fl, list_nm, sample_list = [], [], []
    
    for item in files_:  # 读文件
        
        fid = folder_ + item
        file_one_class = read_single_txt_file(fid)  # 滤波
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
    from sklearn import preprocessing
    r, c = train_data.shape
    train_data = train_data.reshape([r * c, 1])
    XX = preprocessing.MinMaxScaler().fit(train_data)  
    train_data = XX.transform(train_data) 
    train_data = train_data.reshape([r, c])
    return train_data, XX

def gauss_filter(X, sigma):
    
    import matplotlib.pyplot as plt  # 高斯滤波
    import scipy.ndimage
    X = X / 65536  # 预归一化
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
    list_nm, files_sample, list_fl = read_data(folder, files_train)  # 读文件夹，生成训练数据
    files_, XX = min_max_scaler(list_fl)  # 归一化
    df_train = pd.DataFrame(files_, columns=list_nm)
    df_train_sample = pd.DataFrame(files_sample, columns=list_nm)
    
    list_nm, files_sample, list_fl = read_data(folder, [files_test])  # 读文件夹，生成测试数据
    file_test = XX.transform(list_fl)  # 归一化
    df_test = pd.DataFrame(file_test, columns=list_nm)
    df_test_sample = pd.DataFrame(files_sample, columns=list_nm)
    
    data = df_test_sample.values  # 绘测试数据图
    plot_pdf(data, 'aa.pdf')
    
    return df_train, df_test, df_train_sample, df_test_sample

def process(dataframe):
    
    rows = len(df_train)
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
        listL.append(np.hstack((time_feature, frequency_feature)))
    listL = vstack_list(listL)
    return listL

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
    model = KNeighborsClassifier()
    model.fit(trainX, trainY)
    predict_ = model.predict(testX)
    # print(list(predict_.astype(int)), list(testY.astype(int)))
    Precise, Recall, F1Score, Micro_average, accuracy_all = validatePR(predict_, testY)
    print('KNN result:\n', Precise, Recall, '\n', F1Score, '\n', Micro_average, '\n', accuracy_all)
    return model

def LSTM_classifier(n_step, n_input, n_hidden, n_classes, X_train, X_test, X_validate, y_train, y_test, y_validate):
    
    from keras.layers import LSTM, Dense, Dropout, Flatten
    from keras.models import Sequential
    from keras.optimizers import Adam
    
    y_train = y_train.reshape(-1, 1)
    y_validate = y_validate.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    y_train, enc = one_hot(y_train)  # 独热编码处理
    y_validate = enc.transform(y_validate)
    y_test = enc.transform(y_test)

    model = Sequential()
    model.add(LSTM(input_shape=(n_step, n_input), units=n_hidden))
    model.add(Dropout(0.3))
    model.add(Flatten())

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

df_train, df_test, df_train_sample, df_test_sample = main_read_data()
print(df_train.describe(), '\n', df_test.describe())

# 绘图
df = pd.concat([df_train_sample, df_test_sample])
df.plot(subplots=True)
plt.show()

# 初步预处理（归一化、独热编码）
train_data, train_label_list = process(df_train)  # 训练数据
train_label, enc = LabelEncoder(train_label_list)

test_data, test_label_list = process(df_test)  # 测试数据
test_label = (np.ones([test_data.shape[0], 1]) * 10).astype(int)
print(train_data.shape, test_data.shape, train_label[1:5], test_label[1:5].T)

# 特征工程
train_data = feature_enfgineering(train_data)
test_data = feature_enfgineering(test_data)

# 切分数据集
X_train, X_test, X_validate, y_train, y_test, y_validate = train_test_evalation_split(train_data, train_label)
print(X_train.shape, X_test.shape, X_validate.shape)

# 训练和预测
knn_model = knn_classifier(X_train, y_train, X_test, y_test)  # KNN
pre = knn_model.predict(test_data)
print('KNN 预测结果:\n', Counter(pre))

# print(X_train.shape)
# n_input = X_train.shape[1]  # LSTM
# lstm_model = LSTM_classifier(n_step=32, n_input=n_input, n_hidden=300, n_classes=len(files_train), X_train=X_train,
#                 X_test=X_test, X_validate=X_validate, y_train=y_train, y_test=y_test, y_validate=y_validate)
# pre_ = np.argmax(lstm_model.predict(test_data), 1)
# print('LSTM 预测结果:\n', Counter(pre))
