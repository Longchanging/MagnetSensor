# coding:utf-8

import pickle

##### 加载参数，全局变量
with open('config.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    dict_all_parameters = pickle.load(f)
    model_folder = dict_all_parameters['model_folder']
    train_folder = dict_all_parameters['train_folder']
    train_keyword = dict_all_parameters['train_keyword']
    train_data_rate = dict_all_parameters['train_data_rate']
    train_tmp = dict_all_parameters['train_tmp']

from matplotlib import pyplot as plt
import numpy as np
from src.o3_read_data import read__data
from src.o4_preprocess import preprocess
from src.o5_prepare import main_prepare
from src.o7_baseline_traditional import  baseline_trainTest
from src.o7_baseline_LSTM import train_lstm
from src.o7_baseline_fcn import train_fcn

def check_model():
    from sklearn.externals import joblib
    label_encoder = joblib.load(model_folder + "Label_Encoder.m")
    print(label_encoder.classes_)
    return

def plot():
    
    def plot_(X, str_):
        from collections import Counter
        # import matplotlib.pyplot as plt    
        x = range(len(X))
        plt.plot(x, X)
        plt.savefig(model_folder + str_ + '_lineChart.jpg') 

        plt.show()
        dict_ = dict(Counter(X))
        print(dict_.keys(), dict_.values())
        plt.hist(X, bins=6)
        plt.savefig(model_folder + str_ + '_histChart.jpg') 
        # plt.show()
        
    tradictional_predict_label = np.loadtxt(model_folder + 'best_model_predict_labels.txt')
    predict = np.loadtxt(model_folder + 'Predict' + '_final_result.txt')
    test_file_id = open(model_folder + 'evaluate' + '_final_result.txt', 'r')
    for line in test_file_id:
        if 'predict' in line:
            test_predict = line.split('[')[1].split(']')[0]
            test_predict = [int(s.strip()) for s in test_predict[2:-3].split(',')]
        elif 'actual' in line:
            test_true = line.split('[')[1].split(']')[0]
            test_true = [int(s.strip()) for s in test_true[2:-3].split(',')]
    # print(test_true, type(test_true), '\n', test_predict, '\n', tradictional_predict_label, '\n', predict)        
    
    
    plot_(test_true, 'test_true')
    plot_(test_predict, 'test_predict')
    plot_(predict, 'predict') 
    plot_(tradictional_predict_label, 'tradictional_predict')
    
    return

def read_commands():

    # 第二段程序 需要读用户传的命令是什么（训练、测试、预测、基线、模型）
    # 使用命令行参数驱动主程序
    
    import time
    
    read__data(train_folder, train_keyword, train_data_rate, train_tmp)  #### 读数据
    preprocess('train', train_keyword, train_tmp)  #### 预处理
    main_prepare()  #### 准备LSTM系模型输入

    time0 = time.time()

    accuracy_all_list, max_score = baseline_trainTest()  #### 训练KNN、RF等传统模型
    
    time1 = time.time()
    s1 = train_lstm()  #### 训练LSTM模型
    time2 = time.time()
    accuracy_all = train_fcn()  #### 训练 FCN模型
    time3 = time.time()
    
    return accuracy_all_list, (time1 - time0), s1, (time2 - time1), accuracy_all, (time3 - time2)
      
    # check_model()  #### 输出 dict对应的标签
    # plot()  #### 绘制图表

# read_commands()
