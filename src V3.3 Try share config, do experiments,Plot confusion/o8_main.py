# coding:utf-8
# 
from matplotlib import pyplot as plt
import numpy as np
from src.o1_setup import generate_configs
generate_configs()  ##### 此处必须调整位置，在其他包引用config之前生成新config

from src.o2_config import model_folder, train_folder, train_keyword, train_data_rate, train_tmp, \
    test_folder, test_keyword, test_tmp, base, \
    predict_folder, predict_keyword, predict_tmp
from src.o3_read_data import read__data
from src.o4_preprocess import preprocess
from src.o5_prepare import train_test, predict, test
from src.o6_train_test import train_MODEL, test_MODEL, predict_MODEL, test_test_MODEL
from src.o7_baseline import  baseline_trainTest, baseline_predict

# mpl.use('Agg') 
# first generate new configs

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

def main_train():
    train_test() 
    train_MODEL()
    test_MODEL()
    return

def main_test():
    read__data(test_folder, test_keyword, 1, test_tmp)
    preprocess('test', test_keyword, test_tmp)
    test()
    test_test_MODEL()
    return

def main_predict():
    read__data(predict_folder, predict_keyword, 1, predict_tmp)
    preprocess('predict', predict_keyword, predict_tmp)
    predict()
    predict_MODEL()
    return

def baseline():
    baseline_trainTest()
    # baseline_predict()
    return

def read_commands():

    # 第二段程序 需要读用户传的命令是什么（训练、测试、预测、基线、模型）
    # 使用命令行参数驱动主程序
    
    read__data(train_folder, train_keyword, train_data_rate, train_tmp)
    preprocess('train', train_keyword, train_tmp)
    baseline()
    main_train()
    # main_test()
    # main_predict()
    # check_model()
    # plot()

read_commands()
