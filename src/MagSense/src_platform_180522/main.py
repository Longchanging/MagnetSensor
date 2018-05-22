import matplotlib as mpl
mpl.use('Agg') 
from matplotlib import pyplot as plt


from baseline import  baseline_trainTest, baseline_predict
from config import model_folder, train_folder, train_keyword, train_data_rate, train_tmp, \
    test_folder, test_keyword,test_tmp,\
    predict_folder, predict_keyword, predict_tmp
import numpy as np
from prepare import train_test, predict,test
from preprocess import preprocess
from read_data import read__data
from train_test import train_MODEL, test_MODEL, predict_MODEL,test_test_MODEL

def check_model():
    from sklearn.externals import joblib
    label_encoder = joblib.load(model_folder + "Label_Encoder.m")
    print(label_encoder.classes_)
    return

def plot():
    
    def plot_(X, str_):
        from collections import Counter
        #import matplotlib.pyplot as plt    
        x = range(len(X))
        plt.plot(x, X)
        plt.savefig(model_folder + str_ + '_lineChart.jpg') 

        plt.show()
        dict_ = dict(Counter(X))
        print(dict_.keys(), dict_.values())
        plt.hist(X, bins=6)
        plt.savefig(model_folder + str_ + '_histChart.jpg') 
        #plt.show()
        
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
    #print(test_true, type(test_true), '\n', test_predict, '\n', tradictional_predict_label, '\n', predict)        
    
    
    plot_(test_true, 'test_true')
    plot_(test_predict, 'test_predict')
    plot_(predict, 'predict') 
    plot_(tradictional_predict_label, 'tradictional_predict')
    
    return

def main_train():
    read__data(train_folder, train_keyword, train_data_rate, train_tmp)
    preprocess('train', train_keyword, train_tmp)
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
    baseline_predict()
    return

def run_commands():
    import sys

    print('Please input your command: default train and test and predict and baseline and plot,')
    print('you can also type like: "train_predict_baseline" or "train_baseline"')

    command = sys.argv[1]
    
    if command:
        if 'train' in command:
            main_train()
        if 'test' in command:
            main_test()
        if 'predict' in command:
            main_predict()
        if 'base' in command:
            baseline()
        if 'plot' in command:
            plot()

    if command == '1':        
        main_train()
        main_test()
        main_predict()
        check_model()
        baseline()
        plot()
    return

run_commands()
