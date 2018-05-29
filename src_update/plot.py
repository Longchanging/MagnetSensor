# coding:utf-8
# 实现将实验结果绘图功能： 读取文件，在windows下显示保存图像
import matplotlib as mpl
mpl.use('Agg') 
from collections import Counter
from matplotlib import pyplot as plt
import pandas as pd
import os, sys

base = ''
folders = [base + '180517/', base + '180518/']
traditional_predict_file = 'best_model_predict_labels.txt'
fcn_predict_file = 'Predict_final_result.txt'
label_dictional_model = "Label_Encoder.m"

def check_model(model_file):
    from sklearn.externals import joblib
    label_encoder = joblib.load(model_file)
    print('输出类别对应文件名关系：\n', label_encoder.classes_, '\n')
    return

def main_plot():
    
    # 遍历每个文件夹
    for folder_ in folders:
        
        # 取出文件
        traditional_predict = pd.read_csv(folder_ + traditional_predict_file)
        traditional_predict.columns = ['traditional_predict']
        fcn_predict = pd.read_csv(folder_ + fcn_predict_file)
        fcn_predict.columns = ['fcn_predict']
        model_file = folder_ + label_dictional_model

        # 创建图像文件夹
        image_folder = folder_+'/image/'
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        # 输出模型信息，主要为了保证对应的信息准确，0123代表什么
        check_model(model_file)
        print('\n正在处理文件夹下文件：',folder_,'\n')
        print(traditional_predict.describe(),'\n')
        print(fcn_predict.describe(),'\n')
        
        # 输出预测数据各种应用对应的样本个数
        print('传统模型统计信息：\n', traditional_predict.count(), '\n')
        print('FCN预测统计信息：\n', fcn_predict.count(), '\n')
        traditional_predict.plot(kind='hist', title='traditional predict result')
        plt.savefig(image_folder + 'hist of traditional predict result.jpg') 
        fcn_predict.plot(kind='hist', title='fcn  predict result')
        plt.savefig(image_folder + 'hist of fcn predict result.jpg') 
        traditional_predict.plot(kind='line', title='traditional predict result')
        plt.savefig(image_folder + 'line chart of traditional predict result.jpg') 
        fcn_predict.plot(kind='line',title='fcn  predict result')
        plt.savefig(image_folder + 'line chart of fcn predict result.jpg') 
        plt.show()

if __name__ == '__main__':
    main_plot()

