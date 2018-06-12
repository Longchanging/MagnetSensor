# coding:utf-8
# 实现将实验结果绘图功能： 读取文件，在windows下显示保存图像
from collections import Counter
from matplotlib import pyplot as plt
import pandas as pd
mpl.use('Agg') 

folders = [base + '/20180517/', base + '/20180518/']
traditional_predict_file = 'best_model_predict_labels.txt'
fcn_predict_file = 'Predict_final_result'
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
        fcn_predict = pd.read_csv(folder_ + fcn_predict_file)
        model_file = folder_ + label_dictional_model
        
        # 输出模型信息，主要为了保证对应的信息准确，0123代表什么
        check_model(model_file)
        # 输出预测数据各种应用对应的样本个数
        print('传统模型统计信息：\n', traditional_predict.count(), '\n')
        print('FCN预测统计信息：\n', fcn_predict.count(), '\n')
        traditional_predict.plot(kind=hist, xticks='Classes', yticks='Counter', title='traditional predict result', subplots=True)
        fcn_predict.plot(kind=hist, xticks='Classes', yticks='Counter', title='fcn  predict result', subplots=True)
        traditional_predict.plot(kind=line, xticks='Classes', yticks='Counter', title='traditional predict result', subplots=True)
        fcn_predict.plot(kind=line, xticks='Classes', yticks='Counter', title='fcn  predict result', subplots=True)

    return

if __name__ == '__main__':
    main_plot()

