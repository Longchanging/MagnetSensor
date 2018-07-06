# coding:utf-8
########  DO EXPERIMENTS

import os

#######   确定固定部分参数，并写入新配置文件
def generate_configs(train_keywords, train_folder_):
    
    train_folder = train_folder_

    base = '../data/' 

    # 提醒用户输入参数
    train_keyword_ = train_keywords[train_folder]

    # 第一段程序用来处理用户传来的文件夹，即第二个参数
    print('The folder you want to process is:\t', train_folder)
    print('The train_keyword are:\t', train_keyword_)

    # 根据传参文件夹决定主要目录
    train_folder = test_folder = predict_folder = base + '/input/' + '/' + train_folder + '/'

    # 根据传参进来的上述文件夹生成其他文件夹
    base = base + '/tmp/' + train_folder + '/'
    train_tmp, test_tmp, predict_tmp = base + '/tmp/train/', base + '/tmp/test/', base + '/tmp/predict/'  # 读取文件后的数据
    train_tmp_test = base + '/tmp/train/test/'
    model_folder = base + '/model/'
    if not os.path.exists(base):
        os.makedirs(base)
    if not os.path.exists(train_tmp):
        os.makedirs(train_tmp)
    if not os.path.exists(test_tmp):
        os.makedirs(test_tmp)
    if not os.path.exists(predict_tmp):
        os.makedirs(predict_tmp)
    if not os.path.exists(train_tmp_test):
        os.makedirs(train_tmp_test)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    dict_configs = {
        'str1' :"train_keyword = %s" % str(train_keyword_),
        'str2' : "train_folder = '%s'" % str(train_folder),
        'str3' : "test_folder = '%s'" % str(test_folder),
        'str4' : "predict_folder = '%s'" % str(predict_folder),
        'str5' : "train_tmp = '%s'" % str(train_tmp),
        'str6' : "test_tmp = '%s'" % str(test_tmp),
        'str7' : "predict_tmp = '%s'" % str(predict_tmp),
        'str8' : "train_tmp_test = '%s'" % str(train_tmp_test),
        'str9' : "model_folder = '%s'" % str(model_folder),
        'str10': "NB_CLASS = %d" % len(train_keyword_)
    } 
    return train_keyword_,train_folder,test_folder,predict_folder,\
        train_tmp,test_tmp,predict_tmp,train_tmp_test,model_folder,len(train_keyword_)    