# coding:utf-8
'''
@time:    Created on  2018-06-21 09:27:00
@author:  Lanqing
@Func:    Alternative data combine and large scale of experiments.
'''
########
######## 本程序最终实现了批量执行各数据集，生成对应baseline
########

######## 列出 实际使用的， 对实验有帮助的 数据集

# 更换数据集需要且仅需要改变的参数
# 程序调整后将根据选择的文件夹进行统一的调整
train_keywords = {
                 
    'lanqing_20180523':['win_lanqing__05_work_word', 'win_lanqing__07_work_ppt', 'win_lanqing__08_social_wechat', \
                        'win_lanqing__09_social_qq', 'win_lanqing__12_game_plants', 'win_lanqing__13_game_zuma', \
                        'win_lanqing__14_game_candy', 'win_lanqing__15_game_minecraft', 'win_lanqing__16_picture_win3d', \
                        'win_lanqing__17_chrome_surfing', 'win_lanqing__19_chrome_gmail_work', \
                        'win_lanqing__20_chrome_twitter', 'win_lanqing__22_chrome_amazon'],
      
    'panhao_20180524':['win_panhao__05_work_word', 'win_panhao__07_work_ppt', 'win_panhao__08_social_wechat', \
            'win_panhao__09_social_qq', 'win_panhao__12_game_plants', 'win_panhao__13_game_zuma', \
            'win_panhao__14_game_candy', 'win_panhao__15_game_minecraft', 'win_panhao__16_picture_win3d', \
            'win_panhao__17_chrome_surfing', 'win_panhao__18_firefox_surfing', \
            'win_panhao__19_chrome_gmail_work', 'win_panhao__20_chrome_twitter', \
            'win_panhao__21_chrome_youtube', 'win_panhao__22_chrome_amazon'],
                  
    'yuhui_20180527':['win_yuhui__05_work_word', 'win_yuhui__06_work_excel', 'win_yuhui__07_work_ppt', \
            'win_yuhui__08_social_wechat', 'win_yuhui__09_social_qq', 'win_yuhui__12_game_plants', \
            'win_yuhui__13_game_zuma', 'win_yuhui__14_game_candy', 'win_yuhui__15_game_minecraft', \
            'win_yuhui__16_picture_win3d', 'win_yuhui__17_chrome_surfing', 'win_yuhui__18_firefox_surfing', \
            'win_yuhui__19_chrome_gmail_work', 'win_yuhui__20_chrome_twitter', 'win_yuhui__21_chrome_youtube', \
            'win_yuhui__22_chrome_amazon', 'win_yuhui__23_chrome_agar'],
                  
    'wangzhong_20180528':['win_wangzhong__05_work_word', 'win_wangzhong__06_work_excel', 'win_wangzhong__07_work_ppt', \
        'win_wangzhong__13_game_zuma', 'win_wangzhong__14_game_candy', 'win_wangzhong__15_game_minecraft', \
        'win_wangzhong__17_chrome_surfing', 'win_wangzhong__18_firefox_surfing', 'win_wangzhong__20_chrome_twitter', \
        'win_wangzhong__21_chrome_youtube'],
    
    'yeqi_20180526':['win_yeqi__05_work_word', 'win_yeqi__06_work_excel', 'win_yeqi__07_work_ppt', \
        'win_yeqi__08_social_wechat', 'win_yeqi__09_social_qq', 'win_yeqi__13_game_zuma', \
        'win_yeqi__14_game_candy', 'win_yeqi__16_picture_win3d', 'win_yeqi__17_chrome_surfing', \
        'win_yeqi__18_firefox_surfing', 'win_yeqi__19_chrome_gmail_work', \
        'win_yeqi__20_chrome_twitter', 'win_yeqi__21_chrome_youtube', \
        'win_yeqi__22_chrome_amazon', 'win_yeqi__23_chrome_agar'],
    
    }

useful_dict_keys = ['shenzhou', 'win_lanqing+panhao+yuhui', 'win_lanqing+panhao', \
             'mac', 'windows_23_20180523_lanqing', \
             'windows_chrome_0522', 'windows_15_category', '20180519', 'win_lanqing+panhao+yeqi', \
            'windows_among_datasets', 'platform',
            'hp', 'windows_office_only', \
            'wangzhong_20180528']

#### 先生成配置文件
def generate_configs(train_folder_):
    
    train_folder = train_folder_

    import os
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

    # 先检查原文件是否含有冗余信息，有则删掉
    import shutil
    with open('o2_config.py', 'r', encoding='utf-8') as f:
        with open('o2_config.py.new', 'w', encoding='utf-8') as g:
            for line in f.readlines():
                if "train_keyword" not in line and "train_folder" not in line and "test_folder" not in line \
                    and "predict_folder" not in line and "train_tmp" not in line and "test_tmp" not in line \
                    and "predict_tmp" not in line and "train_tmp_test" not in line and "model_folder" not in line \
                    and "NB_CLASS" not in line:             
                    g.write(line)
    shutil.move('o2_config.py.new', 'o2_config.py')

    # 将更改写入config 文件
    fid = open('o2_config.py', 'a')
    for i in range(10):
        fid.write(dict_configs['str' + str(i + 1)])
        fid.write('\n')

    return


def single_train(train_keywords, train_folder_):
        
    ######### 下列全部内容来自 o8_main.py
    ######### 因为要改动头部的config定义方式，所以上面内容改变，其他内容不变
    from imp import reload
    from src.o2_config import model_folder, train_folder, train_keyword, train_data_rate, train_tmp, \
        test_folder, test_keyword, test_tmp, base, \
        predict_folder, predict_keyword, predict_tmp
    import src.o2_config
    reload(src.o2_config)
    
    from src.o3_read_data import read__data
    from src.o4_preprocess import preprocess
    from src.o5_prepare import train_test, predict, test
    from src.o6_train_test import train_MODEL, test_MODEL, predict_MODEL, test_test_MODEL
    from src.o7_baseline import  baseline_trainTest, baseline_predict
    # mpl.use('Agg') 
    # first generate new configs
    
    # importlib.reload()

    def check_model():
        from sklearn.externals import joblib
        label_encoder = joblib.load(model_folder + "Label_Encoder.m")
        print(label_encoder.classes_)
        return
    
    def plot():
        
        import numpy as np
        def plot_(X, str_):
            from collections import Counter
            import matplotlib.pyplot as plt    
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
        # main_train()
        # main_test()
        # main_predict()
        # check_model()
        # plot()
        
    read_commands()
    
###### 主程序
for folder_ in train_keywords.keys():
    
    print('Function started... processing folder %s' % folder_)
    
    #### 生成配置文件
    generate_configs(folder_)
    single_train(train_keywords, folder_)
