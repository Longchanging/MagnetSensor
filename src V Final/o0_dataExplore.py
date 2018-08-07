# coding:utf-8
'''
@time:    Created on  2018-05-31 13:44:11
@author:  Lanqing
@Func:    src.dataExplore : To Totally Explore the data
'''

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
plt.rc('font', family='Helvetica')

##################################################################################################################
#############################         目标是高效 完成数据探索工作                              ################################################
##################################################################################################################

######  文件相关设置 ,更改为自动切换数据集
base_input = '../data/input/'
base_output = 'C:/Users/jhh/Desktop/Magsense/数据探索/'
M, N = 10000, 20000  # 6300, 15000  # M定义数据从第几行开始读，N代表读多少行
sigma = 500  # 高斯滤波

##################################### ##################################
#####################     接收用户参数，决定文件夹等           #########################
##################################### ##################################

######  接收用户参数，决定文件夹等
def interact_With_User():

    import os
    
    train_keyword = {
        'data_0509':['0-netmusic.txt', '1-youkuapp.txt', '2-tencentweb.txt', '4-surfingweb.txt'],
        'windows':['0-offlinevideo_vlc.txt', '1-webvideo_tencent.txt', '2-appvideo_aiqiyi.txt', '3-netmusic.txt'],
        '20180511':['0-netmusic.txt', '1-offlinevideo.txt', '2-surfing.txt'],
        '20180512':['1_netmusic.txt', '2_chrome_surfing.txt', '3_aqiyi.txt', '4_offline_video_potplayer.txt'],
        '20180514':['01_aqiyi.txt', '02_offline_video.txt', '03_chrome_surfing.txt', '04_chrome_agar.txt', '05_word.txt', '06_ppt.txt', '07_wechat.txt'],
        '20180515':['01_aqiyi.txt', '02_offline_video.txt', '03_chrome_surfing.txt', '04_word.txt', '05_ppt.txt'],
        '20180516':['01_word.txt', '02_ppt.txt', '03_offline_video.txt', '04_aqiyi.txt', '05_chrome_surfing.txt'],
        '20180517':['01_chrome_surfing.txt', '02_ppt.txt', '03_offline_video.txt', '04_aqiyi.txt'],
        '20180518':['01_offline_video.txt', '02_chrome_surfing.txt', '03_aqiyi.txt', '04_game_plants.txt'],
        '20180519':['01_offline_video.txt', '02_aqiyi.txt', '03_chrome_surfing.txt', '04_ppt.txt'],
    }
    
    files_test = {
        'data_0509':'13204-testdata.txt',
        'windows':'3201-testdata.txt',
        '20180511':'test_201.txt',
        '20180512':'1234_testdata.txt',
        '20180514':'test_123567.txt',
        '20180515':'test_12345.txt',
        '20180516':'test_12345.txt',
        '20180517':'main_test_1234.txt',
        '20180518':'test_1234.txt',
        '20180519':'test_1234.txt',
    }
    print('\n--------------------------------------------\n')
    print('keys to use: \n', list(train_keyword.keys()))
    print('\n--------------------------------------------\n')
    key_ = input('choose your command: use above keys: \n')
    print('\n--------------------------------------------\n')

    folder = base_input + '/' + str(key_) + '/'
    image_folder = base_output + '/' + str(key_) + '/'
    files_train = train_keyword[key_]
    files_test = files_test[key_]
        
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    return folder, image_folder, files_test, files_train

folder, image_folder, files_test, files_train = interact_With_User()

#######################################################################
######################         读取数据                    ############################
#######################################################################

def read_txt():
        
    from scipy.ndimage import filters
    file_all = []
    
    ######  读单个文件
    def read_(fid):
        i, j, clean_file = 0, 0, []
        for line in fid:
            # print(len(line))
            j += 1
            if  (',' in line) and ('2018' not in line) and (':' not in line) and (i < N) and (j > M):
                i += 1
                line = line.split(',')[:-1]  # exclude the last comma
                clean_file.extend(line)                        
        print('文件行数: %d 行  ( %.2f 秒)' % (j * 10, j * 10 / 10000), '\t 共读取: %d (%.2f 秒) ' % (N, N * 10 / 10000))
        
        clean_file = np.array(clean_file).astype(int) / 65536  ######预处理
        
        ###### 高斯滤波
        gaussian_X = filters.gaussian_filter1d(clean_file, sigma)

        return  gaussian_X

    ######  借用程序，因为不信任reshape的过程，只好list合并
    def vstack_list(tmp):
        if len(tmp) > 1:
            data = np.vstack((tmp[0], tmp[1]))
            for i in range(2, len(tmp)):
                data = np.vstack((data, tmp[i]))
        else:
            data = tmp[0]    
        return data
    
    ######  读所有文件
    files_train.append(files_test)
    for file_ in files_train:
        print('\t正在处理文件：\t %s' % file_)
        fid = open(folder + file_, 'r')
        clean_file = read_(fid)
        file_all.append(clean_file)
        
    ###### 处理成pandas格式
    # file_all = np.array(file_all).reshape([len(file_all[0]), len(file_all)])
    file_all = vstack_list(file_all).T
    file_all = pd.DataFrame(file_all)
    file_all.columns = files_train
    
    return file_all

##################################### ##################################
###########################  进行数据探索   ##################################
##################################### ##################################

def data_explore(file_all):
    
    ######  归一化
    print('\n文件最大值: \n', np.max(file_all), '\n文件最小值: \n', np.min(file_all), '\n')
    print('\n全局最大值: ', np.max(np.max(file_all)), '\n全局最小值: ', np.min(np.min(file_all)))
    file_all = (file_all - np.min(np.min(file_all))) / (np.max(np.max(file_all)) - np.min(np.min(file_all)))
    print(file_all.describe())
    
    ######  探索
    print('\n数据探索滤波后结果:\n', file_all.describe())
    
    file_all.plot()
    plt.show()

    ######  绘图
    
    for file_ in files_train:
        
        values = file_all[file_]
        indexList = np.array(list(range(int(len(values))))) / 10
        
        ###### 直接绘图
        yDown, yUp = np.min(values), np.max(values)
        
        name_ = file_.split('.')[0]
        name = name_.split('-')[1]
        
        fig = plt.figure()    
        plt.title('Magnet signals of %s' % name, fontsize=30)  
        ax = fig.add_subplot(111) 
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        fig.set_size_inches(10, 7)  # 18.5, 10.5
        plt.axis([0, len(indexList) / 10, yDown, yUp])  # 0.240 0.2455 50ms #0.2375, 0.245 100ms
        plt.xlabel("Sample Time (ms) ", size=24)
        plt.ylabel("Magnet Signal (uT)", size=24)  
        ax.plot(indexList, values, 'b-', linewidth=3)
        ax.legend(loc='best')
        # plt.show()
        plt.savefig(image_folder + '%s处理后图像.pdf' % file_)

    return

##################################### ##################################
#######################     预处理和准备数据             #############################
##################################### ##################################


def preprocess(file_all):
    
    return

if __name__ == '__main__':
    
    file_all = read_txt()
    data_explore(file_all)
    preprocess(file_all)
