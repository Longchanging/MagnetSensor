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
cmap = plt.cm.jet  # winter

######  文件相关设置 ,更改为自动切换数据集
base_input = '../data/input/'
base_output = 'C:/Users/jhh/Desktop/Magsense/数据探索/'

M, N = 10000, 200000  # 6300, 15000  # M定义数据从第几行开始读，N代表读多少行
ML = 10
sigma = 3
M = 4  # 绘制几张子图
linewidth = 5
n = 50  # 50ms 绘图取样一次
use_package = 'platform'
use_gauss = False

M, N = 10000, 200000  # 6300, 15000  # M定义数据从第几行开始读，N代表读多少行
ML = 10
sigma = 3
M = 4  # 绘制几张子图
linewidth = 2
NN = 100  # 50ms 绘图取样一次
fontsize = 25
minMI = 0.04  # 0.09  # 0.04  # 0.09 
maxMI = 12.46  # 13.79  # 12.46  # 13.79
fig_size = (11, 8)

# sigma = 0.1  # 500  # 高斯滤波

##################################### ##################################
#####################     接收用户参数，决定文件夹等           #########################
##################################### ##################################

######  接收用户参数，决定文件夹等
def interact_With_User():

    import os
    
    key_ = use_package
    
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
        # '20180519':['01_offline_video.txt', '02_aqiyi.txt', '03_chrome_surfing.txt', '04_ppt.txt'],
        'data_0509':['13204-testdata.txt'],
        # '20180516':['test_12345.txt'],
        'platform_plot':['hp_surf.txt', 'lenovo_surf.txt', 'mac_surf.txt', 'shenzhou_surf.txt'],
        
        'apps':['win_panhao__05_work_word.txt', 'win_panhao__06_work_excel.txt',
            'win_panhao__07_work_ppt.txt',
            'win_panhao__08_social_wechat.txt', 'win_panhao__09_social_qq.txt', 'win_panhao__13_game_zuma.txt',
            'win_panhao__14_game_candy.txt',
            'win_panhao__15_game_minecraft.txt', 'win_panhao__16_picture_win3d.txt', 'win_panhao__17_chrome_surfing.txt', \
            'win_panhao__18_firefox_surfing.txt',
            'win_panhao__19_chrome_gmail_work.txt', 'win_panhao__20_chrome_twitter.txt', 'win_panhao__22_chrome_amazon.txt',
            'win_panhao__23_chrome_agar.txt'],
        
        'user_word':['win_fangliang__05_work_word.txt', 'win_lanqing__05_work_word.txt', 'win_panhao__05_work_word.txt',
                      'win_wangzhong__05_work_word.txt',
                      'win_weilun__05_work_word.txt', 'win_yeqi__05_work_word.txt', 'win_zhoujie__05_work_word.txt'],

        'windows_chrome_0522':['01_gmail_read.txt', '02_twitter_normal.txt', '03_youtube.txt', '04_taobao.txt', '05_game_agar.txt', '06_work_github_read.txt', '07_news.txt'],
        # '20180515':['test_12345.txt'],
        # 'mac':['mac__iina_offline_video_1.txt', 'mac__iina_offline_video_2.txt',
        #       'mac__iina_offline_video_3.txt', 'mac__iqiyi_online_video_1.txt', 'mac__iqiyi_online_video_2.txt',
        #       'mac__safari_web_live_video_2.txt', 'mac__safari_web_live_video_3.txt'],
                     
        # 'mac':['mac__iina_offline_video_3.txt', 'mac__ms_word_1.txt', 'mac__netmusic_app_1.txt', 'mac__safari_surfing_3.txt', 'mac__safari_web_live_video_3.txt'],
        'mac':['mac__iina_offline_video_1.txt'],
    
        'user_word_mac':['mac__ms_word_1.txt', 'mac__ms_word_2.txt', 'mac__ms_word_3.txt',
        'shenzhou_word_1_(moveandclick).txt', 'shenzhou_word_2_(keystrokes).txt', 'shenzhou_word_3_(moveandclick).txt'],
        'yeqi_20180526':['win_yeqi__05_work_word.txt', 'win_yeqi__06_work_excel.txt', 'win_yeqi__07_work_ppt.txt', 'win_yeqi__08_social_wechat.txt', 'win_yeqi__09_social_qq.txt', 'win_yeqi__13_game_zuma.txt', 'win_yeqi__14_game_candy.txt', 'win_yeqi__16_picture_win3d.txt', 'win_yeqi__17_chrome_surfing.txt', 'win_yeqi__18_firefox_surfing.txt', 'win_yeqi__19_chrome_gmail_work.txt', 'win_yeqi__20_chrome_twitter.txt', 'win_yeqi__21_chrome_youtube.txt', 'win_yeqi__22_chrome_amazon.txt', 'win_yeqi__23_chrome_agar.txt'],
        'platform':['hp_edge_surfing_1.txt', 'hp_edge_surfing_2.txt', 'hp_edge_surfing_3.txt', 'hp_edge_tencent_web_live_video_1.txt', 'hp_edge_tencent_web_live_video_2.txt', 'hp_edge_tencent_web_live_video_3.txt', 'hp_netmusic_playingmusic_1.txt', 'hp_netmusic_playingmusic_2.txt', 'hp_netmusic_playingmusic_3.txt', 'hp_offline_winplayer_app_video_1.txt', 'hp_offline_winplayer_app_video_2.txt', 'hp_offline_winplayer_app_video_3.txt', 'hp_online_tencent_app_live_video_1.txt', 'hp_online_tencent_app_live_video_2.txt', 'hp_online_tencent_app_live_video_3.txt', 'hp_powerpoint_1.txt', 'hp_powerpoint_2.txt', 'hp_powerpoint_3.txt', 'mac__iina_offline_video_1.txt', 'mac__iina_offline_video_2.txt', 'mac__iina_offline_video_3.txt', 'mac__iqiyi_online_video_1.txt', 'mac__iqiyi_online_video_2.txt', 'mac__iqiyi_online_video_3.txt', 'mac__ms_word_1.txt', 'mac__ms_word_2.txt', 'mac__ms_word_3.txt', 'mac__netmusic_app_1.txt', 'mac__netmusic_app_2.txt', 'mac__netmusic_app_3.txt', 'mac__safari_surfing_1.txt', 'mac__safari_surfing_2.txt', 'mac__safari_surfing_3.txt', 'mac__safari_web_live_video_1.txt', 'mac__safari_web_live_video_2.txt', 'mac__safari_web_live_video_3.txt', 'shenzhou_chrome_surfing_1.txt', 'shenzhou_chrome_surfing_2.txt', 'shenzhou_chrome_surfing_3.txt', 'shenzhou_chrome_tclive_web_video_1.txt', 'shenzhou_chrome_tclive_web_video_2.txt', 'shenzhou_chrome_tclive_web_video_3.txt', 'shenzhou_game_myworld_1.txt', 'shenzhou_game_myworld_2.txt', 'shenzhou_game_myworld_3.txt', 'shenzhou_iqiyi_online_video_1.txt', 'shenzhou_iqiyi_online_video_2.txt', 'shenzhou_iqiyi_online_video_3.txt', 'shenzhou_net_music_1.txt', 'shenzhou_net_music_2.txt', 'shenzhou_net_music_3.txt', 'shenzhou_offvideo_potplayer_1.txt', 'shenzhou_offvideo_potplayer_2.txt', 'shenzhou_offvideo_potplayer_3.txt', 'shenzhou_powerpoint_1_(moveandclick).txt', 'shenzhou_powerpoint_2_(editpictures).txt', 'shenzhou_powerpoint_3_(editpages).txt', 'shenzhou_word_1_(moveandclick).txt', 'shenzhou_word_2_(keystrokes).txt', 'shenzhou_word_3_(moveandclick).txt', 'windows__0-offlinevideo_vlc.txt', 'windows__1-webvideo_tencent.txt', 'windows__2-appvideo_aiqiyi.txt', 'windows__3-netmusic.txt', 'windows__3201-testdata.txt', 'windows_0519_01_offline_video.txt', 'windows_0519_02_aqiyi.txt', 'windows_0519_03_chrome_surfing.txt', 'windows_0519_04_ppt.txt', 'windows_0519_test_1234.txt', 'windows_data_0509_0-netmusic.txt', 'windows_data_0509_1-youkuapp.txt', 'windows_data_0509_2-tencentweb.txt', 'windows_data_0509_3-iinavideo.txt', 'windows_data_0509_4-surfingweb.txt', 'windows_data_0509_13204-testdata.txt'],
        'platform':['hp_powerpoint_1.txt']  # , 'hp_powerpoint_2.txt', 'hp_powerpoint_3.txt'],
    }
    
    print('\n--------------------------------------------\n')
    print('keys to use: \n', list(train_keyword.keys()))
    print('\n--------------------------------------------\n')
    # key_ = input('choose your command: use above keys: \n')
    # print('\n--------------------------------------------\n')

    folder = base_input + '/' + str(key_) + '/'
    image_folder = base_output + '/' + str(key_) + '/'
    files_train = train_keyword[key_]
        
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    return folder, image_folder, files_train

folder, image_folder, files_train = interact_With_User()

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
            if  (',' in line) and ('2018' not in line) and (':' not in line) and (i < N) and (j > M)  :
                i += 1
                line = line.split(',')[0]  # exclude the last comma
                if  (i % n == 0):
                    clean_file.append(line)        
                                
        print('文件行数: %d 行  ( %.2f 秒)' % (j * 10, j * 10 / 10000), '\t 共读取: %d (%.2f 秒) ' % (N, N * 10 / 10000))
        
        clean_file = np.array(clean_file).astype(int) / 65536  ######预处理
        
        
        gaussian_X = filters.gaussian_filter1d(clean_file, sigma) if use_gauss  else clean_file
        ###### 高斯滤波
        # gaussian_X = clean_file
        # gaussian_X = filters.gaussian_filter1d(clean_file, sigma)

        # plt.plot(clean_file)
        # plt.show()
        
        
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

def plot(file_, values, i):
    
    indexList = np.array(list(range(int(len(values))))) / 10
    
    ###### 直接绘图
    yDown, yUp = np.min(values), np.max(values)
    
    name = file_.split('.')[0]
    if '-' in name:
        name = name.split('-')[1]

    
    fig = plt.figure(figsize=(100, 20))  # (figsize=(100, 20))    
    # plt.title('Magnetic signals of %s' % name, fontsize=30)  
    ax = fig.add_subplot(111) 
    # plt.xticks([])
    # plt.yticks([])
    
    #     frame = plt.gca()
    #     # y 轴不可见
    #     frame.axes.get_yaxis().set_visible(False)
    #     # x 轴不可见
    #     frame.axes.get_xaxis().set_visible(False)

    # plt.axis('off')

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # fig.set_size_inches(24, 14)  # 18.5, 10.5
    plt.axis([0, len(indexList) / 10, yDown, yUp])  # 0.240 0.2455 50ms #0.2375, 0.245 100ms
    plt.xlabel("Sample Time (second) ", size=24)
    plt.ylabel("Magnetic Signal (uT)", size=24)  
    ax.plot(indexList, values, 'b-', linewidth=linewidth)
    # ax.legend(loc='best')
    plt.savefig(image_folder + '%s%d.png' % (file_, i))
    # plt.show()
    print('saved to %s' % (image_folder + file_))

    return

def plot_new(file_, values, i):
    
    indexList = np.array(list(range(int(len(values))))) / 10
    
    ###### 直接绘图
    yDown, yUp = np.min(values), np.max(values)
    
    name = file_.split('.')[0]
    if '-' in name:
        name = name.split('-')[1]

    
    fig = plt.figure()  # (figsize=(100, 20))  # (figsize=(100, 20))    
    # plt.title('Magnetic signals of %s' % name, fontsize=30)  
    ax = fig.add_subplot(111) 
    # plt.xticks([])
    # plt.yticks([])
    
    #     frame = plt.gca()
    #     # y 轴不可见
    #     frame.axes.get_yaxis().set_visible(False)
    #     # x 轴不可见
    #     frame.axes.get_xaxis().set_visible(False)

    # plt.axis('off')

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    fig.set_size_inches(fig_size)  # 18.5, 10.5
    
    #### 认真标定 x 轴数据
    count_line = len(values)
    window_length = int(NN)  # # 相邻数据点的时间间隔
    total_time = int(count_line * NN)  # # 总的时间长度(ms)
    print(count_line, window_length, total_time)
    X = np.ceil(np.array(list(range(0, total_time, window_length))) / 1000)
    print(X)
    
    print(X, values)
    
    maxMI__ = int(np.max(np.max(values)))
    minMI__ = int(np.min(np.min(values)))
    
    min__ = np.min(np.min(values))
    max__ = np.max(np.max(values))
    
    tmp = [float('%.1f' % min__)]
    
    y_tick = list(range(minMI__ + 1, maxMI__, 3))
    tmp.extend(y_tick)
    tmp.append(float('%.1f' % max__))
    
    y_tick = np.array(tmp) 
    m = np.max(y_tick)
    n = np.min(y_tick)
    tt = float('%.1f' % ((m - n) / 5))
    y_tick = np.arange(n, m + 0.5 * tt, tt)        
    print(y_tick)
    
    plt.axis([0, np.ceil(total_time / 1000), yDown, yUp])  # 0.240 0.2455 50ms #0.2375, 0.245 100ms
    plt.xlabel("Sample Time (second) ", fontsize=fontsize)
    plt.ylabel("Magnetic Signal (mT)", fontsize=fontsize)  
    plt.xticks([50, 100, 150, 200])
    plt.yticks(y_tick)
            
    ax.plot(X, values, 'b-', linewidth=linewidth)
    # ax.legend(loc='best')
    plt.savefig(image_folder + '%s.png' % (file_.split('.')[0]))
    # plt.show()
    print('saved to %s' % (image_folder + file_.split('.')[0]))

    return


def plot_fft(file_, y , i):
    
    import numpy
    import pylab
    
    
    t = 1 / 800  # t[1] - t[0]
    Y = numpy.fft.fft(y)
    freq = numpy.fft.fftfreq(len(y), t)
    
    pylab.figure()
    pylab.plot(freq, numpy.abs(Y))
    pylab.figure()
    pylab.plot(freq, numpy.angle(Y))
    plt.savefig(image_folder + '%d%s.png' % (i, file_.split('.')[0]))

    # pylab.show()

    return

def plot_(file_, signal , i):
    
    from matplotlib.pyplot import specgram
    # data = np.random.rand(256)
    
    frame = plt.gca()
    frame.axes.get_yaxis().set_visible(False)
    frame.axes.get_xaxis().set_visible(False)
    plt.axis('off')
   
    specgram(signal, NFFT=512, Fs=512, cmap=cmap)

    plt.savefig(image_folder + '%d%s.png' % (i, file_.split('.')[0]))
    plt.show()
    
    return
    

def data_explore(file_all):
    
    ######  归一化
    print('\n文件最大值: \n', np.max(file_all), '\n文件最小值: \n', np.min(file_all), '\n')
    print('\n全局最大值: ', np.max(np.max(file_all)), '\n全局最小值: ', np.min(np.min(file_all)))
    
    file_all = np.max(np.max(file_all)) + np.min(np.min(file_all)) - file_all
    
    
    file_all = (file_all - np.min(np.min(file_all))) / (np.max(np.max(file_all)) - np.min(np.min(file_all)))
    print(file_all.describe())
    
    ######  探索
    print('\n数据探索滤波后结果:\n', file_all.describe())
    
    # file_all.plot()
    # plt.show()

    ######  绘图
    
    for file_ in files_train:
        
        if not use_gauss:
            values = file_all[file_]
            plot(file_, values, 1)
            
            slide = int(len(values) / M)
            print(len(values), slide)
            for i in range(M):
                print(i)
                tmp = values.iloc[i * slide : (i + 1) * slide]
                # plot('filtered', tmp, i)
                plot_new('filtered', tmp, i)

            
        else:
            
            values = file_all[file_]
            
            plot('filted_all', values, 1)
            
            slide = int(len(values) / M)
            print(len(values), slide)
            for i in range(M):
                tmp = values.iloc[i * slide : (i + 1) * slide]
                plot('filtered', tmp, i)
                # plot_(file_ + 'fft', tmp, i)
            
#         else:
#             for i in range(M):
#                 pass
        

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
