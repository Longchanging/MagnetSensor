# coding:utf-8

base = '../data/' 

#更换数据集需要且仅需要改变的参数
#程序调整后将根据选择的文件夹进行统一的调整
train_keyword = {
    'data_0509':['0-netmusic','1-youkuapp','2-tencentweb','4-surfingweb'],
    'windows':['0-offlinevideo_vlc','1-webvideo_tencent','2-appvideo_aiqiyi','3-netmusic']
    '20180511':['0-netmusic','1-offlinevideo','2-surfing'],
    '20180512':['1_netmusic','2_chrome_surfing','3_aqiyi','4_offline_video_potplayer'],
    '20180514':['01_aqiyi','02_offline_video','03_chrome_surfing','04_chrome_agar','05_word','06_ppt','07_wechat'],
    '20180515':['01_aqiyi','02_offline_video','03_chrome_surfing','04_word','05_ppt'],
    '20180516':['01_word','02_ppt','03_offline_video','04_aqiyi','05_chrome_surfing'],
    '20180517':['01_chrome', '02_ppt', '03_offline', '04_aqiyi'],
    '20180518':['01_offline_video', '02_chrome_surfing', '03_aqiyi', '04_game_plants'],
}

print('\n -------------------------------------------------------------------- \n')
print('Please input your command: default train and test and predict and baseline and plot,')
print('you can also type like: "train_predict_baseline" or "train_baseline"')
print('You can choose the folders like this: \n',train_keyword.keys(),'\n')
print('\n -------------------------------------------------------------------- \n')

# 提醒用户输入参数
str_input = input('Input your command of folder now: ')
train_keyword_ = train_keyword[str(str_input)]

#第一段程序用来处理用户传来的文件夹，即第二个参数
print('The folder you want to process is:\t',str_input)
print('The train_keyword are:\t',train_keyword_)

# 根据传参文件夹决定主要目录
train_folder, test_folder, predict_folder = base + '/input/' + '/' + str_input + '/'

# 根据传参进来的上述文件夹生成其他文件夹
base = base + '/' + str_input + '/'
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