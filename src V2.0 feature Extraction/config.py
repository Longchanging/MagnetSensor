# coding:utf-8
# input and preprocess。Train 包括完整的train evaluation test, test 指的是完全相同的数据类型代入计算，predict指的是没有标签。
base = '../data/' 

# config 
# folder = '../data/input/data_0509/'
# folder = '../data/input/windows/'
# folder = '../data/input/20180511/'
# config 
# folder = '../data/input/data_0509/'
# folder = '../data/input/windows/'
# folder = '../data/input/20180511/'
# folder = '../data/input/20180512/'
# folder = '../data/input/20180513/'
# folder = '../data/input/20180514/'
folder = '../data/input/20180515/'
# folder = '../data/input/20180516/'


# files_train = ['0-netmusic.txt', '1-youkuapp.txt', '2-tencentweb.txt', '3-iinavideo.txt', '4-surfingweb.txt']
# files_train = ['0-offlinevideo_vlc.txt', '1-webvideo_tencent.txt', '2-appvideo_aiqiyi.txt', '3-netmusic.txt']
# files_train = ['0-netmusic.txt', '1-offlinevideo.txt', '2-surfing.txt']
# files_train = ['1_netmusic.txt', '2_chrome_surfing.txt', '3_aqiyi.txt', '4_offline_video_potplayer.txt']
# files_train = ['01_music.txt', '02_aqiyi.txt', '03_chrome.txt']
# files_train = ['01_aqiyi.txt', '02_offline_video.txt', '03_chrome_surfing.txt', '04_chrome_agar.txt', '05_word.txt', '06_ppt.txt', '07_wechat.txt']
files_train = ['01_aqiyi.txt', '02_offline_video.txt', '03_chrome_surfing.txt', '04_word.txt', '05_ppt.txt']
# files_train = ['01_word.txt', '02_ppt.txt', '03_offline_video.txt', '04_aqiyi.txt']

# files_test = '13204-testdata.txt'
# files_test = '3201-testdata.txt'
# files_test = 'test_201.txt'
# files_test = '1234_newtest.txt'
# files_test = 'test_123.txt'
# files_test = 'test_123567.txt'
# files_test = 'new_new_test_123567.txt'
files_test = 'test_12345.txt'
# files_test = 'test_12345.txt'


train_tmp, test_tmp, predict_tmp = base + '/tmp/train/', base + '/tmp/test/', base + '/tmp/predict/'  # 读取文件后的数据
train_tmp_test = base + '/tmp/train/test/'
model_folder = base + '/model/'

# Model detail 
epochs, MAX_NB_VARIABLES = 50, 40  # essential
batch_size, train_batch_size = 5, 1  # essential
TRAINABLE , NB_CLASS = True, len(files_train)
