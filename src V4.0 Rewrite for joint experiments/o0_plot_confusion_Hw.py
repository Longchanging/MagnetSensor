# coding:utf-8
'''
@time:    Created on  2018-06-20 14:18:55
@author:  Lanqing
@Func:    Plot Results
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plot_dir = 'C:/Users/jhh/Desktop/Magsense/NN_HW_pictures/'
size = (8, 8)
cmap = plt.cm.Blues   
# cmap = plt.cm.jet

def plot_confusion(conf_arr, title_, figure_saved, alphabet, size):
        
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j) / float(a))
        norm_conf.append(tmp_arr)
    
    fig = plt.figure(figsize=size)
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap,
                    interpolation='nearest')
    
    width, height = conf_arr.shape
    
    for x in range(width):
        for y in range(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')
    
    cb = fig.colorbar(res)
    plt.title(title_)
    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])
    # fig.set_size_inches(18.5, 100.5)
    plt.savefig(plot_dir + figure_saved, format='pdf')
    plt.show()
    return

def plot_platform():
    platform = np.array([
                         [9832, 833, 0, 0], [831, 9430, 0, 0], [0, 0, 12343, 0], [0, 0, 0, 7770]
                         ])
    
    title_ = 'Classification of different Platforms'
    figure_saved = 'Classification of different Platforms.pdf'
    alphabet = ['hp', 'mac', 'shenzhou', 'windows']
    plot_confusion(platform, title_, figure_saved, alphabet, size)
    return
    
def plot_chrome_5():
    diff_bhvr = np.array([
                         [494, 1, 1, 3, 0], [1, 509, 0, 2, 0], [0, 0, 690, 1, 0], [4, 0, 61, 579, 67], [0, 0, 0, 28, 457]
                         ])
    
    title_ = 'Classification of chrome different Behaviors'
    figure_saved = 'Classification of chrome different Behaviors.pdf'
    alphabet = ['gmail', 'twitter', 'youtube', 'agar', 'news']
    plot_confusion(diff_bhvr, title_, figure_saved, alphabet, size)
    return

def plot_apps_21():
    diff_apps = np.array([
    [173, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 149, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 125, 1, 9, 10, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 2, 10, 0, 0], [0, 0, 0, 153, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 1, 151, 0, 18, 0, 0, 0, 0, 0, 1, 5, 0, 0, 0, 0, 0, 0, 0], [0, 0, 6, 0, 1, 163, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 0, 1, 0, 149, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 168, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 197, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 170, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 159, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 170, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 132, 0, 9, 9, 5, 0, 1, 1, 1], [0, 0, 1, 0, 1, 0, 6, 8, 0, 0, 0, 0, 0, 137, 0, 0, 0, 0, 5, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 184, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 1, 74, 0, 15, 15, 0, 49],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 189, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 169, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 6, 7, 4, 2, 143, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 170, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 162]                         
    ])
    title_ = 'Classification of  different Apps'
    figure_saved = 'Classification of different Apps.pdf'
    alphabet = ['potplayer', 'winplayer', 'word', 'excel', 'ppt', 'wechat', 'qq', 'baiducloud',
                'camera', 'plants', 'zuma', 'candy', 'minecraft', 'win3d', 'chrome', 'firefox', 'gmail',
                'twitter', 'youtube', 'amazon', 'agar']
    labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    plot_confusion(diff_apps, title_, figure_saved, labels, size)
    return

def plot_stetes_9():
    diff_stetes = np.array([
    [157, 0, 0, 0, 0, 0, 0, 0, 0], [0, 162, 0, 0, 0, 0, 0, 0, 0], [0, 0, 661, 0, 0, 0, 13, 0, 1],
    [0, 0, 0, 291, 0, 0, 0, 12, 0], [0, 11, 0, 0, 140, 0, 0, 0, 5], [0, 0, 0, 0, 11, 332, 0, 0, 10],
    [0, 0, 7, 0, 1, 0, 352, 0, 4], [0, 0, 0, 138, 0, 0, 0, 188, 1], [0, 0, 0, 0, 6, 32, 2, 0, 642]                    
    ])
    title_ = 'Classification of  different States'
    figure_saved = 'Classification of different States.pdf'
    alphabet = ['camera', 'dwnld', 'game', 'music', 'pic', 'social', 'surf', 'video', 'work']
    plot_confusion(diff_stetes, title_, figure_saved, alphabet, size)
    return

def plot_diff_user():
    diff_user = np.array([
        [127, 6, 8, 1, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0, 1, 0, 0, 2, 10, 1, 1, 5, 2, 0], [3, 168, 0, 1, 0, 0, 0, 7, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [14, 0, 163, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 150, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 156, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 3, 0, 0, 112, 0, 7, 2, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22], [1, 3, 0, 0, 0, 0, 0, 147, 0, 0, 0, 0, 1, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 3, 0, 178, 0, 0, 0, 3, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 189, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 166, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 156, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 8, 0, 0, 0, 90, 0, 0, 0, 0, 0, 4, 21, 4, 0, 0, 0], [0, 0, 0, 0, 9, 12, 10, 0, 0, 0, 0, 0, 0, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 170, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 158, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 174, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 154, 0, 0, 0, 0, 0, 0],
        [7, 2, 0, 0, 0, 0, 10, 0, 1, 0, 0, 0, 2, 3, 13, 6, 0, 0, 115, 0, 0, 0, 1, 5], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 124, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 172, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 174, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 153, 0], [1, 0, 0, 0, 0, 7, 25, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 1, 0, 0, 0, 0, 113]
    ])
    title_ = 'Classification of  different User'
    figure_saved = 'Classification of different User.pdf'
    labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    plot_confusion(diff_user, title_, figure_saved, labels, size)
    return

def plot_accuracy():
    dict_ = {
    'mac'   :96.00,
    'hp'    :96.55,
    'shenzhou'   :91.40,
    'windows'    :98.26,
    'platforms': 96.20,
    'apps'  : 92.46,
    'users':90.61
    }
    df = pd.Series(dict_)
    df.plot(kind='barh')
    plt.title('Classification results using FCN_LSTM')
    # fig.set_size_inches(18.5, 100.5)
    plt.savefig(plot_dir + 'Classification results using FCN_LSTM.pdf', format='pdf')
    plt.show()
    return

def plot_multi_algorithms():
    # Experiments    traditional    FCN_LSTM
    
    dict_ = {
    'Hp':    [83.32, 87.12  , 97.33],
    'Shenzhou':    [82.43, 88.5 , 91.94],
    'mac':    [85.63, 92.8 , 97.86],
    'Platforms':    [81.61, 89.5  , 92.2],
    'Apps'    :[83.13, 87.1 , 93.94],
    'Users':   [ 85.5, 88.2 , 91.3]
    }
    df = pd.DataFrame(dict_, index=['RF', 'LSTM', 'FCN_LSTM']).T
    size = (8, 6)
    df.plot(kind='barh', figsize=size)
    plt.xlim((80, 100))
    plt.xlabel('Accuracy')
    plt.ylabel('Items')
    plt.title('Classification of different algorithms')
    plt.savefig(plot_dir + 'Classification of different algorithms.pdf', format='pdf')
    plt.show()

    return

def plot_predict_labels():
    arr_pre = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0], [6, 0, 0], [7, 0, 0],
        [8, 0, 0], [9, 0, 0], [10, 0, 0], [11, 0, 0], [12, 0, 0], [13, 0, 0], [14, 0, 0], [15, 0, 0],
        [16, 0, 0], [17, 0, 0], [18, 0, 0], [19, 0, 0], [20, 0, 0], [21, 0, 0], [22, 0, 0], [23, 0, 0], [24, 0, 0], [25, 0, 0],
        [26, 0, 0], [27, 0, 0], [28, 0, 1], [29, 0, 0], [30, 0, 0], [31, 0, 0], [32, 0, 0], [33, 0, 0], [34, 0, 0], [35, 0, 0],
        [36, 0, 0], [37, 0, 0], [38, 0, 0], [39, 0, 0], [40, 0, 0], [41, 0, 1], [42, 0, 0], [43, 0, 0], [44, 0, 0], [45, 0, 0],
        [46, 0, 0], [47, 0, 0], [48, 0, 0], [49, 0, 0], [50, 0, 0], [51, 0, 0], [52, 0, 0], [53, 0, 0], [54, 0, 0], [55, 0, 0],
        [56, 0, 1], [57, 0, 0], [58, 0, 0], [59, 0, 0], [60, 0, 0], [61, 0, 0], [62, 0, 0], [63, 0, 0], [64, 0, 0], [65, 0, 0],
        [66, 0, 0], [67, 0, 0], [68, 0, 0], [69, 0, 0], [70, 0, 0], [71, 0, 0], [72, 0, 0], [73, 0, 0], [74, 0, 0], [75, 0, 0],
        [76, 0, 0], [77, 0, 0], [78, 0, 0], [79, 0, 0], [80, 0, 0], [81, 0, 0], [82, 0, 0], [83, 0, 0], [84, 0, 0], [85, 0, 0],
        [86, 0, 0], [87, 0, 0], [88, 0, 0], [89, 0, 0], [90, 0, 0], [91, 0, 0], [92, 0, 0], [93, 0, 0], [94, 0, 0], [95, 0, 0],
        [96, 0, 0], [97, 0, 0], [98, 0, 0], [99, 0, 0], [100, 0, 1], [101, 1, 1], [102, 1, 1], [103, 1, 1], [104, 1, 1],
        [105, 1, 1], [106, 1, 1], [107, 1, 1], [108, 1, 1], [109, 1, 1], [110, 1, 0], [111, 1, 1], [112, 1, 1], [113, 1, 1],
        [114, 1, 1], [115, 1, 1], [116, 1, 1], [117, 1, 1], [118, 1, 1], [119, 1, 1], [120, 1, 1], [121, 1, 1], [122, 1, 1],
        [123, 1, 1], [124, 1, 1], [125, 1, 1], [126, 1, 1], [127, 1, 1], [128, 1, 1], [129, 1, 1], [130, 1, 1], [131, 1, 1],
        [132, 1, 1], [133, 1, 1], [134, 1, 0], [135, 1, 1], [136, 1, 1], [137, 1, 1], [138, 1, 1], [139, 1, 1], [140, 1, 1],
        [141, 1, 1], [142, 1, 1], [143, 1, 1], [144, 1, 1], [145, 1, 0], [146, 1, 0], [147, 1, 1], [148, 1, 1], [149, 1, 1],
        [150, 1, 1], [151, 1, 1], [152, 1, 1], [153, 1, 1], [154, 1, 1], [155, 1, 1], [156, 1, 1], [157, 1, 1], [158, 1, 1],
        [159, 1, 1], [160, 1, 1], [161, 1, 1], [162, 1, 1], [163, 1, 1], [164, 1, 1], [165, 1, 1], [166, 1, 1], [167, 1, 1],
        [168, 1, 1], [169, 1, 1], [170, 1, 1], [171, 1, 1], [172, 1, 1], [173, 1, 1], [174, 1, 1], [175, 1, 1], [176, 1, 1],
        [177, 1, 1], [178, 1, 1], [179, 1, 2], [180, 1, 1], [181, 1, 1], [182, 1, 1], [183, 1, 1], [184, 1, 1], [185, 1, 1],
        [186, 1, 1], [187, 1, 1], [188, 1, 1], [189, 1, 1], [190, 1, 1], [191, 1, 1], [192, 1, 1], [193, 1, 1], [194, 1, 1],
        [195, 1, 1], [196, 1, 1], [197, 1, 1], [198, 1, 1], [199, 1, 1], [200, 2, 2], [201, 2, 2], [202, 2, 2], [203, 2, 2],
        [204, 2, 2], [205, 2, 2], [206, 2, 2], [207, 2, 2], [208, 2, 2], [209, 2, 2], [210, 2, 2], [211, 2, 2], [212, 2, 2],
        [213, 2, 2], [214, 2, 2], [215, 2, 2], [216, 2, 1], [217, 2, 1], [218, 2, 2], [219, 2, 2], [220, 2, 2], [221, 2, 2],
        [222, 2, 2], [223, 2, 2], [224, 2, 2], [225, 2, 2], [226, 2, 2], [227, 2, 2], [228, 2, 2], [229, 2, 2], [230, 2, 2],
        [231, 2, 2], [232, 2, 2], [233, 2, 2], [234, 2, 2], [235, 2, 2], [236, 2, 2], [237, 2, 2], [238, 2, 2], [239, 2, 2],
        [240, 2, 2], [241, 2, 2], [242, 2, 1], [243, 2, 2], [244, 2, 2], [245, 2, 2], [246, 2, 2], [247, 2, 2], [248, 2, 2],
        [249, 2, 2], [250, 2, 2], [251, 2, 2], [252, 2, 2], [253, 2, 2], [254, 2, 2], [255, 2, 2], [256, 2, 2], [257, 2, 2],
        [258, 2, 2], [259, 2, 2], [260, 2, 2], [261, 2, 2], [262, 2, 2], [263, 2, 2], [264, 2, 2], [265, 2, 2], [266, 2, 2],
        [267, 2, 2], [268, 2, 0], [269, 2, 2], [270, 2, 2], [271, 2, 2], [272, 2, 2], [273, 2, 2], [274, 2, 2], [275, 2, 2],
        [276, 2, 2], [277, 2, 2], [278, 2, 2], [279, 2, 2], [280, 2, 2], [281, 2, 2], [282, 2, 2], [283, 2, 2], [284, 2, 2],
        [285, 2, 2], [286, 2, 2], [287, 2, 2], [288, 2, 2], [289, 2, 2], [290, 2, 1], [291, 2, 2], [292, 2, 2], [293, 2, 2],
        [294, 2, 2], [295, 2, 2], [296, 2, 2], [297, 2, 2], [298, 2, 2], [299, 2, 2]])
    df = pd.DataFrame(arr_pre[:, 1:], columns=['true', 'predict'])
    size = (8, 6)
    x = range(len(arr_pre))
    # plt.scatter(x, arr_pre[:, 1:], '-.', x, arr_pre[:, 1:], ':')
    
    ######## 为解决颜色 、legend 不一致，需要将函数拆分
    
    # plt.plot(x, arr_pre[:, 1:], '-.', x, arr_pre[:, 1:], ':.')
    plt.plot(x, arr_pre[:, 1], color='#40ff00', linestyle='-.')
    plt.plot(x, arr_pre[:, 2], color='b', linestyle=':')
    plt.legend(['true_label', 'predict_label'])
    plt.ylim((0, 3))
    plt.xlabel('Time(second)')
    plt.ylabel('Labels(user id)')
    plt.title('real-time user prediction')
    plt.savefig(plot_dir + 'True_predict.pdf', format='pdf')
    plt.show()
    
    return

def plot_window_scale():
    window_scale = np.array([[10, 20, 50, 100, 200, 500, 1000, 1500, 2000],
                             [90.1, 98.1, 96.9 , 96.5, 96.3, 96.2, 92.6, 92.1, 90],
                             [81.2, 83.7, 85.3, 87.2, 89, 92.1, 92.0, 92.0, 92.0],
                             [83.7, 84.5, 87.3, 92.5, 91.2, 91.0, 90.4, 90.1, 90.0],
                             [80.2, 83.4, 85.7, 86.1, 88, 90.6, 91.2, 92.1, 92.8]])
    window_scale_df = pd.DataFrame(window_scale[1:]).T
    
    win_scale_index = window_scale[0]
    os = window_scale[1]
    state = window_scale[2]
    app = window_scale[3]
    user = window_scale[4]
    
    # window_scale_df.plot(kind='barh')
    x = win_scale_index
    plt.plot(x, os, color='#40ff00', linestyle='-.')
    plt.plot(x, state, color='#0000ff', linestyle=':')
    plt.plot(x, app, color='#ffbf00', linestyle='--')
    plt.plot(x, user, color='#ff0080', linestyle=':')

    plt.legend(['os', 'state', 'app', 'user'])
    # plt.ylim((0, 3))
    plt.xlabel('Window length(second)')
    plt.ylabel('Accuracy(%)')
    plt.title('Performance with different window length')
    plt.savefig(plot_dir + 'window_scale.pdf', format='pdf')
    
    plt.show()

    return

def plot_raw_data_():
    return
if __name__ == '__main__':
    
    #     plot_window_scale()
    #     plot_predict_labels()
    #     plot_multi_algorithms()
    #     plot_accuracy()
    #     plot_platform()
    #     plot_chrome_5()
    #     plot_apps_21()
    #     plot_stetes_9()
    #     plot_diff_user()
    pass