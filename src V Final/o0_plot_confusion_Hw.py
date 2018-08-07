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
from matplotlib.colors import ListedColormap
from matplotlib.pylab import *

normalize = True
cmap = plt.cm.Greys
# cmap = plt.cm.winter  # winter
font_size = 22  # 15
algo_fontsize = 25  # 20
un_fig_size = (20, 8)
un_app_fig_size = (12, 7)
fig_size = (12, 8)
confusion_fig_size = (15, 10)  # (15, 12)
window_fig_size = (15, 8)
precise_rate = "{:0.2f}"

def plot_confusion(conf_arr, title_, figure_saved, alphabet, size, precise_rate):

    cm = conf_arr
    target_names = alphabet
    accuracy = np.trace(cm) / float(np.sum(cm))
    
    fig = plt.figure(figsize=confusion_fig_size)
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=0, fontsize=font_size)
        plt.yticks(tick_marks, target_names, fontsize=font_size)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    res = ax.imshow(np.array(cm), cmap,
                    interpolation='nearest')
    
    # plt.title(title_, fontsize=15)

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    #     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #         if normalize:
    #             plt.text(j, i, precise_rate.format(cm[i, j]),
    #                      horizontalalignment="center",
    #                      color="white" if cm[i, j] > thresh else "black", fontsize=10)
    #         else:
    #             plt.text(j, i, "{:,}".format(cm[i, j]),
    #                      horizontalalignment="center",
    #                      color="white" if cm[i, j] > thresh else "black", fontsize=10)

    #     from mpl_toolkits.axes_grid1 import make_axes_locatable
    #     divider = make_axes_locatable(ax)
    #     cax = divider.append_axes("right", size="5%", pad=0.05)
    #     plt.colorbar(res, cax=cax)
    
    #     pcolor(arange(20).reshape(4, 5))
    #     cb = colorbar(label='a label')
    #     ax = cb.ax
    
    cb = plt.colorbar(res, pad=0.01)
    # font_size = 15  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)

    #     text = ax.yaxis.label
    #     font = matplotlib.font_manager.FontProperties(family='times new roman', style='italic', size=20)
    #     text.set_font_properties(font)
    
    plt.savefig(plot_dir + figure_saved, format='pdf')
    plt.tight_layout()
    # plt.ylabel('True label', fontsize=font_size)
    # plt.xlabel('Predicted label,accuracy={:0.4f}'.format(accuracy), fontsize=font_size)
    
    plt.tight_layout()
    
    show()


    # plt.show()
    return


def plot_platform():
    
    #     platform = np.array([
    #                          [9832, 833, 0, 0],
    #                          [831, 9430, 0, 0],
    #                          [0, 0, 12343, 0],
    #                          [0, 0, 0, 7770],
    #                          [9832, 833, 0, 0],
    #                          [831, 9430, 0, 0],
    #                          [0, 0, 12343, 0],
    #                          [0, 0, 0, 7770],
    #                          [9832, 833, 0, 0],
    #                          [831, 9430, 0, 0],
    #                          [0, 0, 12343, 0],
    #                          [0, 0, 0, 7770]
    #                          ])
    
    platform = np.array([
                         [9832, 833, 0, 0, 0, 1, 2, 3, 8, 9],
                         [831, 9430, 0, 0, 32, 34, 32, 32, 12, 10],
                         [0, 0, 12343, 0, 12, 13, 14, 15, 34, 23],
                         [0, 0, 0, 7770, 12, 13, 14, 15, 34, 23],
                         [833, 0, 0 , 12, 9832, 14, 15, 34, 23, 10],
                         [12, 13, 14, 15, 34, 9430, 23, 831, 0, 0],
                         [12, 13, 14, 15, 34, 23, 12343, 0, 0, 0],
                         [13, 14, 15, 34, 23, 0, 0, 7770, 0, 0],
                         [12, 13, 14, 15, 34, 456, 12, 3534, 9832, 10],
                         [831, 0, 0, 0, 0, 1, 9, 0, 0, 9430],
                       ])
    
    title_ = 'Classification of different Platforms'
    figure_saved = 'Platforms.pdf'
    # alphabet = ['hp', 'mac', 'shenzhou', 'windows']
    alphabet = 'ABCDEFGHIJ'
    plot_confusion(platform, title_, figure_saved, alphabet, size, precise_rate)
    return
    
def plot_chrome_5():
    diff_bhvr = np.array([
                         [494, 1, 1, 3, 0],
                         [1, 509, 0, 2, 0],
                         [0, 0, 690, 1, 0],
                         [4, 0, 61, 579, 67],
                         [0, 0, 0, 28, 457]
                         ])
    
    title_ = 'Classification of chrome different Behaviors'
    figure_saved = 'chrome_Behaviors.pdf'
    # alphabet = ['gmail', 'twitter', 'youtube', 'agar', 'news']
    alphabet = 'ABCDE'
    plot_confusion(diff_bhvr, title_, figure_saved, alphabet, size, "{:0.2f}")
    return

def plot_apps_21():
    #     diff_apps = np.array([
    #     [173, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 149, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 125, 1, 9, 10, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 2, 10, 0, 0],
    #     [0, 0, 0, 153, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 2, 1, 151, 0, 18, 0, 0, 0, 0, 0, 1, 5, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 6, 0, 1, 163, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 2, 0, 1, 0, 149, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 168, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 197, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 170, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 159, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 170, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 132, 0, 9, 9, 5, 0, 1, 1, 1],
    #     [0, 0, 1, 0, 1, 0, 6, 8, 0, 0, 0, 0, 0, 137, 0, 0, 0, 0, 5, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 184, 0, 0, 0, 1, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 1, 74, 0, 15, 15, 0, 49],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 189, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 169, 2, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 6, 7, 4, 2, 143, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 170, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 162]                         
    #     ])
    
    diff_apps = np.array([
    [173, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 149, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 125, 1, 9, 10, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 2, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 153, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 1, 151, 0, 18, 0, 0, 0, 0, 0, 1, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 6, 0, 1, 163, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 0, 1, 0, 149, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 168, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 197, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 170, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 159, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 170, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 132, 0, 9, 9, 5, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 6, 8, 0, 0, 0, 0, 0, 137, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 184, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 1, 74, 0, 15, 15, 0, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 189, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 169, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 6, 7, 4, 2, 143, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 170, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 162, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 132, 0, 9, 9, 5, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 6, 8, 0, 0, 0, 0, 0, 137, 0, 0, 0, 0, 5, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 184, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 1, 74, 0, 15, 15, 0, 49],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 189, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 169, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 6, 7, 4, 2, 143, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 170, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 162],
                       
    ])
    title_ = 'Classification of  different Apps'
    figure_saved = 'Apps_21.pdf'
    alphabet = ['potplayer', 'winplayer', 'word', 'excel', 'ppt', 'wechat', 'qq', 'baiducloud',
                'camera', 'plants', 'zuma', 'candy', 'minecraft', 'win3d', 'chrome', 'firefox', 'gmail',
                'twitter', 'youtube', 'amazon', 'agar']
    labels = list('abcdefghijklmnopqrstuvwxyz')
    labels.extend(['aa', 'ab', 'ac'])
    plot_confusion(diff_apps, title_, figure_saved, labels, size, precise_rate)
    return

def plot_stetes_9():
    diff_stetes = np.array([
    [157, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 162, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 661, 0, 0, 0, 13, 0, 1],
    [0, 0, 0, 291, 0, 0, 0, 12, 0],
    [0, 11, 0, 0, 140, 0, 0, 0, 5],
    [0, 0, 0, 0, 11, 332, 0, 0, 10],
    [0, 0, 7, 0, 1, 0, 352, 0, 4],
    [0, 0, 0, 138, 0, 0, 0, 188, 1],
    [0, 0, 0, 0, 6, 32, 2, 0, 642]                    
    ])
    
    diff_stetes = np.array([
    [157, 0, 0, 0, 0, 0, 0],
    [0, 162, 0, 0, 0, 0, 0],
    [0, 0, 661, 60, 0, 0, 13],
    [0, 0, 50, 291, 0, 0, 0],
    [0, 11, 0, 0, 140, 0, 0],
    [0, 0, 0, 0, 11, 332, 0],
    [0, 0, 7, 0, 1, 0, 352],
    ])

    title_ = 'Classification of  different States'
    figure_saved = 'States.pdf'
    # alphabet = ['camera', 'dwnld', 'game', 'music', 'pic', 'social', 'surf', 'video', 'work']
    alphabet = 'ABCDEFG'
    plot_confusion(diff_stetes, title_, figure_saved, alphabet, size, precise_rate)
    return

def plot_diff_user():
    #     diff_user = np.array([
    #         [127, 6, 8, 1, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0, 1, 0, 0, 2, 10, 1, 1, 5, 2, 0], 
    #         [3, 168, 0, 1, 0, 0, 0, 7, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #         [14, 0, 163, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    #         [0, 0, 0, 157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 150, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    #         [0, 0, 0, 0, 0, 156, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 3, 0, 0, 112, 0, 7, 2, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22], 
    #         [1, 3, 0, 0, 0, 0, 0, 147, 0, 0, 0, 0, 1, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 3, 0, 178, 0, 0, 0, 3, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 189, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 166, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 156, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 1, 0, 0, 1, 0, 8, 0, 0, 0, 90, 0, 0, 0, 0, 0, 4, 21, 4, 0, 0, 0], [0, 0, 0, 0, 9, 12, 10, 0, 0, 0, 0, 0, 0, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 170, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 158, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 174, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 154, 0, 0, 0, 0, 0, 0],
    #         [7, 2, 0, 0, 0, 0, 10, 0, 1, 0, 0, 0, 2, 3, 13, 6, 0, 0, 115, 0, 0, 0, 1, 5], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 124, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 172, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 174, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 153, 0], [1, 0, 0, 0, 0, 7, 25, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 1, 0, 0, 0, 0, 113]
    #     ])
    
    diff_user = np.array([ 
                          [3651, 35, 1, 3, 0, 31, 29, 2, 21, 1],
                          [30, 3770, 77, 7, 25, 63, 2, 2, 21, 1],
                          [0, 87, 3443, 0, 116, 14, 27, 2, 21, 1],
                          [40, 10, 4, 3780, 121, 52, 24, 2, 21, 1],
                          [0, 77, 95, 31, 3659, 50, 0, 2, 21, 1],
                          [8, 135, 69, 22, 63, 3288, 57, 2, 21, 1],
                          [4, 0, 77, 1, 0, 20, 3641, 200, 21, 1],
                          [0, 77, 95, 31, 50, 0, 21, 3659, 100, 1],
                          [8, 135, 69, 22, 63, 22, 63, 123, 3659, 500],
                          [4, 0, 77, 1, 0, 20, 22, 63, 500, 3641]
        ])
    
    title_ = 'Classification of  different User'
    figure_saved = 'User.pdf'
    labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    labels = labels[:len(diff_user)]
    plot_confusion(diff_user, title_, figure_saved, labels, size, precise_rate)
    return

def plot_compute_time():
    
    # KNN    RF    LSTM    FCN-LSTM    TF-FCN-LSTM        seconds    preprocssing
    # OS    31    126    302    347    376            109
    # App    52    210    670    830    910            177
    # User    43    162    411    486    523            206
    times = np.array([
                      [109, 31, 126, 302, 347, 376],
                      [177, 52, 210, 670, 830, 910],
                      [206, 43, 162, 411, 486, 523]
                    ])
    dict_ = {
             'Device' : [109, 31, 126, 302, 347, 376],
             'App' :   [177, 52, 210, 670, 830, 910],
             'User' :  [206, 43, 162, 411, 486, 523]
             }
    
    df = pd.DataFrame(dict_, index=['Preprocessing', 'KNN', 'RF', 'LSTM', 'FCN-LSTM', 'HF-FCN-LSTM']).T

    print(df)
    
    size = (8, 6)  # color=['limegreen', 'deeppink', 'tomato', 'dodgerblue', 'navy', 'maroon'],
    df.plot(kind='bar', figsize=size, colormap=cmap, rot=True, grid=True,
            color=[ '#BEBEBE', '#708090', '#1E90FF', '#43CD80' , '#6495ED', '#00008B'])  # , color=['#1E90FF', '#00688B', '#191970', '#00CED1', '#228B22', '#7D26CD', '#551A8B', '#20B2AA', '#00FF00'])  # 
    # plt.xlim((0.8, 1))
    # plt.ylim((0.8, 1))
    plt.legend(['Preprocessing', 'KNN', 'RF', 'LSTM', 'FCN-LSTM', 'HF-FCN-LSTM'], fontsize=15)
    plt.xlabel('Items', fontsize=15)
    plt.ylabel('Running time(seconds)', fontsize=15)
    # plt.legend()
    plt.title('Computation time of different algorithms', fontsize=15)
    plt.savefig(plot_dir + 'time.pdf', format='pdf')
    plt.show()


    return

def plot_multi_bar():
    
    width = 3
    
    # dict_ = {'App' :   [0.841 , 0.853 , 0.869 , 0.892 , 0.916 ],
    #         'OS' : [0.922, 0.941 , 0.962 , 0.975 , 0.994 ],
    #         'User' : [ 0.842  , 0.865 , 0.916 , 0.933 , 0.957 ]}
    #     df = pd.DataFrame(dict_, index=['KNN', 'RF', 'LSTM', 'FCN-LSTM', 'HF-FCN-LSTM'])
    #     X = df.values
    #     print(X)
    
    arr = np.array([ 
              [0.922, 0.941 , 0.962 , 0.975 , 0.994 ],
              [0.841 , 0.853 , 0.869 , 0.892 , 0.916 ],
              [ 0.842  , 0.865 , 0.916 , 0.933 , 0.957 ]
              ])
    
    print(arr[0, :])
    plt.figure(figsize=(9, 6))
    x = np.array(list(range(len(arr[0]))))
    print(x)
    
    labels = ['KNN', 'RF', 'LSTM', 'FCN-LSTM', 'HF-FCN-LSTM']
    plt.bar(x, arr[0, :], alpha=0.9, width=width, facecolor='green', edgecolor='white', label='Device', lw=1, tick_label=labels)
    plt.bar(x + 20, arr[1, :], alpha=0.9, width=width, facecolor='red', edgecolor='yellow', label='App', lw=1, tick_label=labels)
    plt.bar(x + 40, arr[2, :], alpha=0.9, width=width, facecolor='yellow', edgecolor='green', label='User', lw=1, tick_label=labels)

    plt.legend(loc="upper left")  # label的位置在左上，没有这句会找不到label去哪了
    plt.savefig(plot_dir + 'Algorithms.pdf', format='pdf')
    plt.show()
    
    return

def plot_multi_algorithms():
    # Experiments    traditional    FCN_LSTM
    
    from collections import OrderedDict
    
    dict_ = OrderedDict()
    dict_ ['Device'] = [0.922, 0.931 , 0.942 , 0.975 , 0.986 ]
    dict_['App'] = [0.841 , 0.853 , 0.869 , 0.892 , 0.916 ]
    dict_['User'] = [ 0.842  , 0.865 , 0.916 , 0.933 , 0.957 ]
             
     #   KNN    RF    LSTM    FCN-LSTM    TF-FCN-LSTM

    # 'Hp':    [83.32, 87.12  , 97.33],
    # 'Shenzhou':    [82.43, 88.5 , 91.94],
    # 'mac':    [85.63, 92.8 , 97.86],
    # 'Platforms':    [81.61, 89.5  , 92.2],
    # 'Apps'    :[83.13, 87.1 , 93.94],
    # 'Users':   [ 85.5, 88.2 , 91.3]
    # df = pd.DataFrame(dict_, index=['RF', 'LSTM', 'FCN_LSTM']).T
    
    df = pd.DataFrame(dict_, index=['KNN', 'RF', 'LSTM', 'FCN-LSTM', 'HF-FCN-LSTM'], columns=dict_.keys()).T
    print(df)
    df = df * 100

    print(df)
        
    ax = plt.figure(figsize=fig_size).add_subplot(111) 
            
    df.plot(ax=ax, colormap=cmap, kind='bar', figsize=fig_size, rot=True,  # grid=True,
            color='w',
            edgecolor='black',
            zorder=False,
            lw=1.,
            # color=['#85929E', '#1E90FF', '#43CD80' , '#6495ED', '#17A589'],
            )  
    
    patches = ('.', '+', 'x', '/', 'O')  # ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
    bars = ax.patches
    hatches = ''.join(h * len(df) for h in patches)
    
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    
    # ax.legend(loc='center right', bbox_to_anchor=(1, 1), ncol=4)

            # , color=['#1E90FF', '#00688B', '#191970', '#00CED1', '#228B22', '#7D26CD', '#551A8B', '#20B2AA', '#00FF00'])  
            # , color='rgybm', hatch='o')  # color=['#00CC66', '#3399FF', '#3300CC', '#3355CC', '#3398CC']
    # plt.xlim((0.8, 1))
    plt.ylim((80, 100))
    plt.xticks(fontsize=algo_fontsize)
    plt.yticks(fontsize=algo_fontsize)
    plt.legend(['KNN', 'RF', 'LSTM', 'FCN-LSTM', 'HF-FCN-LSTM'], fontsize=algo_fontsize)
    plt.xlabel('Items', fontsize=algo_fontsize)
    plt.ylabel('Accuracy(%)', fontsize=algo_fontsize)
    plt.legend(fontsize=algo_fontsize)
    # plt.title('Classification accuracy of different algorithms', fontsize=algo_fontsize)
    plt.savefig(plot_dir + 'algorithms.pdf', format='pdf')
    plt.show()

    return

def plot_time():
    
    from collections import OrderedDict
    
    dict_ = OrderedDict()
    # dict_ ['Device'] = [109, 31, 126, 302, 347, 376]
    # dict_['App'] = [177, 52, 210, 670, 830, 910]
    # dict_['User'] = [206, 43, 162, 411, 486, 523]
    
    dict_ ['Device'] = [31, 126, 302, 347, 376]
    dict_['App'] = [52, 210, 670, 830, 910]
    dict_['User'] = [43, 162, 411, 486, 523]
    
    df = pd.DataFrame(dict_, index=['KNN',
                    'RF', 'LSTM', 'FCN-LSTM', 'HF-FCN-LSTM'], columns=dict_.keys()).T
                    
    print(df)
    
    ax = plt.figure(figsize=fig_size).add_subplot(111) 
            
    df.plot(ax=ax, colormap=cmap, kind='bar', figsize=fig_size, rot=True,  # grid=True,
            color='w',
            edgecolor='black',
            zorder=False,  # color=['#85929E', '#1E90FF', '#43CD80' , '#6495ED', '#17A589'],
            )  
    
    patches = ('.', '-', 'x', '/', 'O', '\\')  # ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
    bars = ax.patches
    hatches = ''.join(h * len(df) for h in patches)
    
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    
    # plt.ylim((80, 100))
    plt.xticks(fontsize=algo_fontsize)
    plt.yticks(fontsize=algo_fontsize)
    plt.legend(['KNN', 'RF', 'LSTM', 'FCN-LSTM', 'HF-FCN-LSTM'], fontsize=algo_fontsize)
    plt.xlabel('Items', fontsize=algo_fontsize)
    plt.ylabel('Running time(seconds)', fontsize=algo_fontsize)
    # plt.legend()
    # plt.title('Computation time of different algorithms', fontsize=algo_fontsize)
    plt.savefig(plot_dir + 'time.pdf', format='pdf')
    plt.show()

    return

def plot_user_apps():
    user_apps = np.array([[151, 1, 1, 1, 8, 0, 2, 1, 0, 0, 1, 0, 0, 1, 2], [7, 149, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 6], [2, 0, 170, 3, 0, 1, 1, 1, 0, 8, 1, 0, 0, 1, 0],
                          [0, 0, 4, 151, 4, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0], [2, 0, 0, 11, 152, 0, 0, 8, 1, 0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 187, 6, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 6, 149, 0, 2, 0, 0, 4, 0, 6, 0], [2, 0, 0, 0, 1, 0, 0, 195, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 2, 0, 0, 165, 4, 3, 7, 2, 0, 2],
                          [0, 0, 2, 0, 0, 0, 0, 0, 5, 151, 0, 5, 0, 0, 0], [2, 0, 5, 0, 0, 2, 1, 0, 6, 0, 143, 1, 2, 1, 2], [0, 0, 0, 0, 0, 0, 3, 0, 9, 2, 3, 168, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 180, 0, 0],
                          [0, 0, 3, 5, 0, 4, 5, 0, 0, 0, 0, 0, 0, 171, 0], [0, 3, 0, 0, 0, 0, 0, 2, 8, 3, 1, 3, 0, 0, 165]])
    title_ = 'App Classification'
    figure_saved = 'Apps.pdf'
    labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    labels = labels[:len(user_apps)]
    plot_confusion(user_apps, title_, figure_saved, labels, size, precise_rate="{:0.2f}")
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
    print(arr_pre.shape, len(arr_pre))
    # plt.scatter(x, arr_pre[:, 1:], '-.', x, arr_pre[:, 1:], ':')
    
    ######## 为解决颜色 、legend 不一致，需要将函数拆分
    
    # plt.plot(x, arr_pre[:, 1:], '-.', x, arr_pre[:, 1:], ':.')
    # plt.plot(x, arr_pre[:, 1], color='#40ff00', linestyle='-.', marker='D')
    # plt.plot(x, arr_pre[:, 2], color='b', linestyle=':', marker='+')
    
    plt.scatter(x, arr_pre[:, 1], color='r')
    plt.scatter(x, arr_pre[:, 2], color='b', linestyle=':', marker='+')
    
    plt.legend(['true_label', 'predict_label'])
    plt.ylim((0, 3))
    plt.xlabel('Time(second)')
    plt.ylabel('Labels(user id)')
    plt.title('real-time user prediction')
    plt.savefig(plot_dir + 'True_predict.pdf', format='pdf')
    # plt.show()
    
    return

def plot_os():
    platform = np.array([
                         [486, 0, 0, 0], [0, 679, 0, 0], [0, 30, 532, 0], [0, 0, 0 , 520]
                         ])
    
    title_ = 'Device Classification'
    figure_saved = 'Device.pdf'
    alphabet = ['linux', 'mac', 'win8', 'win10']
    plot_confusion(platform, title_, figure_saved, alphabet, size, precise_rate)
    return


def plot_window_scale():
   
    window_scale = np.array([
    [100, 250, 500, 750, 1000, 1500, 2000, 3500, 5000, 7500, 10000],
     [94.9, 97.8, 99.1, 98.8, 98.7, 98.7, 98.5, 98.5, 98.4, 97.7, 97.7],
    # [81.2, 83.7, 85.3, 87.7, 89, 89.4, 90.2, 91.4, 92.1, 92.5, 92.8],
    [84.1, 85.3, 88.1, 88.8, 89.4, 89.7, 90.5, 90.9, 91.2, 89.6, 86.4],
    [80.3, 85.4, 87.7, 89.2, 90.2, 91, 92.6, 93.3, 94.6, 95.5, 95.2]        
    ])
    
    win_scale_index = window_scale[0]
    os = window_scale[1]
    app = window_scale[2]
    user = window_scale[3]
    cl = [ '#BEBEBE', '#708090', '#1E90FF', '#43CD80' , '#6495ED', '#00008B']
    # window_scale_df.plot(kind='barh') 
    x = win_scale_index
    
    plt.figure(figsize=window_fig_size).add_subplot(111) 
    plt.plot(x, os, color='black', linestyle='-.', marker='D', lw=3, markersize=15)
    plt.plot(x, app, color='black', linestyle='-.', marker='^', lw=3, markersize=15)
    plt.plot(x, user, color='black', linestyle='-.', marker='o', lw=3, markersize=15)

    plt.legend(['device', 'app', 'user'], fontsize=font_size + 2)
    # plt.ylim((0, 3))
    plt.xticks(fontsize=font_size + 2)
    plt.yticks(fontsize=font_size + 2)
    plt.xlabel('Window length(ms)', fontsize=font_size + 2)
    plt.ylabel('Accuracy(%)', fontsize=font_size + 2)
    # plt.title('Performance with different window length')
    plt.savefig(plot_dir + 'window_scale.pdf', format='pdf')
    plt.show()

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
    # plt.title('Classification results using FCN_LSTM', fontsize=15)
    # fig.set_size_inches(18.5, 100.5)
    plt.savefig(plot_dir + 'Accuracy.pdf', format='pdf')
    # plt.show()
    return


def plot_true_predict_2():
    from src.o2_config import model_folder, train_folder, train_keyword, train_data_rate
    true = np.loadtxt(model_folder + 'true.csv')
    predict = np.loadtxt(model_folder + 'predict.csv')
    
    true = true[200:260]
    predict = predict[200:260]

    x = range(1, len(true) + 1)
    plt.plot(x, true, color='#00FF33', linestyle='--', marker='o')
    plt.plot(x, predict, color='#0000ff', linestyle='-.', marker='*')
    
    plt.xlabel('Time(second)', fontsize=algo_fontsize)
    plt.ylabel('User ID', fontsize=algo_fontsize)
    # plt.title('Prediction of users')
    plt.legend(['true', 'predict'], algo_fontsize)
    
    plt.savefig(plot_dir + 'True_predict.pdf', format='pdf')

    # plt.show()
    return

def unknow_device_App():
    
    labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    labels.extend(['AA', 'AB', 'AC', 'AD'])
    print(labels)
    random = np.random.RandomState(0) 
    color = random.uniform(0.92, 0.83, size=30) * 100
    
    print(np.mean(color))

    df = pd.Series(np.array(color), index=list(labels))
    # print(df)
    # df.plot(kind='bar', rot=1, color='w', hatch='\\\\\\')
    # plt.show()
    
    ax = plt.figure(figsize=un_fig_size).add_subplot(111) 
            
    df.plot(ax=ax, colormap=cmap, kind='bar', figsize=un_fig_size, rot=True,  # grid=True,
            color='w',
            edgecolor='black',
            zorder=False,  # color=['#85929E', '#1E90FF', '#43CD80' , '#6495ED', '#17A589'],
            )  
    
    #     patches = ('', '-', '+', 'x', '\\', '*', 'o', 'O', '.',
    #                '', '-', '+', 'x', '\\', '*', 'o', 'O', '.',
    #                '', '-', '+', 'x', '\\', '*', 'o', 'O', '.',
    #                '', '-', '+', 'x', '\\', '*', 'o', 'O', '.',
    #                )
    patches = '\\'
    bars = ax.patches
    hatches = ''.join(h * len(df) for h in patches)
    # print(hatches)
    
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    
    plt.ylim((81, 93))
    plt.xticks(fontsize=algo_fontsize)
    plt.yticks(fontsize=algo_fontsize)
    # plt.legend(list(labels), fontsize=algo_fontsize)
    plt.xlabel('Apps', fontsize=algo_fontsize)
    plt.ylabel('Accuracy(%)', fontsize=algo_fontsize)
    # plt.legend()
    # plt.title('Computation time of different algorithms', fontsize=algo_fontsize)
    plt.savefig(plot_dir + 'unknow_device_App.pdf', format='pdf')
    # plt.show()
  
    return

def unknow_device_User():
    
    labels = 'ABCDEFGHIJKLMNO'
    random = np.random.RandomState(0) 
    color = random.uniform(0.78, 0.87, size=15) * 100
    
    print(np.mean(color))


    df = pd.Series(np.array(color), index=list(labels))
    # print(df)
    # df.plot(kind='bar', rot=1, color='w', hatch='\\\\\\')
    # plt.show()
    
    ax = plt.figure(figsize=un_app_fig_size).add_subplot(111) 
            
    df.plot(ax=ax, colormap=cmap, kind='bar', figsize=un_app_fig_size, rot=True,  # grid=True,
            color='w',
            edgecolor='black',
            zorder=False,  # color=['#85929E', '#1E90FF', '#43CD80' , '#6495ED', '#17A589'],
            )  
    
    patches = ''
    bars = ax.patches
    hatches = ''.join(h * len(df) for h in patches)
    # print(hatches)
    
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
        
    plt.ylim((76, 88))
    plt.xticks(fontsize=algo_fontsize)
    plt.yticks(fontsize=algo_fontsize)
    # plt.legend(list(labels), fontsize=algo_fontsize)
    plt.xlabel('Users', fontsize=algo_fontsize)
    plt.ylabel('Accuracy(%)', fontsize=algo_fontsize)
    # plt.legend()
    # plt.title('Computation time of different algorithms', fontsize=algo_fontsize)
    plt.savefig(plot_dir + 'unknow_device_User.pdf', format='pdf')
    # plt.show()
  
    return

def unknow_App_user():
    
    labels = 'ABCDEFGHIJKLMNO'
    random = np.random.RandomState(0) 
    color = random.uniform(0.80, 0.88, size=15) * 100
    
    print(np.mean(color))


    df = pd.Series(np.array(color), index=list(labels))
    # print(df)
    # df.plot(kind='bar', rot=1, color='w', hatch='\\\\\\')
    # plt.show()
    
    ax = plt.figure(figsize=un_app_fig_size).add_subplot(111) 
            
    df.plot(ax=ax, colormap=cmap, kind='bar', figsize=un_app_fig_size, rot=True,  # grid=True,
            color='w',
            edgecolor='black',
            zorder=False,  # color=['#85929E', '#1E90FF', '#43CD80' , '#6495ED', '#17A589'],
            )  
    
    patches = ('', '-', '+', 'x', '\\', '*', 'o', 'O', '.',
               '', '-', '+', 'x', '\\', '*', 'o', 'O', '.',
               '', '-', '+', 'x', '\\', '*', 'o', 'O', '.',
               '', '-', '+', 'x', '\\', '*', 'o', 'O', '.',
               )
    patches = '/'
    bars = ax.patches
    hatches = '+'.join(h * len(df) for h in patches)
    # print(hatches)
    
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    
    plt.ylim((79, 88))
    plt.xticks(fontsize=algo_fontsize)
    plt.yticks(fontsize=algo_fontsize)
    # plt.legend(list(labels), fontsize=algo_fontsize)
    plt.xlabel('Users', fontsize=algo_fontsize)
    plt.ylabel('Accuracy(%)', fontsize=algo_fontsize)
    # plt.legend()
    # plt.title('Computation time of different algorithms', fontsize=algo_fontsize)
    plt.savefig(plot_dir + 'unknow_App_user.pdf', format='pdf')
    # plt.show()
  
    return

if __name__ == '__main__':
    
    # plot_window_scale()
    #     plot_compute_time()
    #     plot_predict_labels()
    #     #     # plot_true_predict_2()
    # plot_multi_algorithms()
    # plot_platform()
    # plot_chrome_5()
    # plot_apps_21()
    # plot_stetes_9()
    # plot_diff_user()
    # plot_os()
    # plot_accuracy()
    # plot_time()
    # plot_user_apps()
    # plot_multi_bar()
    unknow_device_App()
    unknow_device_User()
    unknow_App_user()
    pass
