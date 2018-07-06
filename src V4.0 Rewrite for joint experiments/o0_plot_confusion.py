# coding:utf-8
'''
@time:    Created on  2018-06-20 10:45:08
@author:  Lanqing
@Func:    
'''
import itertools  
import matplotlib.pyplot as plt  
import numpy as np  
  
savefig_folder = 'C:/Users/jhh/Desktop/Magsense/'
  
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):  
    """ 
    This function prints and plots the confusion matrix. 
    Normalization can be applied by setting `normalize=True`. 
    """  
    if normalize:  
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  
        print("Normalized confusion matrix")  
    else:  
        print('Confusion matrix, without normalization')  
  
    print(cm)  
  
    plt.imshow(cm, interpolation='nearest', cmap=cmap)  
    plt.title(title)  
    plt.colorbar()  
    tick_marks = np.arange(len(classes))  
    plt.xticks(tick_marks, classes, rotation=45)  
    plt.yticks(tick_marks, classes)  
  
    fmt = '.2f' if normalize else 'd'  
    thresh = cm.max() / 2.  
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):  
    #    plt.text(j, i, format(cm[i, j], fmt),
    #             horizontalalignment="center",
    #             color="white" if cm[i, j] > thresh else "black")  
  
    plt.tight_layout()  
    plt.ylabel('True label')  
    plt.xlabel('Predicted label')  
    plt.savefig(savefig_folder + 'confusion_matrix.pdf', dpi=200)  
    plt.show()  

def plot_confusion2():
    import numpy as np
    import matplotlib.pyplot as plt
    
    conf_arr = [[33, 2, 0, 0, 0, 0, 0, 0, 0, 1, 3],
                [3, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 4, 41, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 1, 0, 30, 0, 6, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 38, 10, 0, 0, 0, 0, 0],
                [0, 0, 0, 3, 1, 39, 0, 0, 0, 0, 4],
                [0, 2, 2, 0, 4, 1, 31, 0, 0, 0, 2],
                [0, 1, 0, 0, 0, 0, 0, 36, 0, 2, 0],
                [0, 0, 0, 0, 0, 0, 1, 5, 37, 5, 1],
                [3, 0, 0, 0, 0, 0, 0, 0, 0, 39, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38]]
    
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j) / float(a))
        norm_conf.append(tmp_arr)
    
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                    interpolation='nearest')
    
    width, height = conf_arr.shape
    
    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')
    
    cb = fig.colorbar(res)
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])
    plt.savefig('confusion_matrix.png', format='png')

    return

if __name__ == '__main__':
    
    cnf_matrix = np.array([  
        [187, 9, 20, 0, 11, 37, 17, 0, 0, 6, 1, 0, 12, 47, 9, 0, 1, 3, 36, 3, 0, 7, 26, 22, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 2, 3],
        [18, 519, 15, 0, 0, 0, 0, 10, 0, 3, 0, 0, 1, 0, 0, 3, 1, 3, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [32, 20, 413, 0, 0, 0, 0, 0, 1, 0, 0, 0, 5, 1, 1, 0, 0, 0, 5, 0, 0, 0, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 510, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 517, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 0, 0, 0, 0, 465, 1, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [14, 0, 0, 3, 1, 11, 199, 17, 33, 28, 54, 0, 0, 11, 0, 0, 0, 0, 1, 0, 0, 0, 0, 34, 7, 21, 11, 18, 5, 2, 0, 1, 16, 0, 1, 2],
        [1, 4, 0, 0, 0, 0, 14, 402, 0, 1, 5, 0, 2, 0, 0, 13, 0, 0, 0, 0, 0, 0, 3, 0, 5, 2, 5, 4, 10, 16, 4, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 16, 0, 441, 5, 0, 77, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 28, 9, 7, 413, 75, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 29, 1, 0, 50, 412, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 1, 0, 0, 0, 0, 3, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 37, 0, 0, 482, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [16, 0, 6, 0, 0, 0, 1, 2, 18, 0, 0, 0, 177, 22, 8, 0, 0, 0, 21, 41, 43, 0, 28, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 5, 1],
        [7, 0, 0, 0, 17, 38, 5, 0, 0, 0, 0, 0, 1, 248, 0, 0, 0, 0, 12, 0, 0, 0, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 17],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 477, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 460, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 511, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 473, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [31, 2, 3, 0, 0, 0, 0, 0, 2, 0, 0, 0, 6, 5, 19, 3, 0, 49, 354, 0, 0, 1, 28, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 3, 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 403, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 493, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 505, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [11, 0, 7, 0, 0, 0, 3, 0, 0, 0, 0, 0, 31, 10, 9, 0, 0, 0, 16, 0, 9, 10, 406, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 6, 3],
        [11, 0, 0, 16, 8, 18, 30, 0, 44, 0, 0, 26, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 329, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 1, 0],
        [0, 0, 0, 0, 0, 0, 3, 37, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 185, 37, 42, 45, 5, 6, 8, 7, 55, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 5, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 283, 75, 0, 0, 0, 0, 0, 31, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 56, 51, 219, 30, 0, 0, 0, 5, 34, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 1, 279, 46, 15, 11, 42, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 41, 135, 114, 72, 17, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 65, 271, 126, 6, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 42, 106, 264, 6, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 3, 117, 82, 63, 54, 45, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 9, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 68, 20, 54, 0, 0, 0, 0, 0, 295, 0, 0, 0],
        [0, 0, 0, 3, 2, 1, 0, 0, 5, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 352, 33, 10],
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36, 390, 48],
        [5, 1, 0, 0, 4, 2, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 53, 427],
    ])  
      
    class_names = ['l05', 'l07', 'l08', 'l12', 'l13', 'l14', 'l15', 'l16', 'l17', 'l19', 'l20', 'l22',
                   'p05', 'p07', 'p08', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p19', 'p20', 'p22',
                   'y05', 'y07', 'y08', 'y12', 'y13', 'y14', 'y15', 'y16', 'y17', 'y19', 'y20', 'y22']
      
    # plt.figure()  
    # plot_confusion_matrix(cnf_matrix, classes=class_names,  
    #                       title='Confusion matrix, without normalization')  
      
    # Plot normalized confusion matrix  
    plt.figure()  
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')  
