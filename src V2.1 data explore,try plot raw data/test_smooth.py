# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 12:37:30 2016
@author: user
"""

import os

import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import filters

import matplotlib.pyplot as plt
import numpy as np

base = '../data/' 
train_folder, test_folder, predict_folder = base + '/input/20180511/', base + '/input/20180511/', base + '/input/20180511/'  # 机型

rootdir = 'G:/Users/SensorData/FT2232H/DeviceData/20180512/1/'
rootdir = test_folder
listdir = os.listdir(rootdir)

for item in listdir:

    storetxtpath = rootdir + "/" + item
    print(storetxtpath)

    txtfile = open(storetxtpath, 'r') 
     
    dataList = []

    index = 0
    while 1:
        try:
            lineInfo = txtfile.readline()
            if(len(lineInfo) < 26):
                # Time Information
                if(lineInfo == ''):
                    break
                pass
            else:
                # Sensor Data
                if(lineInfo != ''):
                    sensorDataList = lineInfo.split(',')
                try:
                    for eachData in sensorDataList:
                        tmpvalue = int(eachData)
                        if(tmpvalue != 2018)and(tmpvalue < 50000):
                               # print(tmpvalue)
                               dataList.append(tmpvalue)
                except:
                    pass
        except:
            break
        
    txtfile.close()
    print (len(dataList))
    
    datalength = len(dataList)
    plotvaluelist = []
    
    for i in range(datalength):  # (plotnumber):
        plotvaluelist.append(dataList[i] / 65536)

    print('\n', len(plotvaluelist))
    
    gaussianFactor = 500
    sensorRawDataList_GaussianFilter = filters.gaussian_filter1d(plotvaluelist, gaussianFactor)
    
    import matplotlib.pyplot as plt        
    plt.plot(range(len(sensorRawDataList_GaussianFilter[:100000])), sensorRawDataList_GaussianFilter[:100000])
    plt.show()

    
    # sensorRawDataList_GaussianFilter = plotvaluelist
    newsensorRawDataList_GaussianFilter = []
    newindexlist = []
    count = 0
    # print(len(sensorRawDataList_GaussianFilter))
    for i in range(len(sensorRawDataList_GaussianFilter)):
        if(i % 100 == 0):
            # print(i)
            newsensorRawDataList_GaussianFilter.append(sensorRawDataList_GaussianFilter[i])
            newindexlist.append(count)
            count += 1

#     name = item.split('.')[0]
#     pdfString = rootdir + "/" + name + '.pdf'
#     print(pdfString)
#     pp = PdfPages(pdfString)
# 
#     matplotlib.rcParams['agg.path.chunksize'] = 10000   
#     yDown = 0
#     yUp = 0
# 
#     yDown = min(newsensorRawDataList_GaussianFilter)
# 
#     yUp = max(newsensorRawDataList_GaussianFilter)
# 
# 
#     print(len(newindexlist), len(newsensorRawDataList_GaussianFilter))
# 
#     fig = plt.figure()      
#     ax = fig.add_subplot(111) 
#     fig.set_size_inches(240, 20)  # 18.5, 10.5
#     plt.axis([0, len(newindexlist), yDown, yUp])  # 0.240 0.2455 50ms #0.2375, 0.245 100ms
#     plt.title("MagnValue vs. Sampling", size=80)
#     plt.xlabel("Sample Time ", size=50)
#     plt.ylabel("MagnSignal(V)", size=40)   
#     ax.plot(newindexlist, newsensorRawDataList_GaussianFilter, 'b-')
#     ax.legend() 
#     plt.savefig(pp, format='pdf')
#     
#     pp.close()
#     txtfile.close()
