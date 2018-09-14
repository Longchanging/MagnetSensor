# coding:utf-8
'''
@time:    Created on  2018-09-11 15:04:53
@author:  Lanqing
@Func:    src.data
'''
import socket
import numpy as np
from src.src.config import window_length
from src.src.functions import plot
from src.src.main import predict

def get_num(content):
    one_package_data = []
    prior_byte = 0b0
    for bin_byte in content:
        if (bin_byte >> 1 << 1) != bin_byte:  # high byte
            prior_byte = bin_byte
        else:  # low byte
            num = (prior_byte >> 1 << 7) | ((bin_byte >> 1) & 0b1111111)
            # print(num)
            one_package_data.append(num)
    # print(len(one_package_data))
    return one_package_data

def history_data_collector(file_folder, file_name):
    import time
    udpSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udpSocket.bind(("0.0.0.0", 8000))
    start_time = time.time()
    one_time_window_data = []
    pacakage_counter = 0
    collect_time = 10  # collect data of minutes collect a file
    while(True):
        content, destInfo = udpSocket.recvfrom(2048)    
        one_package_data = get_num(content)
        pacakage_counter += 1
        one_time_window_data.append(one_package_data)
        if pacakage_counter % 1000 == 0:
            time_now = time.time()
            if (time_now - start_time) / 60 > collect_time:
                udpSocket.close()
                break
    np.savetxt(file_folder + file_name, np.array(one_time_window_data))  
    return

def real_time_processor():
    import time
    one_time_window_data = []
    pacakage_counter = 0
    udpSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udpSocket.bind(("0.0.0.0", 8000))
    while(True):
        content, destInfo = udpSocket.recvfrom(2048)    
        one_package_data = get_num(content)
        # print('Got')
        # if np.max(np.array(one_package_data)) < 200:
        pacakage_counter += 1
        if pacakage_counter <= window_length * 2:
            one_time_window_data.append(one_package_data)
        else:
            one_window_data = np.array(one_time_window_data).reshape([window_length * 2, len(one_package_data)])
            print(one_window_data.shape)
            predict(one_window_data)
            udpSocket.close()
            break
    return one_window_data

def real_time():
    t = 0
    while True:
        import time
        t += 5
        # time.sleep(2)
        one_window_data = real_time_processor()
        # plot('%d' % t, one_window_data)
    return

############## Real Time ##############
real_time()

############## History ################
# history_data_collector('C:/Users/jhh/Desktop/History_data/0915/', 'safari_surfing.txt')
