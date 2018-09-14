# coding:utf-8
'''
@time:    Created on  2018-09-11 15:04:53
@author:  Lanqing
@Func:    src.data
'''
import socket
import numpy as np
from src.config import window_length
from src.functions import  logistics
from src.main import predict

t = 0
label_list = []

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
        if pacakage_counter <= window_length:
            one_time_window_data.append(one_package_data)
        else:
            one_window_data = np.array(one_time_window_data).reshape([window_length, len(one_package_data)])
            # print(one_window_data.shape)
            id, label_predict = predict(one_window_data)
            udpSocket.close()
            break
    return one_window_data, id[0], label_predict

def do_post():
    from socket import socket, AF_INET, SOCK_STREAM
    s = socket(AF_INET, SOCK_STREAM)
    ip_port = ('', 8001)
    s.bind(ip_port)
    s.listen(5)
    import time
    while 1:
        conn, addr = s.accept()
        while 1:
            string_post, is_common = real_time()
            if  is_common:
                print('send to client:', string_post)
                conn.sendall(string_post.encode(encoding="utf-8"))
            time.sleep(1)
            
    return

def real_time():
    global t
    global label_list
    #### 处理结果需要加入自定义逻辑
    is_common = False
    top_list = []
    dict_apps = {}
    dict_apps['NoLoad'] = 0
    dict_apps['safari_surfing'] = 1
    dict_apps['tencent_video'] = 2
    dict_apps['zuma_game'] = 3 
    dict_apps['panhao'] = 5 
    dict_apps['lanqing'] = 4   
    prior_ones = 'safari_surfing'
    string_post = 'Current Running APP is:0'
    t += 1
    one_window_data, _, label_predict = real_time_processor()
    label_list.append(label_predict)
    print(label_predict)
    if t % 10 == 0:
        from collections import Counter
        word_counts = Counter(label_list)
        top1 = word_counts.most_common(1)[0][0]
        top1, prior_ones = logistics(top1, label_list, prior_ones, top_list)
        print('APP using now:\t', top1)
        prior_ones = top1
        id_ = dict_apps[top1]
        string_post = 'Current Running APP is:%d' % (id_)
        # print(string_post)
        # do_post(string_post)
        top_list.append(top1)
        is_common = True
        label_list = []
    # plot('%d' % t, one_window_data)
    return string_post, is_common

############## History ################
# history_data_collector('C:/Users/jhh/Desktop/', 'word_panhao.txt')

############## Real Time ##############
while True:
    real_time()

# do_post()
