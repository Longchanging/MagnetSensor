# coding:utf-8
import os

def generate_configs():

	base = '../data/' 

	# 更换数据集需要且仅需要改变的参数
	# 程序调整后将根据选择的文件夹进行统一的调整
	train_keyword = {
	    'data_0509':['0-netmusic', '1-youkuapp', '2-tencentweb', '4-surfingweb'],
	    'windows':['0-offlinevideo_vlc', '1-webvideo_tencent', '2-appvideo_aiqiyi', '3-netmusic'],
	    '20180511':['0-netmusic', '1-offlinevideo', '2-surfing'],
	    '20180512':['1_netmusic', '2_chrome_surfing', '3_aqiyi', '4_offline_video_potplayer'],
	    '20180514':['01_aqiyi', '02_offline_video', '03_chrome_surfing', '04_chrome_agar', '05_word', '06_ppt', '07_wechat'],
	    '20180515':['01_aqiyi', '02_offline_video', '03_chrome_surfing', '04_word', '05_ppt'],
	    '20180516':['01_word', '02_ppt', '03_offline_video', '04_aqiyi', '05_chrome_surfing'],
	    '20180517':['01_chrome', '02_ppt', '03_offline', '04_aqiyi'],
	    '20180518':['01_offline_video', '02_chrome_surfing', '03_aqiyi', '04_game_plants'],
	    '20180519':['01_offline_video', '02_aqiyi', '03_chrome_surfing', '04_ppt'],
	    'hp':['surfing', 'web_live_video', 'music', 'offline', 'app_live_video'],
	    'mac':['surfing', 'web_live_video', 'music', 'offline', 'online_video', 'word'],
	    'shenzhou':['surfing', 'music', 'offvideo', 'game', 'word', 'powerpoint'],
	    'platform':['mac_', 'shenzhou_', 'hp_', 'windows_'],
	    'windows_15_category':['01_offline_potvideo', '02_aqiyi', '03_word', '04_excel', '05_ppt', \
					           '06_offline_music_media_player', '07_wangyi_music', '08_open_camera_record_video', \
					           '09_chrome_surfing', '10_wechat', '11_game_plantsVS', '12_baidu_yun', '13_amazon', \
					           '14_online_video_tencent', '15_game_mazu'],
		'windows_office_only':['03_word_edit', '03_word_read', '04_excel_edit', '04_excel_read', \
							'05_ppt_edit', '05_ppt_read'],
		# 'windows_chrome_0522':['01_gmail_read', '02_twitter_normal', '03_youtube', '04_taobao', \
		# 	'05_game_agar', '06_work_github_read', '07_news'],
		'windows_chrome_0522':['01_gmail_read', '02_twitter_normal', '03_youtube', \
			'05_game_agar', '07_news'],
		# 'windows_23_20180523_lanqing':['01_offline_video_potplayer', '02_aqiyi_online_video', '03_wangyi_cloud_online_music', \
		# 	'04_winplayer_offline_music', '05_work_word', '06_work_excel', '07_work_ppt', '08_social_wechat', \
		# 	'09_social_qq', '10_download_baiducloud', '11_camera_windows', '12_game_plants', '13_game_zuma', '14_game_candy', \
		# 	'15_game_minecraft', '16_picture_win3d', '17_chrome_surfing', '18_firefox_surfing', '19_chrome_gmail_work', \
		# 	'20_chrome_twitter', '21_chrome_youtube', '22_chrome_amazon', '23_chrome_agar'],
		# 'windows_23_20180523_lanqing':['01_offline_video_potplayer', \
		# 	'04_winplayer_offline_music', '05_work_word', '06_work_excel', '07_work_ppt', '08_social_wechat', \
		# 	'09_social_qq', '10_download_baiducloud', '11_camera_windows', '12_game_plants', '13_game_zuma', '14_game_candy', \
		# 	'15_game_minecraft', '16_picture_win3d', '17_chrome_surfing', '18_firefox_surfing', '19_chrome_gmail_work', \
		# 	'20_chrome_twitter', '21_chrome_youtube', '22_chrome_amazon', '23_chrome_agar'],
		'windows_23_20180523_lanqing':['video', \
			'music', 'work', 'social', 'download', 'camera', 'game', 'picture', 'surfing'],
		'windows_among_datasets':['video', \
			'music', 'work', 'social', 'game'],
		'wangzhong_20180528':['win_wangzhong__05_work_word', 'win_wangzhong__06_work_excel', 'win_wangzhong__07_work_ppt', \
			# 'win_wangzhong__13_game_zuma', 'win_wangzhong__14_game_candy', 'win_wangzhong__15_game_minecraft', \
			'win_wangzhong__17_chrome_surfing', 'win_wangzhong__18_firefox_surfing', 'win_wangzhong__20_chrome_twitter', \
			'win_wangzhong__21_chrome_youtube'],
		'win_lanqing+panhao':['lanqing__05_work_word', 'lanqing__07_work_ppt', 'lanqing__08_social_wechat', \
							 'lanqing__12_game_plants', 'lanqing__13_game_zuma', \
							  'lanqing__14_game_candy', 'lanqing__15_game_minecraft', 'lanqing__16_picture_win3d', \
							  'lanqing__17_chrome_surfing', 'lanqing__19_chrome_gmail_work', 'lanqing__20_chrome_twitter', \
							   'lanqing__22_chrome_amazon', 'panhao__05_work_word', 'panhao__07_work_ppt', 'panhao__08_social_wechat', \
							    'panhao__12_game_plants', 'panhao__13_game_zuma', 'panhao__14_game_candy', \
							     'panhao__15_game_minecraft', 'panhao__16_picture_win3d', 'panhao__17_chrome_surfing', \
							     'panhao__19_chrome_gmail_work', 'panhao__20_chrome_twitter', 'panhao__22_chrome_amazon'],
		'win_lanqing+panhao+yeqi':['win_lanqing__05_work_word', 'win_lanqing__07_work_ppt', 'win_lanqing__08_social_wechat', \
								'win_lanqing__17_chrome_surfing', 'win_panhao__05_work_word', 'win_panhao__07_work_ppt', \
								'win_panhao__08_social_wechat', 'win_panhao__17_chrome_surfing', \
								'win_yeqi__05_work_word', 'win_yeqi__07_work_ppt', 'win_yeqi__08_social_wechat', \
								 'win_yeqi__17_chrome_surfing'],
		'win_lanqing+panhao+yuhui':['win_lanqing__05_work_word', 'win_lanqing__07_work_ppt', 'win_lanqing__08_social_wechat', \
			'win_lanqing__12_game_plants', 'win_lanqing__13_game_zuma', \
			'win_lanqing__14_game_candy', 'win_lanqing__15_game_minecraft', 'win_lanqing__16_picture_win3d', \
			'win_lanqing__17_chrome_surfing', 'win_lanqing__19_chrome_gmail_work', 'win_lanqing__20_chrome_twitter', \
			'win_lanqing__22_chrome_amazon', 'win_panhao__05_work_word', 'win_panhao__07_work_ppt', \
			'win_panhao__08_social_wechat', 'win_panhao__12_game_plants', \
			'win_panhao__13_game_zuma', 'win_panhao__14_game_candy', 'win_panhao__15_game_minecraft', \
			'win_panhao__16_picture_win3d', 'win_panhao__17_chrome_surfing', 'win_panhao__19_chrome_gmail_work', \
			'win_panhao__20_chrome_twitter', 'win_panhao__22_chrome_amazon', 'win_yuhui__05_work_word', 'win_yuhui__07_work_ppt', \
			'win_yuhui__08_social_wechat', 'win_yuhui__12_game_plants', 'win_yuhui__13_game_zuma', \
			'win_yuhui__14_game_candy', 'win_yuhui__15_game_minecraft', 'win_yuhui__16_picture_win3d', 'win_yuhui__17_chrome_surfing', \
			'win_yuhui__19_chrome_gmail_work', 'win_yuhui__20_chrome_twitter', 'win_yuhui__22_chrome_amazon'],
	
	'lanqing_20180523':['win_lanqing__05_work_word', 'win_lanqing__07_work_ppt', 'win_lanqing__08_social_wechat', \
						'win_lanqing__09_social_qq', 'win_lanqing__12_game_plants', 'win_lanqing__13_game_zuma', \
						'win_lanqing__14_game_candy', 'win_lanqing__15_game_minecraft', 'win_lanqing__16_picture_win3d', \
						'win_lanqing__17_chrome_surfing', 'win_lanqing__19_chrome_gmail_work', \
						'win_lanqing__20_chrome_twitter', 'win_lanqing__22_chrome_amazon'],
	  
	'panhao_20180524':['win_panhao__05_work_word', 'win_panhao__07_work_ppt', 'win_panhao__08_social_wechat', \
			'win_panhao__09_social_qq', 'win_panhao__12_game_plants', 'win_panhao__13_game_zuma', \
			'win_panhao__14_game_candy', 'win_panhao__15_game_minecraft', 'win_panhao__16_picture_win3d', \
			'win_panhao__17_chrome_surfing', 'win_panhao__18_firefox_surfing', \
			'win_panhao__19_chrome_gmail_work', 'win_panhao__20_chrome_twitter', \
			'win_panhao__21_chrome_youtube', 'win_panhao__22_chrome_amazon'],
				  
	'yuhui_20180527':['win_yuhui__05_work_word', 'win_yuhui__06_work_excel', 'win_yuhui__07_work_ppt', \
			'win_yuhui__08_social_wechat', 'win_yuhui__09_social_qq', 'win_yuhui__12_game_plants', \
			'win_yuhui__13_game_zuma', 'win_yuhui__14_game_candy', 'win_yuhui__15_game_minecraft', \
			'win_yuhui__16_picture_win3d', 'win_yuhui__17_chrome_surfing', 'win_yuhui__18_firefox_surfing', \
			'win_yuhui__19_chrome_gmail_work', 'win_yuhui__20_chrome_twitter', 'win_yuhui__21_chrome_youtube', \
			'win_yuhui__22_chrome_amazon', 'win_yuhui__23_chrome_agar'],
				  
	'wangzhong_20180528':['win_wangzhong__05_work_word', 'win_wangzhong__06_work_excel', 'win_wangzhong__07_work_ppt', \
		'win_wangzhong__13_game_zuma', 'win_wangzhong__14_game_candy', 'win_wangzhong__15_game_minecraft', \
		'win_wangzhong__17_chrome_surfing', 'win_wangzhong__18_firefox_surfing', 'win_wangzhong__20_chrome_twitter', \
		'win_wangzhong__21_chrome_youtube'],
	
	'yeqi_20180526':['win_yeqi__05_work_word', 'win_yeqi__06_work_excel', 'win_yeqi__07_work_ppt', \
		'win_yeqi__08_social_wechat', 'win_yeqi__09_social_qq', 'win_yeqi__13_game_zuma', \
		'win_yeqi__14_game_candy', 'win_yeqi__16_picture_win3d', 'win_yeqi__17_chrome_surfing', \
		'win_yeqi__18_firefox_surfing', 'win_yeqi__19_chrome_gmail_work', \
		'win_yeqi__20_chrome_twitter', 'win_yeqi__21_chrome_youtube', \
		'win_yeqi__22_chrome_amazon', 'win_yeqi__23_chrome_agar'],
	}

	print('\n -------------------------------------------------------------------- \n')
	print('Please input your command: default train and test and predict and baseline and plot,')
	print('you can also type like: "train_predict_baseline" or "train_baseline"')
	print('You can choose the folders like this: \n', train_keyword.keys())
	print('\n -------------------------------------------------------------------- \n')

	# 提醒用户输入参数
	str_input = input('Input your command of folder now: ')
	train_keyword_ = train_keyword[str(str_input)]

	# 第一段程序用来处理用户传来的文件夹，即第二个参数
	print('The folder you want to process is:\t', str_input)
	print('The train_keyword are:\t', train_keyword_)

	# 根据传参文件夹决定主要目录
	train_folder = test_folder = predict_folder = base + '/input/' + '/' + str_input + '/'

	# 根据传参进来的上述文件夹生成其他文件夹
	base = base + '/tmp/' + str_input + '/'
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

	dict_configs = {
		'str1' :"train_keyword = %s" % str(train_keyword_),
		'str2' : "train_folder = '%s'" % str(train_folder),
		'str3' : "test_folder = '%s'" % str(test_folder),
		'str4' : "predict_folder = '%s'" % str(predict_folder),
		'str5' : "train_tmp = '%s'" % str(train_tmp),
		'str6' : "test_tmp = '%s'" % str(test_tmp),
		'str7' : "predict_tmp = '%s'" % str(predict_tmp),
		'str8' : "train_tmp_test = '%s'" % str(train_tmp_test),
		'str9' : "model_folder = '%s'" % str(model_folder),
		'str10': "NB_CLASS = %d" % len(train_keyword_)
	}

	# 先检查原文件是否含有冗余信息，有则删掉
	import shutil
	with open('o2_config.py', 'r', encoding='utf-8') as f:
		with open('o2_config.py.new', 'w', encoding='utf-8') as g:
			for line in f.readlines():
				if "train_keyword" not in line and "train_folder" not in line and "test_folder" not in line \
	                and "predict_folder" not in line and "train_tmp" not in line and "test_tmp" not in line \
	                and "predict_tmp" not in line and "train_tmp_test" not in line and "model_folder" not in line \
	                and "NB_CLASS" not in line:             
					g.write(line)
	shutil.move('o2_config.py.new', 'o2_config.py')

	# 将更改写入config 文件
	fid = open('o2_config.py', 'a')
	for i in range(10):
		fid.write(dict_configs['str' + str(i + 1)])
		fid.write('\n')

	return

if __name__ == '__main__':
	generate_configs()
