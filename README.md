# MagnetSensor
using deep learning model with magnetism data to classify different platform and applications.

## Author
Lanqing Yang,SJTU mapleylq@gmail.com     yanglanqing@sjtu.edu.cn

Hao Pan,STTU 

## Folder

##Source code version description##

##V0.0 LSTM&baseline##

src V0.0 Initial LSTM&baseline

	mainly other_classifier.py    (√) 
	config.py                     (√)     
	LSTM.py                       (√)
	Model.py                      (√)
	preprocess.py                 (√)
	read_data.py                  (√)

	other_classifier.py     use other models to do classification
	config.py     important paras for LSTM and baseline
	LSTM.py       main LSTM function to classify
	read_data.py  read data and simply preprocessing(fft,feature engineering)
	preprocess    PCA and min-max preprocessing
	Model.py      construct LSTM model using tf



##V1.0##

src V1.0 FCN Model

	#FCN folder#
	config.py           (√)
	mydata_model.py     (√)
	predict.py          (√)
	prepare_data.py     (√)
	
	#LSTM folder#
	mainly other_classifier.py   
	config.py     
	LSTM.py           
	Model.py          
	preprocess.py     
	read_data.py       

	config.py          config just of FCN
	mydata_model.py    main FCN model
	predict.py         use FCN to predict
	prepare_data.py    prepare data for FCN
	
	
	#LSTM folder#      use them to generate data for FCN and baseline

##V2.0##

src V2.0 feature Extraction

	config.py
	feature_core.py          (√)
	feature_fft.py           (√)
	feature_reExtraction.py  (√)
	feature_time.py          (√)
	train_test.py            (√)

This folder is forked here 

	https://www.zhihu.com/question/41068341?sort=created 
	
Not yet completed,just a trial.

##V2.1##

src V2.1 data explore,try plot raw data

	plot.py                  (√)
	sample_test.py           (√)
	test_smooth.py           (√)

	Failure trial,try plot normal raw data.
	To explore more about min-max model,and others.
	Finally achieved in V3.1  


##V3.0##

src V3.0 Add main,setup,argv
src V3.0 Combine fcn&baseline,call together
Note: From this version, folder structure is quiet different.  

	baseline.py   
	config.py
	main.py             (√)
	plot.py
	prepare.py
	preprocess.py
	read_data.py 
	setup.py            (√)
	train_test.py

	main.py         Combine all functions together
			Entrance of all
			argv included to get user command like 'train/test/predict' 

	setup.py     Generate configs for config.py 
	   

##V3.1##

 src V4.1 Changed package name for convenience

	o0_DataExploer.py  mainly to explore the data and plot some pdfs.
 
	o0_DataExploer.py   (√)
	o0_dataExplore.py   (√)
	o0_plot.py          (√)
	o1_setup.py
	o2_config.py
	o3_read_data.py
	o4_preprocess.py
	o5_prepare.py
	o6_train_test.py
	o7_baseline.py
	o8_main.py
