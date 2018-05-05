
from step1.config import *
from step1.preprocess import preprocess
from step1.read_data import read__data
from step2.config import *
from step2.prepare import train_test, predict
from step2.train_test import train_, test_, predict_

def main_train():
    read__data(input_folder, trainTestdifferent_category, train_percent2read_afterPCA_data, train_after_fft_data)
    preprocess('train', train_after_fft_data, trainTestdifferent_category, train_after_pca_data)
    train_test(train_test_data_folder) 
    train_()
    test_()
    return

def main_predict():
    read__data(predict_folder, predict_different_category, 1, predict_fft_data)
    preprocess('predict', predict_fft_data, predict_different_category, predict_pca_data)
    predict(predict_data_folder)
    predict_()
    return

main_train()
main_predict()