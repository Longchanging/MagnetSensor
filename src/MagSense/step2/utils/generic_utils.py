import os

import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from step2.config import MAX_NB_VARIABLES


mpl.style.use('seaborn-paper')

def load_train_dataset_at(folder_path, normalize_timeseries=False, verbose=True) -> (np.array, np.array):
    if verbose: print("Loading train dataset : ", folder_path, folder_path)

    x_train_path = folder_path + "X_train.npy"
    y_train_path = folder_path + "y_train.npy"
    x_test_path = folder_path + "X_test.npy"
    y_test_path = folder_path + "y_test.npy"

    if os.path.exists(x_train_path):
        X_train = np.load(x_train_path)
        y_train = np.load(y_train_path)
        X_test = np.load(x_test_path)
        y_test = np.load(y_test_path)
    elif os.path.exists(x_train_path[1:]):
        X_train = np.load(x_train_path[1:])
        y_train = np.load(y_train_path[1:])
        X_test = np.load(x_test_path[1:])
        y_test = np.load(y_test_path[1:])

    is_timeseries = True

    # extract labels Y and normalize to [0 - (MAX - 1)] range
    nb_classes = len(np.unique(y_train))
    y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (nb_classes - 1)

    if is_timeseries:
        # scale the values
        if normalize_timeseries:
            X_train_mean = X_train.mean()
            X_train_std = X_train.std()
            X_train = (X_train - X_train_mean) / (X_train_std + 1e-8)

    if verbose: print("Finished processing train dataset..")

    # extract labels Y and normalize to [0 - (MAX - 1)] range
    nb_classes = len(np.unique(y_test))
    y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (nb_classes - 1)

    if is_timeseries:
        # scale the values
        if normalize_timeseries:
            X_test = (X_test - X_train_mean) / (X_train_std + 1e-8)

    if verbose:
        print("Finished loading test dataset..")
        print()
        print("Number of train samples : ", X_train.shape[0], "Number of test samples : ", X_test.shape[0])
        print("Number of classes : ", nb_classes)
        print("Sequence length : ", X_train.shape[-1])

    return X_train, y_train, X_test, y_test, is_timeseries

def load_test_dataset_at(folder_path, normalize_timeseries=False, verbose=True) -> (np.array, np.array):
    if verbose: 
        print("Loading train / test dataset : ", folder_path)

    if folder_path:
        x_test_path = folder_path + "X_test.npy"
        y_test_path = folder_path + "y_test.npy"
    X_test = np.load(x_test_path)
    y_test = np.load(y_test_path)

    is_timeseries = True

    # extract labels Y and normalize to [0 - (MAX - 1)] range
    nb_classes = len(np.unique(y_test))
    y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (nb_classes - 1)
        
    print("Testing data shape:", X_test.shape, y_test.shape)

    return X_test, y_test, is_timeseries


def calculate_dataset_metrics(X_train):
    max_nb_variables = X_train.shape[1]
    max_timesteps = X_train.shape[-1]

    return max_timesteps, max_nb_variables


def cutoff_choice(sequence_length):
    print("Original sequence length was :", sequence_length, "New sequence Length will be : ",
          MAX_NB_VARIABLES)
    choice = input('Options : \n'
                   '`pre` - cut the sequence from the beginning\n'
                   '`post`- cut the sequence from the end\n'
                   '`anything else` - stop execution\n'
                   'To automate choice: add flag `cutoff` = choice as above\n'
                   'Choice = ')

    choice = str(choice).lower()
    return choice


def cutoff_sequence(X_train, X_test, choice, sequence_length):
    assert MAX_NB_VARIABLES < sequence_length, "If sequence is to be cut, max sequence" \
                                                                   "length must be less than original sequence length."
    cutoff = sequence_length - MAX_NB_VARIABLES
    if choice == 'pre':
        if X_train is not None:
            X_train = X_train[:, :, cutoff:]
        if X_test is not None:
            X_test = X_test[:, :, cutoff:]
    else:
        if X_train is not None:
            X_train = X_train[:, :, :-cutoff]
        if X_test is not None:
            X_test = X_test[:, :, :-cutoff]
    print("New sequence length :", MAX_NB_VARIABLES)
    return X_train, X_test


if __name__ == "__main__":
    pass
