import os
import warnings

from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Permute
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

from config import model_folder, MAX_NB_VARIABLES
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.generic_utils import load_train_dataset_at, load_test_dataset_at, calculate_dataset_metrics, cutoff_choice, \
                                cutoff_sequence


mpl.style.use('seaborn-paper')

warnings.simplefilter('ignore', category=DeprecationWarning)


def multi_label_log_loss(y_pred, y_true):
    return K.sum(K.binary_crossentropy(y_pred, y_true), axis=-1)


def _average_gradient_norm(model, X_train, y_train, batch_size):
    # just checking if the model was already compiled
    if not hasattr(model, "train_function"):
        raise RuntimeError("You must compile your model before using it.")

    weights = model.trainable_weights  # weight tensors

    get_gradients = model.optimizer.get_gradients(model.total_loss, weights)  # gradient tensors

    input_tensors = [
        # input data
        model.inputs[0],
        # how much to weight each sample by
        model.sample_weights[0],
        # labels
        model.targets[0],
        # train or test mode
        K.learning_phase()
    ]

    grad_fct = K.function(inputs=input_tensors, outputs=get_gradients)

    steps = 0
    total_norm = 0
    s_w = None

    nb_steps = X_train.shape[0] // batch_size

    if X_train.shape[0] % batch_size == 0:
        pad_last = False
    else:
        pad_last = True

    def generator(X_train, y_train, pad_last):
        for i in range(nb_steps):
            X = X_train[i * batch_size: (i + 1) * batch_size, ...]
            y = y_train[i * batch_size: (i + 1) * batch_size, ...]

            yield (X, y)

        if pad_last:
            X = X_train[nb_steps * batch_size:, ...]
            y = y_train[nb_steps * batch_size:, ...]

            yield (X, y)

    datagen = generator(X_train, y_train, pad_last)

    while steps < nb_steps:
        X, y = next(datagen)
        # set sample weights to one
        # for every input
        if s_w is None:
            s_w = np.ones(X.shape[0])

        gradients = grad_fct([X, s_w, y, 0])
        total_norm += np.sqrt(np.sum([np.sum(np.square(g)) for g in gradients]))
        steps += 1

    if pad_last:
        X, y = next(datagen)
        # set sample weights to one
        # for every input
        if s_w is None:
            s_w = np.ones(X.shape[0])

        gradients = grad_fct([X, s_w, y, 0])
        total_norm += np.sqrt(np.sum([np.sum(np.square(g)) for g in gradients]))
        steps += 1

    return total_norm / float(steps)

def get_matrix_OneHot(y_test, re):
    
    '''
        for one_hot processed results , fetch the result matrix including F1 , Accuracy,recall
    '''
    
    actual_y_list, prediction_y_list = [], []
    for item in y_test:
        actual_y_list.append(np.argmax(item))    
    for item in re:
        prediction_y_list.append(np.argmax(item))
    print('True labels: \n', actual_y_list, '\nPredict Labels: \n', prediction_y_list)
    
    return actual_y_list, prediction_y_list

def get_matrix_OneHot_predict(re):
    
    '''
        for one_hot processed results , fetch the result matrix including F1 , Accuracy,recall
    '''
    
    prediction_y_list = []
    for item in re:
        prediction_y_list.append(np.argmax(item))
    #print('\nPredict Labels: \n', prediction_y_list)
    
    return prediction_y_list


def train_model(model:Model, folder_path, dataset_fold_id=None, epochs=50, batch_size=128, val_subset=None,
                cutoff=None, normalize_timeseries=False, learning_rate=1e-3, monitor='val_acc', optimization_mode='max', compile_model=True):
    
    X_train, y_train, X_test, y_test, is_timeseries = load_train_dataset_at(folder_path=folder_path, normalize_timeseries=normalize_timeseries)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    
    max_timesteps, max_nb_variables = calculate_dataset_metrics(X_train)

    if max_nb_variables != MAX_NB_VARIABLES:
        if cutoff is None:
            choice = cutoff_choice(max_nb_variables)
        else:
            assert cutoff in ['pre', 'post'], 'Cutoff parameter value must be either "pre" or "post"'
            choice = cutoff

        if choice not in ['pre', 'post']:
            return
        else:
            X_train, X_test = cutoff_sequence(X_train, X_test, choice, max_nb_variables)

    classes = np.unique(y_train)
    le = LabelEncoder()
    y_ind = le.fit_transform(y_train.ravel())
    recip_freq = len(y_train) / (len(le.classes_) * 
                           np.bincount(y_ind).astype(np.float64))
    class_weight = recip_freq[le.transform(classes)]

    print("Class weights : ", class_weight)

    y_train = to_categorical(y_train, len(np.unique(y_train)))
    y_test = to_categorical(y_test, len(np.unique(y_test)))

    if is_timeseries:
        factor = 1. / np.cbrt(2)
    else:
        factor = 1. / np.sqrt(2)

    weight_fn = "%s/%s_weights.h5" % (model_folder, folder_path.split('/')[-2])

    model_checkpoint = ModelCheckpoint(weight_fn, verbose=1, mode=optimization_mode,
                                       monitor=monitor, save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=100, mode=optimization_mode,
                                  factor=factor, cooldown=0, min_lr=1e-4, verbose=2)
    callback_list = [model_checkpoint, reduce_lr]

    optm = Adam(lr=learning_rate)

    if compile_model:
        model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

    if val_subset is not None:
        X_test = X_test[:val_subset]
        y_test = y_test[:val_subset]

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callback_list,
              class_weight=class_weight, verbose=2, validation_data=(X_test, y_test))


def evaluate_model(model:Model, folder_path, dataset_fold_id=None, batch_size=128, test_data_subset=None,
                   cutoff=None, normalize_timeseries=False):
    X_test, y_test, is_timeseries = load_test_dataset_at(folder_path=folder_path,
                                                          normalize_timeseries=normalize_timeseries)
    
    max_timesteps, max_nb_variables = calculate_dataset_metrics(X_test)

    if max_nb_variables != MAX_NB_VARIABLES:
        if cutoff is None:
            choice = cutoff_choice(max_nb_variables)
        else:
            assert cutoff in ['pre', 'post'], 'Cutoff parameter value must be either "pre" or "post"'
            choice = cutoff

        if choice not in ['pre', 'post']:
            return
        else:
            _, X_test = cutoff_sequence(None, X_test, choice, max_nb_variables)

    if not is_timeseries:
        X_test = pad_sequences(X_test, maxlen=MAX_NB_VARIABLES, padding='post', truncating='post')
    y_test = to_categorical(y_test, len(np.unique(y_test)))

    optm = Adam(lr=1e-3)
    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

    weight_fn = "%s/%s_weights.h5" % (model_folder, folder_path.split('/')[-2])
        
    model.load_weights(weight_fn)

    if test_data_subset is not None:
        X_test = X_test[:test_data_subset]
        y_test = y_test[:test_data_subset]

    print("\nEvaluating : ")
    loss, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size)
    print()
    print("Final Accuracy : ", accuracy)
    
    re = model.predict(X_test)
    
    actual_y_list, prediction_y_list = get_matrix_OneHot(y_test, re)
    
    from sklearn.metrics import confusion_matrix  
    conf_matrix = confusion_matrix(actual_y_list, prediction_y_list)  

    return actual_y_list, prediction_y_list, accuracy, loss, re, conf_matrix
    
def predict_model(model:Model, folder_path, dataset_fold_id=None, batch_size=128, test_data_subset=None,
                   cutoff=None, normalize_timeseries=False):
    X_test, y_test, is_timeseries = load_test_dataset_at(folder_path=folder_path,
                                                          normalize_timeseries=normalize_timeseries)
    
    max_timesteps, max_nb_variables = calculate_dataset_metrics(X_test)

    if max_nb_variables != MAX_NB_VARIABLES:
        if cutoff is None:
            choice = cutoff_choice(max_nb_variables)
        else:
            assert cutoff in ['pre', 'post'], 'Cutoff parameter value must be either "pre" or "post"'
            choice = cutoff

        if choice not in ['pre', 'post']:
            return
        else:
            _, X_test = cutoff_sequence(None, X_test, choice, max_nb_variables)

    if not is_timeseries:
        X_test = pad_sequences(X_test, maxlen=MAX_NB_VARIABLES, padding='post', truncating='post')

    weight_fn = "%s/%s_weights.h5" % (model_folder, folder_path.split('/')[-2])
        
    model.load_weights(weight_fn)

    if test_data_subset is not None:
        X_test = X_test[:test_data_subset]
        y_test = y_test[:test_data_subset]
    
    re = model.predict(X_test)
    prediction_y_list = get_matrix_OneHot_predict(re)

    return prediction_y_list

def set_trainable(layer, value):
   layer.trainable = value

   # case: container
   if hasattr(layer, 'layers'):
       for l in layer.layers:
           set_trainable(l, value)

   # case: wrapper (which is a case not covered by the PR)
   if hasattr(layer, 'layer'):
        set_trainable(layer.layer, value)


def compute_average_gradient_norm(model:Model, dataset_id, dataset_fold_id=None, batch_size=128,
                cutoff=None, normalize_timeseries=False, learning_rate=1e-3):
    X_train, y_train, X_test, y_test, is_timeseries = load_dataset_at(dataset_id,
                                                                      fold_index=dataset_fold_id,
                                                                      normalize_timeseries=normalize_timeseries)
    max_timesteps, sequence_length = calculate_dataset_metrics(X_train)

    if sequence_length != MAX_NB_VARIABLES:
        if cutoff is None:
            choice = cutoff_choice(sequence_length)
        else:
            assert cutoff in ['pre', 'post'], 'Cutoff parameter value must be either "pre" or "post"'
            choice = cutoff

        if choice not in ['pre', 'post']:
            return
        else:
            X_train, X_test = cutoff_sequence(X_train, X_test, choice, sequence_length)

    y_train = to_categorical(y_train, len(np.unique(y_train)))

    optm = Adam(lr=learning_rate)
    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

    average_gradient = _average_gradient_norm(model, X_train, y_train, batch_size)
    print("Average gradient norm : ", average_gradient)


class MaskablePermute(Permute):

    def __init__(self, dims, **kwargs):
        super(MaskablePermute, self).__init__(dims, **kwargs)
        self.supports_masking = True


def f1_score(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)

    return 2 * ((precision * recall) / (precision + recall))


