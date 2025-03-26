import os
import sys
import copy
import psutil
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import multiprocessing as mp

from time import sleep
from contextlib import contextmanager
import yagmail

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor


class GpuQueue:

    def __init__(self, N_GPUS, all_gpus=False):
        self.queue = mp.Manager().Queue()
        if all_gpus:
            all_idxs = list(range(N_GPUS))
        else:
            all_idxs = [N_GPUS]
        for idx in all_idxs:
            self.queue.put(idx)

    @contextmanager
    def one_gpu_per_process(self):
        current_idx = self.queue.get()
        yield current_idx
        self.queue.put(current_idx)


def remove_zeros_load_ts(static_data, y):
    if static_data['type'] == 'load':
        values_sorted = np.sort(y.values.ravel())
        min_value = 0
        for i in range(y.values.shape[0]):
            if values_sorted[i] > 0:
                min_value = values_sorted[i]
                break
        return y.clip(min_value, np.inf)
    else:
        return y


def enhance_model_layers_distributed_data(what_data, model_layers):
    if what_data == 'row_dict_distributed':
        model_layers['input'] = [('lstm', 1)] + model_layers['input']
    return model_layers


def enhance_model_layers_multi_output(static_data, experiment_tag, experiment, probabilistic):
    if 'cnn' in experiment_tag and static_data['horizon_type'] == 'multi-output' \
            and not probabilistic:
        input_layers = experiment['input']
        input_layers_new = list()
        layer_previous = None
        for layer in input_layers:
            if 'conv' in layer[0]:
                layer_previous = copy.deepcopy(layer)
                input_layers_new.append((f'time_distr_{layer[0]}', layer[1]))
            else:
                if layer_previous is not None:
                    if 'conv' in layer_previous[0]:
                        layer_previous = copy.deepcopy(layer)
                        input_layers_new.append((f'lstm', 1))
                        input_layers_new.append(layer)
                    else:
                        layer_previous = copy.deepcopy(layer)
                        input_layers_new.append(layer)
        experiment['input'] = input_layers_new
    return experiment


def fix_convolutional_names(experiment_tag, experiment, conv_dim):
    if 'cnn' in experiment_tag:
        for group, branch in experiment.items():
            for i, layer in enumerate(branch):
                if 'conv' in layer[0]:
                    if conv_dim is not None:
                        experiment[group][i] = (f'conv_{conv_dim}d', layer[1])
                    else:
                        raise ValueError('Cannot find conv_dim parameter')
    return experiment


def feed_data(batch, x, y, data, target, lr_pl, lr):
    feed_dict = dict()
    if isinstance(x, dict):
        for key in x.keys():
            if isinstance(x[key], dict):
                for key_1 in x[key].keys():
                    feed_dict.update({x[key][key_1]: data[key][key_1][batch]})
            else:
                feed_dict.update({x[key]: data[key][batch]})
    else:
        feed_dict.update({x: data[batch]})
    feed_dict.update({y: target[batch]})
    feed_dict.update({lr_pl: lr})
    return feed_dict


def feed_data_eval(x, data):
    feed_dict = dict()
    if isinstance(x, dict):
        for key in x.keys():
            if isinstance(x[key], dict):
                for key_1 in x[key].keys():
                    feed_dict.update({x[key][key_1]: data[key][key_1]})
            else:
                feed_dict.update({x[key]: data[key]})
    else:
        feed_dict.update({x: data})
    return feed_dict


def distance(obj_new, obj_old, obj_max, obj_min, weights=None):
    if np.any(np.isinf(obj_old)):
        obj_old = obj_new.copy()
        obj_max = obj_new.copy()
        return True, obj_old, obj_max, obj_min
    if np.any(np.isinf(obj_min)) and not np.all(obj_max == obj_new):
        obj_min = obj_new.copy()
    d = 0
    for i in range(obj_new.shape[0]):
        if obj_max[i] < obj_new[i]:
            obj_max[i] = obj_new[i]
        if obj_min[i] > obj_new[i]:
            obj_min[i] = obj_new[i]
        if weights is None:
            if obj_max[i] - obj_min[i] < 1e-6:
                d += (obj_new[i] - obj_old[i])
            else:
                d += (obj_new[i] - obj_old[i]) / (obj_max[i] - obj_min[i])
        else:
            if obj_max[i] - obj_min[i] < 1e-6:
                d += weights[i] * ((obj_new[i] - obj_old[i]))
            else:
                d += weights[i] * ((obj_new[i] - obj_old[i]) / (obj_max[i] - obj_min[i]))

    if weights is not None:
        d = d / np.sum(weights)
    if d < 0:
        obj_old = obj_new.copy()
        return True, obj_old, obj_max, obj_min
    else:
        return False, obj_old, obj_max, obj_min


def split_validation_set(x):
    if isinstance(x, dict):
        for values in x.values():
            x1 = copy.deepcopy(values)
            break
    else:
        x1 = copy.deepcopy(x)
    if x1.shape[0] > 1000:
        partitions = 250
        ind_list = []
        ind = np.arange(x1.shape[0])
        for i in range(0, x1.shape[0], partitions):
            if (i + partitions + 1) > x1.shape[0]:
                ind_list.append(ind[i:])
            else:
                ind_list.append(ind[i:i + partitions])
    else:
        ind_list = [np.arange(x1.shape[0])]
    return ind_list


def calculate_cpus(n_cpus):
    warm = psutil.cpu_percent()
    average_load = np.mean(psutil.cpu_percent(interval=5, percpu=True)[:n_cpus])

    return n_cpus - int(n_cpus * average_load / 100)


def get_tf_config(n_jobs):
    n_cpus = calculate_cpus(n_jobs)
    if sys.platform != 'linux' and n_cpus > int(n_jobs / 3):
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                          intra_op_parallelism_threads=2,
                                          inter_op_parallelism_threads=2)
        config.gpu_options.allow_growth = True
    else:
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
    return config


def lr_schedule(epoch, lr=1e-04):
    if epoch < 50:
        # WarmUp
        return np.linspace(lr / 10, lr, 50)[epoch]
    else:
        lr_step = 0.5 * lr * (1 + np.cos(np.pi * (epoch - 50) / float(20)))
        return np.maximum(lr / 10, lr_step)


def check_if_all_nans(activations, thres_act, return_len_nan=False):
    indices = np.where(np.all(activations.values < thres_act, axis=1))[0]
    len_nan = 0
    if indices.shape[0] > 0:
        len_nan = indices.shape[0]
        for ind in indices:
            act = activations.loc[ind]
            clust = act.idxmax()
            activations.loc[ind, clust] = thres_act
    if return_len_nan:
        return activations, len_nan
    else:
        return activations


def linear_output(X_train, X_val, X_test, y_train, y_val, y_test, act_train, act_val, act_test, params):
    rules = params['rules']
    act_train = pd.DataFrame(act_train, columns=rules)
    act_train = check_if_all_nans(act_train, params['thres_act'])
    act_val = pd.DataFrame(act_val, columns=rules)
    act_val, len_nan_val = check_if_all_nans(act_val, params['thres_act'], return_len_nan=True)
    act_test = pd.DataFrame(act_test, columns=rules)
    act_test, len_nan_test = check_if_all_nans(act_test, params['thres_act'], return_len_nan=True)
    columns = [col for col in X_train.keys() if col not in {'clustering'} and 'act' not in col]
    x_ = np.copy(X_train[columns[0]])
    x_val_ = np.copy(X_val[columns[0]])
    x_test_ = np.copy(X_test[columns[0]])
    if len(x_.shape) > 2:
        shape = x_.shape
        x_ = x_.reshape(-1, np.prod(shape[1:]))
        x_val_ = x_val_.reshape(-1, np.prod(shape[1:]))
        x_test_ = x_test_.reshape(-1, np.prod(shape[1:]))
    if x_.shape[1] > 5:
        x_ = x_[:, :6]
        x_val_ = x_val_[:, :6]
        x_test_ = x_test_[:, :6]
    x = pd.DataFrame(x_)
    x_val = pd.DataFrame(x_val_)
    x_test = pd.DataFrame(x_test_)
    y = pd.DataFrame(y_train[:, 0])
    y_val = pd.DataFrame(y_val[:, 0])
    y_test = pd.DataFrame(y_test[:, 0])
    lin_models = dict()
    total = 0
    for rule in act_train.columns:
        indices = act_train[rule].index[act_train[rule] >= params['thres_act']].tolist()
        if len(indices) != 0:
            X1 = x.loc[indices].values
            y1 = y.loc[indices].values

            lin_models[rule] = LinearRegression().fit(X1, y1.ravel())

    preds = pd.DataFrame(index=x_val.index, columns=sorted(lin_models.keys()))
    for rule in rules:
        indices = act_val[rule].index[act_val[rule] >= params['thres_act']].tolist()
        if len(indices) != 0 and rule in lin_models.keys():
            X1 = x_val.loc[indices].values
            preds.loc[indices, rule] = lin_models[rule].predict(X1).ravel()

    pred = preds.mean(axis=1)
    rated = 1
    err_val = (pred.values.ravel() - y_val.values.ravel()) / rated

    preds = pd.DataFrame(index=x_test.index, columns=sorted(lin_models.keys()))
    for rule in rules:
        indices = act_test[rule].index[act_test[rule] >= params['thres_act']].tolist()
        if len(indices) != 0 and rule in lin_models.keys():
            X1 = x_test.loc[indices].values
            preds.loc[indices, rule] = lin_models[rule].predict(X1).ravel()

    pred = preds.mean(axis=1)
    err_test = (pred.values.ravel() - y_test.values.ravel()) / rated
    mae_val = np.mean(np.abs(err_val)) + len_nan_val
    mae_test = np.mean(np.abs(err_test)) + len_nan_test
    sse_val = np.sum(np.square(err_val)) + len_nan_val
    sse_test = np.sum(np.square(err_test)) + len_nan_test
    return mae_val, mae_test, sse_val, sse_test


def find_free_cpus(path_group):
    free_cpus = 0
    warm = psutil.cpu_percent(percpu=True)

    while free_cpus < 2:
        sleep(5)
        load = psutil.cpu_percent(interval=None, percpu=True)
        n_cpus = len(load)
        available_load = n_cpus - int(n_cpus * np.mean(load) / 100) - 1
        print(f'find total {n_cpus} cpus,  mean load {np.mean(load)}, non_cpus {1}')
        gpu_status = joblib.load(os.path.join(path_group, 'gpu_status.pickle'))
        free_cpus = n_cpus - 1 if gpu_status == 0 else int(n_cpus / 2)
        print(
            f'Find load {int(n_cpus * np.mean(load) / 100)}, available {available_load} cpus, gpu_status {gpu_status},'
            f' {free_cpus} cpus free')
        if free_cpus > available_load:
            free_cpus = available_load
    print(f'Find {free_cpus} cpus free')
    return free_cpus


def send_predictions(message):
    contents = message
    # The mail addresses and password
    sender_address = 'gsdrts@yahoo.gr'
    sender_pass = 'pubmqkxfdtpqtwws'
    yag_smtp_connection = yagmail.SMTP(user=sender_address, password=sender_pass, host='smtp.mail.yahoo.com')
    subject = f'Error for check'
    yag_smtp_connection.send(to='joesider9@gmail.com', subject=subject, contents=contents)
