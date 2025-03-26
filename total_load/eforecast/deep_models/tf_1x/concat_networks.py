import copy
import gc
import os
import random
import time

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from tqdm import tqdm

from eforecast.common_utils.tf_1x_utils import convert_lstm_weights
from eforecast.common_utils.train_utils import distance
from eforecast.common_utils.train_utils import feed_data_eval
from eforecast.common_utils.train_utils import get_tf_config
from eforecast.common_utils.train_utils import lr_schedule
from eforecast.common_utils.train_utils import remove_zeros_load_ts
from eforecast.common_utils.train_utils import enhance_model_layers_distributed_data
from eforecast.common_utils.train_utils import fix_convolutional_names
from eforecast.common_utils.train_utils import enhance_model_layers_multi_output
from eforecast.deep_models.tf_1x.builders import build_graph_for_concat_nets
from eforecast.deep_models.tf_1x.builders import build_output_for_concat_nets
from eforecast.deep_models.tf_1x.builders import create_placeholders_for_concat_nets
from eforecast.deep_models.tf_1x.optimizers import optimize
from eforecast.deep_models.tf_1x.trainer import gather_weights
from eforecast.deep_models.tf_1x.trainer import train_step
from eforecast.deep_models.tf_1x.trainer import validation_step

pd.set_option('display.expand_frame_repr', False)


class DeepConcatNetwork:
    def __init__(self, static_data, path_weights, net_names=None, params=None, probabilistic=False, refit=False):
        self.rules = None
        self.results = None
        self.best_sse_val = None
        self.best_sse_test = None
        self.best_min_act = None
        self.best_max_act = None
        self.best_mean_act = None
        self.best_sum_act = None
        self.best_mae_val = None
        self.best_mae_test = None
        self.best_weights = None
        self.n_batch = None
        self.n_out = None
        self.is_trained = False
        self.refit = refit
        self.probabilistic = probabilistic
        self.static_data = static_data
        self.rated = static_data['rated']
        if params is not None:
            self.net_names = net_names
            self.params = params
            self.method = self.params['method']
            self.name = self.params['name']
            self.epochs = self.params['max_iterations']
            self.learning_rate = self.params['learning_rate']
            self.batch_size = self.params['batch_size']
        self.path_weights = path_weights

        try:
            if not self.refit:
                self.load()
        except:
            pass
        self.refit = refit
        if not hasattr(self, 'params'):
            raise RuntimeError('The network has no params. You should train the net providing params')

    def get_slice(self, x, mask, meta_data, y=None, act=None, rules=None, split_primary_key=False):
        group_layers = dict()
        X_slice = dict()
        dates = meta_data['dates']
        mask = mask.intersection(dates)
        for net_name in self.net_names:
            if act is not None:
                if net_name in act.keys():
                    mask = mask.intersection(act[net_name].index)
        indices = dates.get_indexer(mask)
        y_slice = y.iloc[indices].values if y is not None else None
        for net_name in self.net_names:
            group_layers[net_name] = []
            groups = self.params[net_name]['groups']
            merge = self.params[net_name]['merge']
            if isinstance(x[net_name], list):  # These data is for cnn method because must have data_row
                X_slice[net_name] = dict()
                data, data_row = x[net_name]
                if len(groups) == 0 and isinstance(data, np.ndarray):
                    group_layers[net_name] += ['input']
                    X_slice[net_name]['input'] = data[indices].squeeze()
                elif isinstance(data, dict) and len(groups) == 0:
                    for key in data.keys():
                        X_slice[net_name][key] = x[net_name][key][indices].squeeze()
                elif isinstance(data, dict) and len(groups) > 0:
                    if split_primary_key:
                        for key in data.keys():
                            group_layers[net_name] += merge.split('_')[1:]
                            for group in groups:
                                group_name = '_'.join(group) if isinstance(group, tuple) else group
                                group_name = '_'.join([key, group_name])
                                X_slice[net_name][group_name] = data[key][group_name][indices].squeeze()
                    else:
                        group_layers[net_name] += merge.split('_')[1:]
                        for group in groups:
                            group_name = '_'.join(group) if isinstance(group, tuple) else group
                            X_slice[net_name][group_name] = data[group_name][indices].squeeze()
                X_slice[net_name]['data_row'] = data_row.iloc[indices].values
            elif isinstance(x[net_name], dict) and len(groups) > 0:  # These data is for lstm and mlp method
                X_slice[net_name] = dict()
                if split_primary_key:
                    for key in x[net_name].keys():
                        group_layers[net_name] += merge.split('_')[1:]
                        for group in groups:
                            group_name = '_'.join(group) if isinstance(group, tuple) else group
                            group_name = '_'.join([key, group_name])
                            X_slice[net_name][group_name] = x[net_name][key][group_name].iloc[indices].values
                else:
                    group_layers[net_name] += merge.split('_')[1:]
                    for group in groups:
                        group_name = '_'.join(group) if isinstance(group, tuple) else group
                        X_slice[net_name][group_name] = x[net_name][group_name].iloc[indices].values
            elif isinstance(x[net_name], dict) and len(groups) == 0:  # These data is for lstm and mlp method
                X_slice[net_name] = dict()
                for key in x[net_name].keys():
                    X_slice[net_name][key] = x[net_name][key].iloc[indices].values
            elif isinstance(x[net_name], pd.DataFrame):  # These data is for mlp method
                X_slice[net_name] = x[net_name].iloc[indices].values
            elif isinstance(x, np.ndarray):
                X_slice[net_name] = x[net_name][indices]  # These data is for lstm method
            else:
                raise ValueError('Wrong type of input X')
        if act is None:
            return X_slice, y_slice, group_layers, mask
        else:
            X_slice_rules = dict()
            for net_name in self.net_names:
                if rules[net_name] is not None:
                    X_slice_rules[net_name] = dict()
                    group_layers[net_name] = ['clustering'] + group_layers[net_name]
                    if act is not None:
                        if net_name in act.keys():
                            for rule in act[net_name].columns:
                                X_slice_rules[net_name][f'act_{rule}'] = (
                                    act[net_name][rule].loc[mask].values.reshape(-1, 1))
                    else:
                        raise ValueError('If you provide rules, you should also provide data for clustering x_imp '
                                         'or activations')
                    for rule in rules[net_name]:
                        if self.params[net_name]['replicate_data']:
                            if isinstance(X_slice[net_name], dict):
                                if 'data_row' in X_slice[net_name].keys():
                                    X_slice_rules[net_name]['data_row'] = X_slice[net_name]['data_row']
                                for key, value in X_slice[net_name].items():
                                    if key != 'data_row':
                                        X_slice_rules[net_name]['_'.join([rule, key])] = value
                            else:
                                X_slice_rules[net_name][rule] = X_slice[net_name]
                        else:
                            if not isinstance(X_slice[net_name], dict):
                                raise ValueError('Data in X_slice should be dictionary')
                            if len(X_slice[net_name]) != act[net_name].shape[1]:
                                raise ValueError('The length of X_slice[net_name] should be '
                                                 'equal with the columns of activation')
                            for key, value in X_slice[net_name].items():
                                X_slice_rules[net_name][key] = value
                else:
                    X_slice_rules[net_name] = X_slice[net_name]
            return X_slice_rules, y_slice, group_layers, mask

    def fit(self, X, y, cv_masks, meta_data, activations=None, gpu_id=0):
        y = remove_zeros_load_ts(self.static_data, y)
        if self.is_trained and not self.refit:
            return self.best_mae_test
        quantiles = self.params['quantiles'] if self.probabilistic else None
        self.params['n_out'] = y.shape[1]
        for net_name in self.net_names:
            params = self.params[net_name]
            params['n_out'] = y.shape[1]
            params['experiment'] = enhance_model_layers_distributed_data(params['what_data'], params['experiment'])
            params['experiment'] = fix_convolutional_names(params['experiment_tag'], params['experiment'],
                                                           params['conv_dim'])
            params['experiment'] = enhance_model_layers_multi_output(self.static_data, params['experiment_tag'],
                                                                     params['experiment'], self.probabilistic)
            if params['is_global']:
                if activations is None:
                    raise ValueError('Provide activations or turn is_fuzzy attribute to True')
                try:
                    params['rules'] = activations[net_name].columns
                except Exception as e:
                    print(e)
                    raise ValueError('activations should be dictionary with net_names as keys')
            else:
                params['rules'] = None

            self.params[net_name] = params
        self.rules = {net_name: self.params[net_name]['rules'] for net_name in self.net_names}
        thres_act = 0.001
        X_train, y_train, group_layers, mask = self.get_slice(X, cv_masks[0], meta_data, y=y,
                                                              act=activations,
                                                              rules=self.rules)
        X_val, y_val, _, _ = self.get_slice(X, cv_masks[1], meta_data, y=y, act=activations,
                                            rules=self.rules)
        X_test, y_test, _, _ = self.get_slice(X, cv_masks[2], meta_data, y=y, act=activations,
                                              rules=self.rules)
        for net_name in self.net_names:
            if isinstance(X_train[net_name], dict):
                self.params[net_name]['scopes'] = [scope for scope in X_train[net_name].keys() if 'act' not in scope]
            else:
                self.params[net_name]['scopes'] = ['input']

            self.params[net_name]['group_layers'] = group_layers[net_name]
        with open(os.path.join(self.path_weights, 'parameters.txt'), 'w') as file:
            file.write(yaml.dump(self.params, default_flow_style=False, sort_keys=False))
        self.n_out = y_train.shape[1]
        N = cv_masks[0].intersection(mask).shape[0]
        self.batch_size = np.minimum(self.batch_size, int(N / 7.5))
        self.n_batch = int(N / self.batch_size)
        batches = [np.random.choice(N, self.batch_size, replace=False) for _ in range(self.n_batch + 1)]

        config = get_tf_config(self.static_data['n_jobs'])

        tf.compat.v1.reset_default_graph()
        graph_cnn = tf.Graph()
        print('Create graph....')
        with graph_cnn.as_default():
            with tf.device('/device:GPU:' + str(gpu_id)):
                y_pl = tf.compat.v1.placeholder(tf.float32, shape=[None, self.params['n_out']], name='target')
                x_pl = dict()
                model_output_list = []
                model_layers_built = dict()
                act_pl = dict()
                act_nan_err = []
                for net_name in self.net_names:
                    model_layers = self.params[net_name]['experiment']
                    is_fuzzy = self.params[net_name]['is_fuzzy']
                    is_global = self.params[net_name]['is_global']
                    is_for_cluster = self.params[net_name]['is_for_cluster']
                    with tf.name_scope(net_name) as scope:
                        x_pl[net_name] = create_placeholders_for_concat_nets(X_train[net_name], self.params[net_name])
                        model_output_temp, model_layers_built_temp, \
                            act_pl_temp, act_nan_err_temp = build_graph_for_concat_nets(x_pl[net_name],
                                                                                        model_layers,
                                                                                        self.params[net_name],
                                                                                        is_fuzzy=
                                                                                        is_fuzzy,
                                                                                        is_global=
                                                                                        is_global,
                                                                                        is_for_cluster=
                                                                                        is_for_cluster,
                                                                                        thres_act=thres_act)
                    model_output_list.append(model_output_temp)
                    model_layers_built[net_name] = model_layers_built_temp
                    act_pl[net_name] = act_pl_temp
                    if act_nan_err_temp is not None:
                        act_nan_err.append(act_nan_err_temp)
                model_output, model_layers_built = build_output_for_concat_nets(model_output_list, model_layers_built,
                                                                                self.params[self.net_names[-1]],
                                                                                self.probabilistic,
                                                                                quantiles,
                                                                                train=True)
                act_nan_err = tf.add_n(act_nan_err)
                trainers, MAEs, SSEs, learning_rate = optimize(model_output, y_pl,
                                                               act_nan_err=act_nan_err,
                                                               rated=self.rated,
                                                               is_global=is_global,
                                                               is_fuzzy=is_fuzzy,
                                                               probabilistic=self.probabilistic,
                                                               quantiles=quantiles)
        len_performers = 2
        mae_old, sse_old = np.inf * np.ones(len_performers), np.inf * np.ones(len_performers)
        mae_max, sse_max = np.inf * np.ones(len_performers), np.inf * np.ones(len_performers)
        mae_min, sse_min = np.inf * np.ones(len_performers), np.inf * np.ones(len_performers)

        best_mae_val, best_mae_test = np.inf, np.inf
        train_flag, best_weights, best_clusters, init_clusters = True, None, None, None
        warm = self.params['warming_iterations']
        wait, best_iteration, best_tot_iteration, loops, n_iter = 0, 0, 0, 0, 0
        epochs = self.epochs
        patience, exam_period = int(self.epochs / 3), int(self.epochs / 5)

        results_columns = ['Iteration', 'best_iteration', 'best_mae_val', 'best_mae_test', 'mae_val_out',
                           'mae_test_out', 'sse_val_out', 'sse_test_out']
        results = pd.DataFrame(columns=results_columns)

        print(f"Start training of {self.name} using {self.method} with {self.n_batch} batches and {self.epochs} epochs")

        with tf.compat.v1.Session(graph=graph_cnn, config=config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            variables = [v for v in tf.compat.v1.trainable_variables()]
            while train_flag:
                for epoch in tqdm(range(epochs)):
                    lr = lr_schedule(epoch, lr=self.learning_rate)
                    random.shuffle(batches)
                    train_step(sess, trainers['bulk'], batches, x_pl, y_pl, X_train, y_train, learning_rate, lr)
                    warm = 0
                    sse_val = validation_step(sess, SSEs, np.arange(y_val.shape[0]), x_pl, y_pl, X_val,
                                              y_val, learning_rate, lr)
                    mae_val = validation_step(sess, MAEs, np.arange(y_val.shape[0]), x_pl, y_pl, X_val,
                                              y_val, learning_rate, lr)
                    sse_test = validation_step(sess, SSEs, np.arange(y_test.shape[0]), x_pl, y_pl, X_test,
                                               y_test, learning_rate, lr)
                    mae_test = validation_step(sess, MAEs, np.arange(y_test.shape[0]), x_pl, y_pl, X_test,
                                               y_test, learning_rate, lr)
                    mae = np.hstack([mae_val[-1], mae_test[-1]])
                    flag_mae, mae_old, mae_max, mae_min = distance(mae, mae_old, mae_max, mae_min)
                    sse = np.hstack([sse_val[-1], sse_test[-1]])
                    flag_sse, sse_old, sse_max, sse_min = distance(sse / 10, sse_old, sse_max, sse_min)
                    flag_best = flag_mae and flag_sse
                    if flag_best:
                        best_weights = gather_weights(sess, variables)
                        best_tot_iteration = n_iter
                        best_iteration = epoch
                        wait = 0
                    else:
                        wait += 1
                    best_mae_val = mae_val[-1] if best_mae_val >= mae_val[-1] else best_mae_val
                    best_mae_test = mae_test[-1] if best_mae_test >= mae_test[-1] else best_mae_test
                    evaluation = np.array([n_iter, best_tot_iteration, best_mae_val, best_mae_test,
                                           mae_val[-1], mae_test[-1], sse_val[-1], sse_test[-1]])

                    print_columns = ['best_mae_val', 'best_mae_test', 'mae_val_out', 'mae_test_out']

                    res = pd.DataFrame(evaluation.reshape(-1, 1).T, index=[n_iter], columns=results_columns)
                    results = pd.concat([results, res])
                    n_iter += 1
                    print(res[print_columns])
                    if wait > patience and epoch > 400:
                        train_flag = False
                        break
                if (epochs - best_iteration) <= exam_period and epoch > 400:
                    if loops > 3:
                        train_flag = False
                    else:
                        epochs, exam_period = patience, int(patience / 3)
                        best_iteration = 0
                        loops += 1
                else:
                    train_flag = False
            sess.close()
        lstm_flag = any(['lstm' in weight for weight in best_weights])

        if lstm_flag:
            lstm_weights = dict()
            for name, weight in best_weights.items():
                if 'lstm' in name:
                    layer_name = '_'.join(name.split('/')[:-1])
                    if layer_name not in lstm_weights.keys():
                        lstm_weights[layer_name] = dict()
                    weight_name = name.split('/')[-1].split(':')[0]
                    lstm_weights[layer_name][weight_name] = weight
                    lstm_weights[layer_name][f'{weight_name}_name'] = name
            for layer_name, layer_weights in lstm_weights.items():
                weights = [layer_weights['kernel'], layer_weights['recurrent_kernel'], layer_weights['bias']]
                weights = convert_lstm_weights(weights)
                kernel_name = layer_weights['kernel_name']
                recurrent_kernel_name = layer_weights['recurrent_kernel_name']
                bias_name = layer_weights['bias_name']
                best_weights[kernel_name] = weights[0]
                best_weights[recurrent_kernel_name] = weights[1]
                best_weights[bias_name] = weights[2]

        self.best_weights = best_weights
        self.best_mae_test = results['mae_test_out'].iloc[best_tot_iteration]
        self.best_mae_val = results['mae_val_out'].iloc[best_tot_iteration]
        self.best_sse_test = results['sse_test_out'].iloc[best_tot_iteration]
        self.best_sse_val = results['sse_val_out'].iloc[best_tot_iteration]
        self.results = results.iloc[best_tot_iteration]
        results.to_csv(os.path.join(self.path_weights, 'results.csv'))
        self.is_trained = True
        self.save()
        gc.collect()
        print(f"Total accuracy of validation: {self.best_mae_val} and of testing {self.best_mae_test}")

    def predict(self, X, metadata, cluster_dates=None, activations=None):
        self.load()
        quantiles = self.params['quantiles'] if self.probabilistic else None
        thres_act = 0.001
        if not hasattr(self, 'best_weights'):
            raise NotImplementedError(f'The {self.method} network is not train. '
                                      f'The location path is {self.path_weights}')
        cluster_dates = metadata['dates'] if cluster_dates is None else cluster_dates.intersection(metadata['dates'])
        inp_x, _, _, cluster_dates = self.get_slice(X, cluster_dates, metadata, act=activations,
                                                    rules=self.rules)
        tf.compat.v1.reset_default_graph()
        graph_net = tf.Graph()
        print('Create graph....')
        with graph_net.as_default():
            with tf.device('/CPU:0'):
                x_pl = dict()
                model_output_list = []
                model_layers_built = dict()
                act_pl = dict()
                act_nan_err = []
                for net_name in self.net_names:
                    model_layers = self.params[net_name]['experiment']
                    is_fuzzy = self.params[net_name]['is_fuzzy']
                    is_global = self.params[net_name]['is_global']
                    is_for_cluster = self.params[net_name]['is_for_cluster']
                    with tf.name_scope(net_name) as scope:
                        x_pl[net_name] = create_placeholders_for_concat_nets(inp_x[net_name], self.params[net_name])
                        model_output_temp, model_layers_built_temp, \
                            act_pl_temp, act_nan_err_temp = build_graph_for_concat_nets(x_pl[net_name],
                                                                                        model_layers,
                                                                                        self.params[net_name],
                                                                                        is_fuzzy=
                                                                                        is_fuzzy,
                                                                                        is_global=
                                                                                        is_global,
                                                                                        is_for_cluster=
                                                                                        is_for_cluster,
                                                                                        thres_act=thres_act)
                    model_output_list.append(model_output_temp)
                    model_layers_built[net_name] = model_layers_built_temp
                    act_pl[net_name] = act_pl_temp
                    if act_nan_err_temp is not None:
                        act_nan_err.append(act_nan_err_temp)
                model_output, model_layers_built = build_output_for_concat_nets(model_output_list, model_layers_built,
                                                                                self.params[self.net_names[-1]],
                                                                                self.probabilistic,
                                                                                quantiles,
                                                                                train=True)

        config = get_tf_config(self.static_data['n_jobs'])
        with tf.compat.v1.Session(graph=graph_net, config=config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            for variable in tf.compat.v1.trainable_variables():
                sess.run(tf.compat.v1.assign(variable, self.best_weights[variable.name]))
            feed_dict = feed_data_eval(x_pl, inp_x)
            y_pred = sess.run([model_output], feed_dict=feed_dict)
            sess.close()
        if self.static_data['horizon_type'] == 'multi-output':
            cols = [f'hour_ahead_{h}' for h in range(self.static_data['horizon'])]
        else:
            cols = [self.method]
        if self.probabilistic:
            return y_pred[0]
        else:
            y_pred = pd.DataFrame(y_pred[0], index=cluster_dates, columns=cols)
            return y_pred

    def load(self):
        if os.path.exists(os.path.join(self.path_weights, 'net_weights.pickle')):
            try:
                tmp_dict = joblib.load(os.path.join(self.path_weights, 'net_weights.pickle'))
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot load weights for cnn model' + self.path_weights)
        else:
            raise ImportError('Cannot load weights for cnn model' + self.path_weights)

    def save(self):
        tmp_dict = {}
        for k in self.__dict__.keys():
            if k not in ['static_data', 'path_weights', 'refit']:
                tmp_dict[k] = self.__dict__[k]
        joblib.dump(tmp_dict, os.path.join(self.path_weights, 'net_weights.pickle'), compress=9)
