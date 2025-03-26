import os
import sys

path_pycharm = (os.path.normpath(os.path.dirname(__file__))).split(os.sep)
path_pycharm = os.path.join(*path_pycharm[:-2])
if sys.platform == 'linux':
    path_pycharm = '/' + path_pycharm
print(path_pycharm)
sys.path.append(path_pycharm)
import gc
import time
import shutil
import joblib
import traceback

import numpy as np
import pandas as pd

from GPUtil import getGPUs
import multiprocessing as mp

from eforecast.optimizers.optimizer_hyper import HyperoptOptimizer

from eforecast.common_utils.dataset_utils import sync_datasets
from eforecast.common_utils.dataset_utils import sync_target_with_tensors
from eforecast.common_utils.train_utils import send_predictions

from eforecast.dataset_creation.data_feeder import DataFeeder
from eforecast.deep_models.tf_1x.network import DeepNetwork

trial_number = int(sys.argv[1])
method = sys.argv[2]
cluster_name = sys.argv[3]
path_cluster = sys.argv[4]
gpu_id = int(sys.argv[5])
refit = bool(sys.argv[6])


# trial_number = 2
# method = 'LSTM'
# cluster_name = 'RBF_rule_0'
# path_cluster = '/media/smartrue/HHD1/George/models/my_projects/IPTO_ver6_ver0/load/lv_load/day-ahead/model_ver0/cluster_organizer/RBF/rule_0'
# gpu_id = 1
# refit = bool(0)


class Process(mp.Process):
    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    def run(self):
        try:
            mp.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
            print("".join(tb))
            send_predictions(" ".join(tb))
            self._cconn.send(-1)

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


class Objective():
    def __init__(self, static_data, cluster_name, cluster_dir, method, refit=False):
        self.space_structure = None
        self.static_data = static_data
        self.method = method
        self.cluster_dir = cluster_dir
        self.cluster_name = cluster_name
        self.refit = refit
        self.warming = self.static_data[method]['warming_iterations']

        self.nwp_data_merge = self.static_data['nwp_data_merge']
        self.nwp_data_compress = self.static_data['compress_data']
        self.scale_nwp_methods = self.static_data['scale_nwp_method']
        self.data_structure = self.static_data['data_structure']
        self.feature_selection_methods = self.static_data['feature_selection_methods']
        self.data_feeder = DataFeeder(self.static_data, online=False, train=True)
        self.initialize(refit)

    def initialize(self, refit=False):
        self.define_space()
        self.define_structure_space()

    def load_data(self, merge, compress, scale_method, what_data, feature_selection_method=None):
        X, metadata = self.data_feeder.feed_inputs(merge=merge, compress=compress,
                                                   scale_nwp_method=scale_method,
                                                   what_data=what_data,
                                                   feature_selection_method=feature_selection_method,
                                                   cluster={'cluster_name': self.cluster_name,
                                                            'cluster_path': self.cluster_dir})
        y, _ = self.data_feeder.feed_target(inverse=False)
        if isinstance(X, pd.DataFrame):
            X, y = sync_datasets(X, y, name1='inputs', name2='target')
        elif isinstance(X, list):
            data_row = X[1]
            data = X[0]
            X, y = sync_target_with_tensors(target=y, data_tensor=data, dates_tensor=metadata['dates'],
                                            data_row=data_row)
        elif isinstance(X, dict):
            X, y = sync_target_with_tensors(target=y, data_tensor=X, dates_tensor=metadata['dates'])
        else:
            X, y = sync_target_with_tensors(target=y, data_tensor=X, dates_tensor=metadata['dates'])
        metadata['dates'] = y.index
        return X, y, metadata

    def get_param(self, name, type, dtype, space=None, range=None, values=None):
        return {name: {'type': type,
                       'dtype': dtype,
                       'space': space,
                       'range': range,
                       'values': values}}

    def select_structure(self, trial_structure, experiment_tag, exp):
        exp_sel = dict()
        for key, layers in exp.items():
            exp_sel[key] = []
            for i, layer in enumerate(layers):
                layer_type = layer[0]
                param = f'{experiment_tag}_{key}_{layer_type}_{i}'
                size = trial_structure[param] if param in trial_structure.keys() \
                    else self.fix_params_structure[experiment_tag][param]
                if isinstance(size, str):
                    if size not in 'linear':
                        size = float(size)
                elif isinstance(size, int):
                    size = float(size)
                exp_sel[key].append((layer_type, size))
        return exp_sel

    def define_structure_space(self):
        experiment_tags = list(self.static_data[self.method]['experiment_tag'])
        self.space_structure = dict()
        self.fix_params_structure = dict()
        self.param_layer_names = dict()
        for experiment_tag in experiment_tags:
            exp = self.static_data['experiments'][experiment_tag]
            self.space_structure[experiment_tag] = dict()
            self.fix_params_structure[experiment_tag] = dict()
            self.param_layer_names[experiment_tag] = []
            for key, layers in exp.items():
                for i, layer in enumerate(layers):
                    layer_type = layer[0]
                    sizes = layer[1]
                    param = f'{experiment_tag}_{key}_{layer_type}_{i}'
                    if isinstance(sizes, list):
                        if len(sizes) > 1:
                            self.space_structure[experiment_tag].update(
                                self.get_param(param, 'real', 'float', range=list(sizes)))
                        elif len(sizes) == 1:
                            self.fix_params_structure[experiment_tag][param] = list(sizes)[0]
                        else:
                            self.fix_params_structure[experiment_tag][param] = sizes
                    elif not isinstance(sizes, list) and not isinstance(sizes, set):
                        self.fix_params_structure[experiment_tag][param] = sizes
                    else:
                        if len(sizes) > 1:
                            self.space_structure[experiment_tag].update(
                                self.get_param(param, 'cat', 'float', values=list(sizes)))
                        elif len(sizes) == 1:
                            self.fix_params_structure[experiment_tag][param] = sizes[0]
                        else:
                            self.fix_params_structure[experiment_tag][param] = sizes
        self.param_structure_names = dict()
        for experiment_tag, experiment in self.space_structure.items():
            self.param_structure_names[experiment_tag] = []
            for param_name, param_attr in experiment.items():
                self.param_structure_names[experiment_tag].append(param_name)

    def define_space(self):
        self.space = dict()
        self.fix_params = dict()
        if len(self.nwp_data_merge) > 1 and self.method not in {'LSTM'}:
            self.space.update(self.get_param('merge', 'cat', 'string', values=self.nwp_data_merge))
        else:
            self.fix_params['merge'] = list(self.nwp_data_merge)[0]
        if len(self.nwp_data_compress) > 1 and self.method not in {'LSTM', 'CNN'}:
            self.space.update(self.get_param('compress', 'cat', 'string', values=self.nwp_data_compress))
        elif self.method in {'LSTM'}:
            self.fix_params['compress'] = 'load'
        elif self.method in {'CNN'}:
            self.fix_params['compress'] = None
        else:
            self.fix_params['compress'] = list(self.nwp_data_compress)[0]
        if len(self.feature_selection_methods) > 1 and self.method not in {'CNN'}:
            self.space.update(
                self.get_param('feature_selection_method', 'cat', 'string', values=self.feature_selection_methods))
        elif self.method in {'CNN'}:
            self.fix_params['feature_selection_method'] = None
        else:
            self.fix_params['feature_selection_method'] = self.feature_selection_methods[0]

        data_structure = [data_struct for data_struct in self.data_structure
                          if data_struct in {'row', 'row_all',
                                             'row_dict',
                                             'row_dict_distributed'}]
        if len(data_structure) == 0 and self.method not in {'LSTM', 'CNN'}:
            raise ValueError('Cannot find what_data structure to fit')
        elif len(data_structure) > 1 and self.method not in {'LSTM', 'CNN'}:
            self.space.update(self.get_param('what_data', 'cat', 'string', values=data_structure))
        elif self.method in {'LSTM', 'CNN'}:
            self.fix_params['what_data'] = str.lower(self.method)
        else:
            self.fix_params['what_data'] = data_structure[0]

        if len(self.scale_nwp_methods) > 1:
            self.space.update(self.get_param('scale', 'cat', 'string', values=self.scale_nwp_methods))
        else:
            self.fix_params['scale'] = list(self.scale_nwp_methods)[0]

        for param, value in self.static_data[self.method].items():
            if isinstance(value, set):
                if len(value) > 1:
                    if isinstance(list(value)[0], str):
                        self.space.update(self.get_param(param, 'cat', 'string', values=list(value)))
                    elif isinstance(list(value)[0], int):
                        self.space.update(self.get_param(param, 'cat', 'int', values=list(value)))
                    else:
                        self.space.update(self.get_param(param, 'cat', 'float', values=list(value)))
                else:
                    self.fix_params[param] = list(value)[0]
            elif isinstance(value, list):
                if len(value) > 1:
                    if isinstance(value[0], int):
                        self.space.update(self.get_param(param, 'int', 'int', range=value))
                    else:
                        self.space.update(self.get_param(param, 'real', 'float', range=value))
                else:
                    self.fix_params[param] = value[0]
            else:
                self.fix_params[param] = value
        self.param_names = []
        for param_name, param_attr in self.space.items():
            self.param_names.append(param_name)

    @staticmethod
    def _fit(model, X, y, cv_mask, metadata, gpu_i):
        model.fit(X, y, cv_mask, metadata, gpu_id=gpu_i)

    def fit_trial(self, trial_number, trials, gpu_i):
        print(trials)
        optimizer = HyperoptOptimizer(self.space)
        if len(trials) > 0:
            y_trial = []
            X_trial = []
            for trial in trials:
                param_dict = dict()
                for key in self.param_names:
                    param_dict[key] = trial[key]
                X_trial.append(param_dict)
                y_trial.append(trial['value'])
            X_trial = pd.DataFrame(X_trial)
            optimizer.observe(X_trial, np.array(y_trial))
        trial = optimizer.suggest(n_suggestions=1)[0]
        merge = trial['merge'] if 'merge' in trial.keys() else self.fix_params['merge']
        compress = trial['compress'] if 'compress' in trial.keys() else self.fix_params['compress']
        what_data = trial['what_data'] if 'what_data' in trial.keys() else self.fix_params['what_data']
        scale = trial['scale'] if 'scale' in trial.keys() else self.fix_params['scale']
        feature_selection_method = trial['feature_selection_method'] if 'feature_selection_method' in trial.keys() \
            else self.fix_params['feature_selection_method']
        X, y, metadata = self.load_data(merge, compress, scale, what_data,
                                        feature_selection_method=feature_selection_method)

        experiment_tag = trial['experiment_tag'] if 'experiment_tag' in trial.keys() \
            else self.fix_params['experiment_tag']

        cv_masks = joblib.load(os.path.join(self.cluster_dir, 'cv_mask.pickle'))

        experiment_params = {'trial_number': trial_number,
                             'method': self.method,
                             'name': self.cluster_name,
                             'merge': merge,
                             'compress': compress,
                             'what_data': what_data,
                             'feature_selection_method': feature_selection_method,
                             'scale_nwp_method': scale,
                             'groups': metadata['groups'],
                             'experiment_tag': experiment_tag}
        for param, value in trial.items():
            if param not in experiment_params.keys():
                experiment_params[param] = value
        experiment_params.update(self.fix_params)

        optimizer_structure = HyperoptOptimizer(self.space_structure[experiment_tag])
        if len(trials) > 0:
            y_trial_structure = []
            indices = []
            X_trial_structure = []
            for i, trial in enumerate(trials):
                param_dict = dict()
                for key in self.param_structure_names[experiment_tag]:
                    if key in trial.keys():
                        param_dict[key] = trial[key]
                if len(param_dict) > 0:
                    indices.append(i)
                    X_trial_structure.append(param_dict)
                    y_trial_structure.append(trial['value'])
            if len(X_trial_structure) > 0:
                X_trial_structure = pd.DataFrame(X_trial_structure)
                optimizer_structure.observe(X_trial_structure, np.array(y_trial_structure))
        trial_structure = optimizer_structure.suggest(n_suggestions=1)[0]

        experiment_params['experiment'] = self.select_structure(trial_structure, experiment_tag,
                                                                self.static_data['experiments'][experiment_tag])

        path_weights = os.path.join(self.cluster_dir, self.method, f'test_{trial_number}')
        if os.path.exists(path_weights) and self.refit:
            shutil.rmtree(path_weights)
        if not os.path.exists(path_weights):
            os.makedirs(path_weights)
        acc = np.inf
        model = DeepNetwork(self.static_data, path_weights, experiment_params, refit=self.refit)
        if model.is_trained:
            acc = model.best_mae_test
            for param in model.params.keys():
                if param in trial.keys():
                    trial[param] = model.params[param]
        else:
            while True:
                gpus = getGPUs()
                gpuUtil = gpus[gpu_i].load
                if gpuUtil < 0.9:
                    break
                else:
                    time.sleep(10)

            while True:
                p = Process(target=self._fit, args=(model, X, y, cv_masks, metadata, gpu_i))
                p.start()
                p.join()
                gpus = getGPUs()
                memory_util = gpus[gpu_i].memoryUtil
                if p.exception or not os.path.exists(os.path.join(path_weights, 'net_weights.pickle')):
                    if memory_util < 0.12:
                        print('Trial aboard due to gpu memory utilization')
                        print()
                        model.best_weights = {}
                        model.best_mae_test = np.inf
                        model.best_mae_val = np.inf
                        model.results = pd.DataFrame()
                        model.is_trained = True
                        model.save()
                        raise RuntimeError('deep model failed')

                    print('Network do not fit to GPU memory')
                    time.sleep(30)
                    continue
                else:

                    break
            model.load()
            acc = model.best_mae_test
        print(acc)
        experiment_params['value'] = acc
        experiment_params['mae_test'] = model.best_mae_test
        experiment_params['mae_val'] = model.best_mae_val
        experiment_params['sse_test'] = model.best_sse_test
        experiment_params['sse_val'] = model.best_sse_val

        columns = ['trial_number', 'value', 'mae_test', 'mae_val', 'sse_val', 'sse_test'] + self.param_names
        trial = {key: value for key, value in experiment_params.items() if key in columns}
        trial.update(trial_structure)
        path_trials = os.path.join(self.cluster_dir, self.method, 'trials')
        file_trial = f'trial{trial_number}.pickle'
        joblib.dump(trial, os.path.join(path_trials, file_trial))
        del model
        gc.collect()


def get_results(static_data, cluster_dir, trial, method):
    path_weights = os.path.join(cluster_dir, method, f'test_{trial}')
    model = DeepNetwork(static_data, path_weights)

    return model.results


if __name__ == '__main__':
    static_data = joblib.load(os.path.join(path_cluster, 'static_data.pickle'))
    path_trials = os.path.join(path_cluster, method, 'trials')
    if not os.path.exists(path_trials):
        os.makedirs(path_trials)
    trials = []
    for trial in sorted(os.listdir(path_trials)):
        trials.append(joblib.load(os.path.join(path_trials, trial)))
    objective = Objective(static_data, cluster_name, path_cluster, method, refit=refit)
    try:
        objective.fit_trial(trial_number, trials, gpu_id)
        sys.exit(0)
    except:
        sys.exit(-1)
