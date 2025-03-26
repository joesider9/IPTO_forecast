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
import traceback
import joblib

import numpy as np
import pandas as pd

from GPUtil import getGPUs

from eforecast.optimizers.optimizer_hyper import HyperoptOptimizer

from eforecast.common_utils.dataset_utils import sync_datasets
from eforecast.common_utils.dataset_utils import sync_target_with_tensors
from eforecast.common_utils.train_utils import send_predictions

from eforecast.dataset_creation.data_feeder import DataFeeder
from eforecast.dataset_creation.files_manager import FilesManager

from eforecast.deep_models.tf_1x.global_network import DistributedDeepNetwork

import multiprocessing as mp

trial_number = int(sys.argv[1])
path_project = sys.argv[2]
gpu_id = int(sys.argv[3])
refit = bool(sys.argv[4])
# trial_number = 0
# path_project = '/media/smartrue/HHD1/George/models/EMS_forecast/Kythnos_ver0/pv/kythnos_ems/multi-output/model_ver0/Distributed'
# gpu_id = 0
# refit = bool(1)
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


class Objective(object):
    def __init__(self, static_data , refit=False):
        self.space_structure = None
        self.static_data = static_data
        self.param_space = self.static_data['Global']
        self.method = 'Global'
        self.refit = refit
        self.warming = self.static_data['Global']['warming_iterations']
        self.nwp_data_merge = self.static_data['Global']['nwp_data_merge']
        self.nwp_data_compress = self.static_data['Global']['compress_data']
        self.scale_nwp_methods = self.static_data['Global']['scale_nwp_method']
        self.what_data = self.static_data['Global']['what_data']
        self.feature_selection_methods = self.static_data['feature_selection_methods']
        self.data_feeder = DataFeeder(self.static_data, online=False, train=True)
        self.file_manager = FilesManager(self.static_data, is_online=False, train=True)
        self.initialize(refit)

    def initialize(self, refit=False):
        self.define_space()
        self.define_structure_space()

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

    def load_data(self, merge, compress, scale_method, what_data, feature_selection_method=None):
        X, metadata = self.data_feeder.feed_inputs(merge=merge, compress=compress,
                                                   scale_nwp_method=scale_method,
                                                   what_data=what_data,
                                                   feature_selection_method=feature_selection_method)
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

    def define_space(self):
        self.space = dict()
        self.fix_params = dict()
        for param, value in self.static_data[self.method].items():
            if param in {'rbf_var_imp'}:
                self.fix_params[param] = value
                continue
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
        experiment_tag = trial['experiment_tag'] if 'experiment_tag' in trial.keys() \
            else self.fix_params['experiment_tag']

        merge = self.nwp_data_merge

        conv_dim = None
        if 'mlp' in experiment_tag:
            if self.static_data['type'] in {'load', 'fa'}:
                compress = 'load'
            else:
                compress = self.nwp_data_compress
            feature_selection_method = None
            what_data = self.what_data
        else:
            if 'cnn' in experiment_tag:
                feature_selection_method = None
                conv_dim = self.param_space['conv_dim']
                what_data = 'cnn'
            elif 'lstm' in experiment_tag:
                feature_selection_method = None
                what_data = 'lstm'
            else:
                raise ValueError(f'Unknown method {experiment_tag} in experiment_tag')
            compress = None

        scale = self.scale_nwp_methods

        X, y, metadata = self.load_data(merge, compress, scale, what_data,
                                        feature_selection_method=feature_selection_method)

        cv_mask = self.file_manager.check_if_exists_cv_data()
        name = f'Distributed_{trial_number}'
        experiment_params = {'trial_number': trial_number,
                             'name': name,
                             'experiment_tag': experiment_tag,
                             'merge': merge,
                             'compress': compress,
                             'what_data': what_data,
                             'conv_dim': conv_dim,
                             'feature_selection_method': feature_selection_method,
                             'scale_nwp_method': scale,
                             'is_fuzzy': trial['is_fuzzy'] if 'is_fuzzy' in trial.keys() else self.fix_params['is_fuzzy'],
                             'n_rules': trial['n_rules'] if 'n_rules' in trial.keys() else self.fix_params['n_rules'],
                             'clustering_method': trial['clustering_method'],
                             'rbf_var_imp': self.param_space['rbf_var_imp'],
                             'data_type': self.param_space['data_type']}
        for param, value in trial.items():
            if param not in experiment_params.keys():
                experiment_params[param] = value
        for param, value in self.fix_params.items():
            if param not in experiment_params.keys():
                experiment_params[param] = value
        if len(self.space_structure[experiment_tag]) > 0:
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
        else:
            trial_structure = dict()
            experiment_params['experiment'] = self.static_data['experiments'][experiment_tag]

        path_weights = os.path.join(self.static_data['path_model'], 'Distributed', f'Distributed_{trial_number}')
        if not os.path.exists(path_weights):
            os.makedirs(path_weights)
        model = DistributedDeepNetwork(self.static_data, path_weights, is_online=False, train=True,
                                       params=experiment_params, refit=self.refit)
        if model.is_trained:
            acc = model.best_mae_test
            for param in model.params.keys():
                if param in trial.keys():
                    trial[param] = model.params[param]
        else:
            while True:
                gpus = getGPUs()
                print(gpus)
                gpuUtil = gpus[gpu_i].load
                if gpuUtil < 0.9:
                    break
                else:
                    time.sleep(10)

            while True:
                p = Process(target=self._fit, args=(model, X, y, cv_mask, metadata, gpu_i))
                p.start()
                p.join()
                gpus = getGPUs()
                memory_util = gpus[gpu_i].memoryUtil
                if p.exception or not os.path.exists(os.path.join(path_weights, 'distributed_model.pickle')):
                    if memory_util < 0.12:
                        print('Trial aboard due to gpu memory utilization')
                        print()
                        model.best_weights = {}
                        model.best_mae_test = np.inf
                        model.best_mae_val = np.inf
                        model.results = pd.DataFrame()
                        model.is_trained = True
                        model.save()
                        return np.inf

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
        path_trials = os.path.join(self.static_data['path_model'], 'Distributed', 'trials')
        file_trial = f'trial{trial_number}.pickle'
        joblib.dump(trial, os.path.join(path_trials, file_trial))
        del model
        gc.collect()


def get_results(static_data, cluster_dir, trial, method):
    path_weights = os.path.join(cluster_dir, method, f'test_{trial}')
    model = DistributedDeepNetwork(static_data, path_weights)

    return model.results

if __name__ == '__main__':

    static_data = joblib.load(os.path.join(path_project, 'static_data.pickle'))
    path_trials = os.path.join(path_project, 'trials')
    if not os.path.exists(path_trials):
        os.makedirs(path_trials)
    trials = []
    for trial in sorted(os.listdir(path_trials)):
        trials.append(joblib.load(os.path.join(path_trials, trial)))
    objective = Objective(static_data, refit=refit)
    try:
        objective.fit_trial(trial_number, trials, gpu_id)
        exit(0)
    except:
        exit(-1)


