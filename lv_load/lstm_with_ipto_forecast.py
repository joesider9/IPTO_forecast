import os
import joblib

import pandas as pd
import numpy as np

from day_ahead.configuration.config import config
from eforecast.init.initialize import initializer
from eforecast.dataset_creation.dataset_creator import DatasetCreator
from eforecast.dataset_creation.files_manager import FilesManager

from eforecast.dataset_creation.resampling.data_sampler import DataSampler
from eforecast.data_preprocessing.data_split import Splitter
from eforecast.deep_models.tf_1x.network import DeepNetwork
from eforecast.common_utils.eval_utils import compute_metrics



static_data = initializer(config())

experiment_tag = list(sorted(static_data['LSTM']['experiment_tag']))[0]

merge = 'all'
compress = 'load'
scale_method = 'minmax'
what_data = 'lstm'
ID = 6
methods = ['CatBoost_model']

path_weights = os.path.join(static_data['path_model'], 'post_processing', f'post_processing_{ID}')
if not os.path.exists(path_weights):
    os.makedirs(path_weights)


def get_data():
    path_data = static_data['path_data']
    pred_eval = joblib.load(os.path.join(path_data, 'predictions_regressors_eval.pickle'))
    pred_train = joblib.load(os.path.join(path_data, 'predictions_regressors_train.pickle'))
    dataset = DatasetCreator(static_data, recreate=False, train=True, resampling=True)
    dataset.resample_data()
    files_manager = FilesManager(static_data, is_online=False, train=False)
    targ_eval = files_manager.check_if_exists_target()
    files_manager = FilesManager(static_data, is_online=False, train=True)
    targ_train = files_manager.check_if_exists_target()
    targ_train_sampled = files_manager.check_if_exists_target(resampling=True)

    file_ipto = '/media/smartrue/HHD1/George/models/my_projects/IPTO_ver6_ver0/DATA/LoadForecastISP.csv'
    ipto_pred = pd.read_csv(file_ipto, index_col=0, header=0, parse_dates=True)

    return pred_train, pred_eval, targ_train, targ_eval, targ_train_sampled, ipto_pred

def split():
    splitter = Splitter(static_data, is_online=False, train=True)
    splitter.split(refit=False)
    file_manager = FilesManager(static_data, is_online=False, train=True)
    cv_mask = file_manager.check_if_exists_cv_data()
    return cv_mask

def fit(X, y, cv_mask):
    experiment_params = {'name': f'post_processing_{experiment_tag}',
                         'trial_number': 0,
                         'experiment_tag': experiment_tag,
                         'merge': merge,
                         'compress': compress,
                         'what_data': what_data,
                         'conv_dim': 2,
                         'feature_selection_method': None,
                         'scale_nwp_method': scale_method}
    for param, value in static_data['LSTM'].items():
        if param not in experiment_params.keys():
            if isinstance(value, set):
                v = list(value)[0]
            elif isinstance(value, list):
                v = value[0]
            else:
                v = value
            experiment_params[param] = v
    metadata = dict()
    metadata['dates'] = y.index
    experiment_params['experiment'] = static_data['experiments'][experiment_tag]
    model = DeepNetwork(static_data, path_weights, experiment_params, refit=True)
    model.fit(X, y, cv_mask, metadata, gpu_id=0)


def evaluate(model, X, y):
    metadata = dict()
    metadata['dates'] = y.index
    pred = model.predict(X, metadata)
    print(compute_metrics(pred, y, None, f'post_lstm')['mae'].values)


if __name__ == '__main__':
    pred_train, pred_eval, targ_train, targ_eval, targ_train_sampled, ipto_pred = get_data()
    cv_mask = split()

