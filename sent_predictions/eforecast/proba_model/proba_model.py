import os
import joblib
import pickle

import pandas as pd
import numpy as np

from eforecast.common_utils.dataset_utils import sync_datasets

from eforecast.deep_models.tf_1x.network import DeepNetwork
from eforecast.dataset_creation.data_feeder import DataFeeder
from eforecast.dataset_creation.files_manager import FilesManager
from eforecast.data_preprocessing.data_scaling import Scaler


class ProbaNetwork:
    def __init__(self, static_data, is_online=False, train=False, refit=False):
        self.static_data = static_data
        self.train = train
        self.online = is_online
        self.refit = refit
        self.multi_output = True if self.static_data['horizon_type'] == 'multi-output' else False
        if self.multi_output:
            self.horizon = np.arange(self.static_data['horizon'])
        else:
            self.horizon = [0]

        if self.multi_output:
            self.path_weights = [os.path.join(self.static_data['path_model'], 'Probabilistic', f'Proba_{hor}')
                                 for hor in range(self.static_data['horizon'])]
            for path in self.path_weights:
                if not os.path.exists(path):
                    os.makedirs(path)
        else:
            self.path_weights = os.path.join(self.static_data['path_model'], 'Probabilistic')
            if not os.path.exists(self.path_weights):
                os.makedirs(self.path_weights)
        if self.train:
            self.predictions = joblib.load(os.path.join(self.static_data['path_data'],
                                                        'predictions_regressors_resampling.pickle'))
        elif not self.train and not self.online:
            self.predictions = joblib.load(os.path.join(self.static_data['path_data'],
                                                        'predictions_regressors_eval.pickle'))
        else:
            self.predictions = joblib.load(os.path.join(self.static_data['path_data'],
                                                        'predictions_regressors_online.pickle'))
        self.params = self.static_data['Proba']
        self.experiment_tag = self.params['experiment_tag']
        self.params['experiment'] = self.static_data['experiments'][self.experiment_tag]
        self.scale_method = self.params['data_type']['scaling']
        self.merge = self.params['data_type']['merge']
        self.compress = self.params['data_type']['compress']
        self.what_data = self.params['data_type']['what_data']
        self.resampling_method = self.params['resampling_method']
        self.scaler = Scaler(static_data, recreate=False, online=False, train=True)
        self.scale_target_method = self.static_data['scale_target_method']
        self.file_manager = FilesManager(self.static_data, is_online=False, train=True)

    def feed_data(self, hor, resampling=False):
        print('Read data for Clustering....')
        data_feeder = DataFeeder(self.static_data, online=self.online, train=self.train, resampling=resampling)
        X, metadata = data_feeder.feed_inputs(merge=self.merge, compress=self.compress,
                                              scale_nwp_method=self.scale_method,
                                              what_data=self.what_data)
        cols_static = [col for col in X.columns if 'hor' not in col]
        cols = cols_static + [col for col in X.columns if f'hor{hor}' in col]
        return X[cols], metadata

    def feed_target(self, hor):
        print('Read target for evaluation....')
        data_feeder = DataFeeder(self.static_data, train=True, resampling=False)
        y, _ = data_feeder.feed_target(inverse=False)
        return y

    def feed_predictions(self, hor):
        pred_df = pd.DataFrame()
        if 'clusters' in self.predictions.keys():
            for clusterer_name in sorted(self.predictions['clusters'].keys()):
                for method in sorted(self.predictions['clusters'][clusterer_name]['averages'].keys()):
                    method_pred = self.predictions['clusters'][clusterer_name]['averages'][method].iloc[:, hor].to_frame()
                    pred_df = pd.concat([pred_df, method_pred], axis=1)
        if 'distributed' in self.predictions.keys():
            for distributed_name in sorted(self.predictions['distributed'].keys()):
                distributed_pred = self.predictions['distributed'][distributed_name].iloc[:, hor].to_frame()
                pred_df = pd.concat([pred_df, distributed_pred], axis=1)
        if 'models' in self.predictions.keys():
            for combining_model_name in sorted(self.predictions['models'].keys()):
                combining_model_pred = self.predictions['models'][combining_model_name].iloc[:, hor].to_frame()
                pred_df = pd.concat([pred_df, combining_model_pred], axis=1)
        return pred_df

    def create_data(self, hor, resampling=False):
        X, metadata = self.feed_data(hor, resampling=resampling)
        metadata['groups'] = ['data', 'prediction']
        pred_df = self.feed_predictions(hor)
        pred_df, X = sync_datasets(pred_df, X, name1='predictions', name2='data_all')
        if self.train:
            y = self.feed_target(hor)
            pred_df, y = sync_datasets(pred_df, y, name1='predictions', name2='target')
            X, y = sync_datasets(X, y, name1='data_all', name2='target')
            if self.static_data['type'] == 'load':
                values_sorted = np.sort(y.values.ravel())
                min_value = 0
                for i in range(y.values.shape[0]):
                    if values_sorted[i] > 0:
                        min_value = values_sorted[i]
                        break
                y = y.clip(min_value, np.inf)
            metadata['dates'] = y.index
            return {'data': X, 'prediction': pred_df}, y, metadata
        else:
            metadata['dates'] = X.index
            return {'data': X, 'prediction': pred_df}, metadata

    def fit(self):
        for hor in self.horizon:
            X, y, metadata = self.create_data(hor, resampling=True)
            if self.multi_output:
                path_weights = self.path_weights[hor]
            else:
                path_weights = self.path_weights
            self.params['groups'] = metadata['groups']
            self.params['name'] = f'Proba_{hor}'
            self.params['merge'] = self.merge
            self.params['compress'] = self.compress
            self.params['scale_nwp_method'] = self.scale_method
            self.params['what_data'] = 'row_all'
            cv_mask = self.file_manager.check_if_exists_cv_data()

            network = DeepNetwork(self.static_data, path_weights, params=self.params, refit=self.refit,
                                  probabilistic=True)
            network.fit(X, y, cv_mask, metadata, gpu_id=0)

    def predict(self):
        proba_predictions = dict()
        for hor in self.horizon:
            X, metadata = self.create_data(hor, resampling=False)
            if self.multi_output:
                path_weights = self.path_weights[hor]
            else:
                path_weights = self.path_weights
            self.params['groups'] = metadata['groups']
            network = DeepNetwork(self.static_data, path_weights, probabilistic=True)
            proba_pred = network.predict(X, metadata)
            tag = self.static_data['horizon_type'] if not self.multi_output else 'hour_ahead_' + str(hor + 1)
            for i in range(len(proba_pred)):
                pred1 = np.clip(proba_pred[i], 0, 1)
                if self.multi_output:
                    pred1 = pd.DataFrame(np.tile(pred1, self.static_data['horizon']), index=metadata['dates'],
                                         columns=[f'hour_ahead{h}' for h in range(self.static_data['horizon'])])
                    pred1 = self.scaler.inverse_transform_data(pred1,
                                                               f'target_{self.scale_target_method}').iloc[:, hor].\
                        to_frame()
                else:
                    pred1 = pd.DataFrame(pred1, index=metadata['dates'],
                                         columns=[f'Proba_q{100 * i}'])
                    pred1 = self.scaler.inverse_transform_data(pred1, f'target_{self.scale_target_method}')
                proba_pred[i] = pred1
            proba_predictions[tag] = pd.DataFrame(np.array(proba_pred).squeeze().T,
                                                  columns=[f'Q{100 * q}' for q in self.params['quantiles']],
                                                  index=metadata['dates'])
            return proba_predictions
