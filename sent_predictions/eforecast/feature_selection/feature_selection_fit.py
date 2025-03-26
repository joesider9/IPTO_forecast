import os
import joblib

import numpy as np
import pandas as pd

from eforecast.common_utils.dataset_utils import sync_datasets
from eforecast.common_utils.dataset_utils import sync_target_with_tensors
from eforecast.common_utils.dataset_utils import sync_data_with_dates
from eforecast.dataset_creation.files_manager import FilesManager
from eforecast.dataset_creation.data_feeder import DataFeeder

from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LassoCV, Lasso
from sklearn.linear_model import MultiTaskLassoCV, MultiTaskLasso
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from catboost import Pool

CategoricalFeatures = ['dayweek', 'hour', 'month', 'sp_index']


class FeatureSelector:
    def __init__(self, static_data, recreate=False, online=False, train=False):
        self.feature_selectors = dict()
        self.online = online
        self.train = train
        self.static_data = static_data
        self.is_Fuzzy = self.static_data['is_Fuzzy']
        if self.is_Fuzzy:
            self.clusters = joblib.load(os.path.join(static_data['path_model'], 'clusters.pickle'))

        cluster_path = os.path.join(static_data['path_model'], 'Distributed')
        if not os.path.exists(cluster_path):
            os.makedirs(cluster_path)
        self.clusters.update({'Distributed': cluster_path})
        self.recreate = recreate
        self.nwp_data_merge = self.static_data['nwp_data_merge']
        self.nwp_data_compress = self.static_data['compress_data']
        self.scale_nwp_method = self.static_data['scale_nwp_method']
        self.data_structure = self.static_data['data_structure']
        self.feature_selection_methods = self.static_data['feature_selection_methods']
        self.rated = static_data['rated']
        self.files_manager = FilesManager(static_data, is_online=online)

    def load_data(self, merge, compress, scale_method, what_data):
        data_feeder = DataFeeder(self.static_data, online=self.online, train=self.train)

        X, metadata = data_feeder.feed_inputs(merge=merge, compress=compress,
                                              scale_nwp_method=scale_method,
                                              what_data=what_data)
        y, _ = data_feeder.feed_target()
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

    def estimator(self, method, alpha=None, multi_output=False):
        if method == 'lasso':
            if alpha is None:
                if multi_output:
                    return MultiTaskLassoCV(max_iter=150000, n_jobs=self.static_data['n_jobs'])
                else:
                    return LassoCV(max_iter=150000, positive=True, n_jobs=self.static_data['n_jobs'])
            else:
                if multi_output:
                    return MultiTaskLasso(max_iter=150000, alpha=alpha)
                else:
                    return Lasso(max_iter=150000, positive=True, alpha=alpha)
        elif method == 'FeatureImportance':
            return RandomForestRegressor(n_estimators=100, n_jobs=self.static_data['n_jobs'])
        elif method == 'ShapValues':
            loss_function = 'MultiRMSE' if self.static_data['horizon_type'] == 'multi-output' else 'RMSE'
            return CatBoostRegressor(iterations=100, loss_function=loss_function, allow_writing_files=False)
        else:
            raise ValueError(f'Unknown feature selection method {method}')

    @staticmethod
    def importance_vector(method, estimator, train_pool=None):
        if method == 'lasso':
            return np.abs(estimator.coef_)
        elif method == 'FeatureImportance':
            return estimator.feature_importances_
        elif method == 'ShapValues':
            fi = estimator.get_feature_importance(data=train_pool,
                                                  type=method,
                                                  prettified=True)
            if isinstance(fi, np.ndarray):
                return np.mean(np.mean(np.abs(fi), axis=1), axis=0)[:-1]
            else:
                return fi.abs().mean(axis=0).values[:-1]
        else:
            raise ValueError(f'Unknown feature selection method {method}')

    def fit_catboost(self, selector, x_train, y_train, cols, cat_feats):
        try:
            selector.fit(x_train, y_train[cols], cat_features=cat_feats, verbose=False)
        except:
            if self.static_data['horizon_type'] == 'multi-output':
                selector = self.estimator('ShapValues', multi_output=True)
                selector.fit(x_train, y_train[cols] +
                             pd.DataFrame(np.random.uniform(0, 0.0001, list(y_train[cols].shape)),
                                          index=y_train.index, columns=cols),
                             cat_features=cat_feats, verbose=False)
            else:
                raise ValueError('Cannot fit Catboost')
        return selector

    def fit_method(self, method, x_train, y_train, x_test, y_test, cat_feats=None):
        if y_train.shape[1] > 1:
            cols = y_train.columns
            multi_output = True
        else:
            cols = y_train.columns[0]
            multi_output = False
        feature_selector = dict()
        thresholds = [0] + np.logspace(-6, -1, 6).tolist() + [1]
        selector = self.estimator(method, multi_output=multi_output)
        if cat_feats is not None:
            cat_feats = list(set([v_name for v_name in x_train.columns
                         for c_feats in CategoricalFeatures if c_feats in v_name]))
            x_train[cat_feats] = x_train[cat_feats].astype('int')
        if method == 'ShapValues':
            selector = self.fit_catboost(selector, x_train, y_train, cols, cat_feats)
        else:
            selector.fit(x_train, y_train[cols])
        if method == 'lasso':
            alpha = selector.alpha_
        else:
            alpha = None
        mae = []
        importance = self.importance_vector(method, selector, Pool(x_train, y_train[cols], cat_features=cat_feats))
        if len(importance.shape) > 1 and multi_output:
            importance = np.sum(importance, axis=0)
        for threshold in thresholds:
            indices = np.where(importance > threshold)[0]
            if indices.shape[0] > 4:
                names = x_train.columns[indices]
                selector_temp = self.estimator(method, alpha=alpha, multi_output=multi_output)
                if cat_feats is not None:
                    cat_feats = list(set([v_name for v_name in names for c_feats in CategoricalFeatures
                                          if c_feats in v_name]))
                    x_train[cat_feats] = x_train[cat_feats].astype('int')
                    x_test[cat_feats] = x_test[cat_feats].astype('int')
                if method == 'ShapValues':
                    selector_temp = self.fit_catboost(selector_temp, x_train[names], y_train, cols, cat_feats)
                    pred = selector_temp.predict(Pool(x_test[names], cat_features=cat_feats))
                else:
                    selector_temp.fit(x_train[names], y_train[cols])
                    pred = selector_temp.predict(x_test[names])
                if self.rated is not None:
                    mae.append(np.mean(np.abs(pred - y_test[cols].values)))
                else:
                    mae.append(np.mean(np.abs(pred - y_test[cols].values) / y_test[cols].values))
            else:
                mae.append(np.inf)
        if np.all(np.isinf(mae)):
            best_threshold = np.mean(importance)
        else:
            best_threshold = thresholds[np.argmin(mae)]
        feature_indices = np.where(importance > best_threshold)[0]
        if feature_indices.shape[0] < 4:
            feature_indices = np.arange(importance.shape[0])
        feature_names = x_train.columns[feature_indices]
        feature_selector['indices'] = feature_indices
        feature_selector['names'] = feature_names
        return feature_selector

    def fit_method_lstm(self, method, x_train, y_train, x_test, y_test):
        feature_selector = dict()
        x_train_, x_test_ = self.compress_lstm(x_train, x_test, axis=1)
        feature_selector_ = self.fit_method_on_compressed(method, x_train_, y_train, x_test_, y_test)
        ind_lags = feature_selector_['indices']
        x_train_, x_test_ = self.compress_lstm(x_train[:, ind_lags, :], x_test[:, ind_lags, :], axis=2)
        feature_selector_ = self.fit_method_on_compressed(method, x_train_, y_train, x_test_, y_test)
        ind_vars = feature_selector_['indices']

        feature_selector['lags'] = ind_lags
        feature_selector['variables'] = ind_vars
        return feature_selector

    @staticmethod
    def compress_lstm(x, x_test, axis=2):
        x_compress = None
        x_compress_test = None
        for var in range(x.shape[axis]):
            if axis == 1:
                X = x[:, var, :]
                X_test = x_test[:, var, :]
            elif axis == 2:
                X = x[:, :, var]
                X_test = x_test[:, :, var]
            else:
                raise ValueError("Axis parameter should be 1 or 2")
            X1 = np.concatenate([X, X_test])
            m = MLPRegressor(activation='identity', hidden_layer_sizes=(1,), max_iter=1000).fit(X1, X1)
            x_ = np.matmul(X, m.coefs_[0])
            x_test_ = np.matmul(X_test, m.coefs_[0])
            x_compress = np.concatenate([x_compress, x_], axis=1) if x_compress is not None else x_
            x_compress_test = np.concatenate([x_compress_test, x_test_], axis=1) \
                if x_compress_test is not None else x_test_
        return x_compress, x_compress_test

    def fit_method_on_compressed(self, method, x_train, y_train, x_test, y_test):
        cols = [f'col_{i}' for i in range(x_train.shape[1])]
        x_train = pd.DataFrame(x_train, columns=cols)
        x_test = pd.DataFrame(x_test, columns=cols)
        cat_feats = [col for col in x_train.columns if x_train[col].unique().shape[0] < 30]
        x_train[cat_feats] = x_train[cat_feats].astype('int')
        x_test[cat_feats] = x_test[cat_feats].astype('int')
        feature_selector_ = self.fit_method(method, x_train, y_train, x_test, y_test, cat_feats=cat_feats)
        return feature_selector_

    def fit_method_distributed(self, method, x_train, y_train, x_test, y_test):
        feature_selector = dict()
        feature_selector_ = self.fit_method_on_compressed(method, x_train, y_train, x_test, y_test)
        feature_selector['variables'] = feature_selector_['indices']
        return feature_selector

    @staticmethod
    def compress_row_dict_distributed(x):
        x_compress = None
        for var in range(x.shape[2]):
            x_ = PCA(n_components=1).fit_transform(x[:, :, var])
            x_compress = np.concatenate([x_compress, x_], axis=1) if x_compress is not None else x_
        return x_compress

    def _fit(self, x, y, cv_mask, dataset_name, cluster_name, feature_selectors, metadata=None):
        if self.static_data['type'] == 'load':
            values_sorted = np.sort(y.values.ravel())
            min_value = 0
            for i in range(y.values.shape[0]):
                if values_sorted[i] > 0:
                    min_value = values_sorted[i]
                    break
            y = y.clip(min_value, np.inf)
        if metadata is not None:
            if 'row_dict_distributed' in dataset_name:
                x = self.compress_row_dict_distributed(x)
            x_train = np.concatenate([sync_data_with_dates(x, cv_mask[0], dates_x=metadata['dates']),
                                      sync_data_with_dates(x, cv_mask[1], dates_x=metadata['dates'])])
            y_train = pd.concat([sync_data_with_dates(y, cv_mask[0]), sync_data_with_dates(y, cv_mask[1])])

            x_test = sync_data_with_dates(x, cv_mask[2], dates_x=metadata['dates'])
            y_test = sync_data_with_dates(y, cv_mask[2])
        else:
            x_train = pd.concat([sync_data_with_dates(x, cv_mask[0]), sync_data_with_dates(x, cv_mask[1])])
            y_train = pd.concat([sync_data_with_dates(y, cv_mask[0]), sync_data_with_dates(y, cv_mask[1])])

            x_test = sync_data_with_dates(x, cv_mask[2])
            y_test = sync_data_with_dates(y, cv_mask[2])

        for method in self.feature_selection_methods:
            if method is not None:
                if f'feature_selector_{cluster_name}_{method}_{dataset_name}' not in feature_selectors.keys():
                    print(f'Fitting feature_selector_{cluster_name}_{method}_{dataset_name}')
                    if 'lstm' in dataset_name:
                        feature_selectors[f'feature_selector_{cluster_name}_{method}_{dataset_name}'] = \
                            self.fit_method_lstm(method, x_train, y_train, x_test, y_test)
                    else:
                        if isinstance(x_train, pd.DataFrame):
                            feature_selectors[f'feature_selector_{cluster_name}_{method}_{dataset_name}'] = \
                                self.fit_method(method, x_train, y_train, x_test, y_test)
                        elif 'row_dict_distributed' in dataset_name:
                            feature_selectors[f'feature_selector_{cluster_name}_{method}_{dataset_name}'] = \
                                self.fit_method_distributed(method, x_train, y_train, x_test, y_test)
                        else:
                            raise ValueError('Cannot recognize action for feature selection')

        return feature_selectors

    def fit(self):
        fit_lstm = False
        if any([data_struct == 'lstm' for data_struct in self.data_structure]):
            fit_lstm = True
        data_structure = [data_struct for data_struct in self.data_structure
                          if data_struct in {'row', 'row_all',
                                             'row_dict',
                                             'row_dict_distributed'}]
        if len(data_structure) == 0:
            raise ValueError('Cannot find what_data structure to fit MLP')
        for cluster_name, cluster_path in self.clusters.items():
            feature_selectors = dict()
            filename = os.path.join(cluster_path, 'feature_selectors.pickle')
            if os.path.exists(filename):
                if self.recreate:
                    if os.path.exists(filename):
                        os.remove(filename)
                else:
                    feature_selectors.update(joblib.load(filename))
                    continue
            if cluster_name == 'Distributed':
                cv_mask = self.files_manager.check_if_exists_cv_data()
            else:
                cv_mask = joblib.load(os.path.join(cluster_path, 'cv_mask.pickle'))
            if fit_lstm:
                for scale_method in self.scale_nwp_method:
                    x, y, metadata = self.load_data('all', 'load', scale_method, 'lstm')
                    dataset_name = f'{scale_method}_lstm'
                    feature_selectors = self._fit(x, y, cv_mask, dataset_name, cluster_name,
                                                  feature_selectors, metadata=metadata)
            for merge in self.nwp_data_merge:
                for compress in self.nwp_data_compress:
                    for scale_method in self.scale_nwp_method:
                        for what_data in data_structure:
                            x, y, metadata = self.load_data(merge, compress, scale_method, what_data)
                            groups = metadata['groups']
                            if len(groups) == 0 or what_data in {'row', 'row_all'}:
                                dataset_name = f'{merge}_{compress}_{scale_method}_{what_data}'
                                feature_selectors = self._fit(x, y, cv_mask, dataset_name, cluster_name,
                                                              feature_selectors,
                                                              metadata=metadata if what_data == 'row_dict_distributed'
                                                              else None)
                            else:
                                for group in groups:
                                    group_name = '_'.join(group) if isinstance(group, tuple) else group
                                    dataset_name = f'{merge}_{compress}_{scale_method}_{group_name}_{what_data}'
                                    feature_selectors = self._fit(x[group_name], y, cv_mask, dataset_name, cluster_name,
                                                                  feature_selectors,
                                                                  metadata=metadata
                                                                  if what_data == 'row_dict_distributed'
                                                                  else None)
            self.save(cluster_path, feature_selectors)

    @staticmethod
    def save(cluster_path, feature_selectors):
        filename = os.path.join(cluster_path, 'feature_selectors.pickle')
        joblib.dump(feature_selectors, filename)

    def transform(self, cluster_name, data, method, merge, compress, scale_method, what_data, groups=None):
        feature_selectors = dict()
        filename = os.path.join(self.clusters[cluster_name], 'feature_selectors.pickle')
        if os.path.exists(filename):
            feature_selectors.update(joblib.load(filename))
        if method is not None:
            if len(groups) == 0 or what_data == 'row_all':
                if what_data == 'lstm':
                    dataset_name = f'{scale_method}_{what_data}'
                else:
                    dataset_name = f'{merge}_{compress}_{scale_method}_{what_data}'
                feature_selector = self.feature_selectors[f'feature_selector_{cluster_name}_{method}_{dataset_name}']
                if what_data == 'lstm':
                    data_new = data[:, feature_selector['lags'], feature_selector['variables']]
                else:
                    if 'row_dict_distributed' in dataset_name:
                        data_new = data[feature_selector['variables']]
                    else:
                        data_new = data[feature_selector['names']]
            else:
                data_new = dict()
                for group in groups:
                    group_name = '_'.join(group) if isinstance(group, tuple) else group
                    dataset_name = f'{merge}_{compress}_{scale_method}_{group_name}_{what_data}'
                    feature_selector = feature_selectors[
                        f'feature_selector_{cluster_name}_{method}_{dataset_name}']
                    if 'row_dict_distributed' in dataset_name:
                        data_new[group_name] = data[group_name][feature_selector['variables']]
                    else:
                        data_new[group_name] = data[group_name][feature_selector['names']]
        else:
            data_new = data
        return data_new
