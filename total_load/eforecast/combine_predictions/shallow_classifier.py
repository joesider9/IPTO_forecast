import os
import joblib

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from catboost import Pool

from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

pd.set_option('display.expand_frame_repr', False)

CategoricalFeatures = ['dayweek', 'hour', 'month', 'sp_index']


class ShallowModelClassifier:
    def __init__(self, static_data, path_weights, predictors=None, params=None, n_jobs=1, refit=False):
        self.best_mae_val = None
        self.best_mae_test = None
        self.is_trained = False
        self.refit = refit
        self.static_data = static_data
        self.rated = static_data['rated']
        self.n_jobs = n_jobs
        self.predictors = predictors
        if params is not None:
            self.params = params
            self.method = self.params['method']
            self.name = self.params['name']
            self.merge = self.params['merge']
            self.compress = self.params['compress']
            self.scale_nwp_method = self.params['scale_nwp_method']
            self.groups = self.params['groups']
            if self.method == 'CatBoost':
                self.model = CatBoostClassifier(task_type="GPU",
                                                devices='0' + ''.join(
                                                    [f':{i}' for i in range(1, self.static_data['n_gpus'])]),
                                                allow_writing_files=False)
            else:
                raise ValueError(f'Unknown method {self.method} for shallow models')
            self.best_params = {'iterations': 1000,
                                'learning_rate': 0.005,
                                'l2_leaf_reg': 1,
                                "objective": "RMSE",
                                'min_data_in_leaf': 2,
                                "depth": 1,
                                "boosting_type": "Ordered",
                                "bootstrap_type": "Bayesian",
                                "eval_metric": "MAE"}
            for param, value in self.params.items():
                if param in self.best_params.keys():
                    self.best_params[param] = value
            self.model.set_params(**self.best_params)
        self.path_weights = path_weights
        try:
            if not self.refit:
                self.load()
        except:
            pass
        self.refit = refit
        if not hasattr(self, 'params'):
            raise RuntimeError('The network has no params. You should train the net providing params')

    @staticmethod
    def get_slice(x, mask, meta_data, y=None):
        dates = meta_data['dates']
        mask = mask.intersection(dates)
        indices = dates.get_indexer(mask)
        y_slice = y.iloc[indices] if y is not None else None
        if isinstance(x, pd.DataFrame):
            X_slice = x.iloc[indices]
        else:
            raise ValueError('Wrong type of input X for shallow models')
        return X_slice, y_slice

    def fit(self, X, y, cv_masks, meta_data):
        cat_feats = list(set([v_name for v_name in X.columns
                              for c_feats in CategoricalFeatures if c_feats in v_name]))
        X[cat_feats] = X[cat_feats].astype('int')

        X_train, y_train = self.get_slice(X, cv_masks[0], meta_data, y=y)
        X_val, y_val = self.get_slice(X, cv_masks[1], meta_data, y=y)
        X_test, y_test = self.get_slice(X, cv_masks[2], meta_data, y=y)

        if self.method in {'CatBoost'}:
            self.model.fit(X_train, y_train, cat_features=cat_feats, use_best_model=True, eval_set=[(X_val, y_val)],
                           verbose=False,
                           early_stopping_rounds=30)
            y_pred = self.model.predict_proba(Pool(X_test, cat_features=cat_feats))
        else:
            raise ValueError(f'Unknown method {self.method} for shallow models')

        y_test = y_test.values
        if len(self.model.classes_) == 2:
            self.best_mae_test = roc_auc_score(y_test.ravel(), y_pred[:, 1])
        else:
            self.best_mae_test = roc_auc_score(y_test.ravel(), y_pred, multi_class='ovr')
        self.is_trained = True
        self.save()
        return self.best_mae_test

    def predict_proba(self, X, metadata, cluster_dates=None):
        cat_feats = list(set([v_name for v_name in X.columns for c_feats in CategoricalFeatures if c_feats in v_name]))
        X[cat_feats] = X[cat_feats].astype('int')
        cluster_dates = metadata['dates'] if cluster_dates is None else cluster_dates.intersection(metadata['dates'])
        X, _ = self.get_slice(X, cluster_dates, metadata)
        if self.method == 'CatBoost':
            return self.model.predict_proba(Pool(X, cat_features=cat_feats))
        else:
            raise ValueError(f'Unknown method {self.method} for shallow models')

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
        joblib.dump(tmp_dict, os.path.join(self.path_weights, 'net_weights.pickle'))
