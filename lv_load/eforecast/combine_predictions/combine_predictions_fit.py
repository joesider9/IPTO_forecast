import copy
import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.cluster import KMeans
from eforecast.clustering.clustering_manager import ClusterOrganizer
from eforecast.dataset_creation.files_manager import FilesManager

from eforecast.common_utils.dataset_utils import sync_datasets
from eforecast.dataset_creation.data_feeder import DataFeeder
from eforecast.data_preprocessing.data_scaling import Scaler

from eforecast.combine_predictions.algorithms import bcp_fit
from eforecast.combine_predictions.algorithms import kmeans_fit
from eforecast.combine_predictions.train_combine_classifier import train_classifier
from sklearn.linear_model import ElasticNetCV

if tf.version.VERSION[0] == '1':
    from eforecast.deep_models.tf_1x.concat_networks import DeepConcatNetwork
else:
    from eforecast.deep_models.tf_2x.concat_networks import DeepConcatNetwork
from sklearn.model_selection import train_test_split

CategoricalFeatures = ['hour', 'month', 'sp_index']
gpu_methods = ['TRANSFORMER', 'CNN', 'LSTM', 'RBFNN', 'MLP', 'RBF-CNN']
cpu_methods = ['CatBoost', 'RF', 'lasso', 'RBFols', 'GA_RBFols']


class CombinerFit:
    def __init__(self, static_data, refit=False):
        self.kmeans = None
        self.y_resample = None
        self.num_feats = []
        self.labels = None
        self.cat_feats = []
        self.metadata = None
        self.X = None
        self.y = None
        self.static_data = static_data
        self.refit = refit
        self.is_Fuzzy = self.static_data['is_Fuzzy']
        if self.is_Fuzzy:
            self.clusters = joblib.load(os.path.join(static_data['path_model'], 'clusters.pickle'))
            if static_data['type'] not in {'load', 'fa'}:
                self.predictions_resampled = joblib.load(os.path.join(self.static_data['path_data'],
                                                                  'predictions_regressors_resampling.pickle'))
        else:
            self.clusters = dict()
            self.predictions_resampled = dict()
        self.is_Global = self.static_data['is_Global']
        self.scale_method = self.static_data['combining']['data_type']['scaling']
        self.merge = self.static_data['combining']['data_type']['merge']
        self.compress = self.static_data['combining']['data_type']['compress']
        self.what_data = self.static_data['combining']['data_type']['what_data']
        self.problem_type = self.static_data['type']
        self.n_jobs = self.static_data['n_jobs']

        self.predictions = joblib.load(os.path.join(self.static_data['path_data'],
                                                    'predictions_regressors_train.pickle'))
        self.resampling_method = self.static_data['combining']['resampling_method']
        self.scaler = Scaler(static_data, recreate=False, online=False, train=True)
        self.scale_target_method = self.static_data['scale_target_method']
        self.rated = self.static_data['rated']
        self.combine_methods = self.static_data['combining']['methods']
        self.methods = [method for method, values in static_data['project_methods'].items()
                        if values and (method in cpu_methods or method in gpu_methods)]
        if self.static_data['horizon_type'] == 'multi-output':
            self.horizon = np.arange(self.static_data['horizon'])
        else:
            self.horizon = [0]
        self.combine_clusters = dict()
        for cluster_name, cluster_dir in self.clusters.items():
            path_combine_cluster = os.path.join(cluster_dir, 'combine')
            self.combine_clusters.update({cluster_name: path_combine_cluster})

    def feed_data(self, methods=True):
        which = 'methods' if methods else 'models'
        print(f'Read data for Clustering {which}....')
        if methods:
            data_feeder = DataFeeder(self.static_data, train=True,
                                     resampling=True if self.static_data['type'] not in {'load', 'fa'} else False)
            self.X, self.metadata = data_feeder.feed_inputs(merge=self.merge, compress=self.compress,
                                                            scale_nwp_method=self.scale_method,
                                                            what_data=self.what_data)

            if self.resampling_method is not None:
                data_feeder_target = DataFeeder(self.static_data, train=True,
                                                resampling=True if self.static_data['type'] not in {'load', 'fa'} else False)
                self.y, _ = data_feeder_target.feed_target()
                if self.static_data['type'] not in {'load', 'fa'}:
                    if self.static_data['horizon_type'] == 'multi-output':
                        cols = [col for col in self.y.columns if self.resampling_method in col]
                    else:
                        cols = self.resampling_method
                    if isinstance(cols, str):
                        self.y = self.y[cols].to_frame()
                    else:
                        self.y = self.y[cols]
            else:
                data_feeder_target = DataFeeder(self.static_data, train=True, resampling=False)
                self.y, _ = data_feeder_target.feed_target()
        else:
            data_feeder = DataFeeder(self.static_data, train=True)
            self.X, self.metadata = data_feeder.feed_inputs(merge=self.merge, compress=self.compress,
                                                            scale_nwp_method=self.scale_method,
                                                            what_data=self.what_data)
            self.y, _ = data_feeder.feed_target()
        self.X, self.y = sync_datasets(self.X, self.y, name1='inputs_for_combine', name2='target_for_combine')
        if self.static_data['type'] == 'load':
            values_sorted = np.sort(self.y.values.ravel())
            min_value = 0
            for i in range(self.y.values.shape[0]):
                if values_sorted[i] > 0:
                    min_value = values_sorted[i]
                    break
            self.y = self.y.clip(min_value, np.inf)
        self.metadata['dates'] = self.X.index
        self.cat_feats = [v_name for v_name in self.X.columns
                          for c_feats in CategoricalFeatures if c_feats in v_name]
        self.num_feats = [v_name for v_name in self.X.columns if v_name not in self.cat_feats]

    def apply_kmeans_X(self):
        n_clusters = 5 if self.problem_type == 'pv' else 12
        cat_feats = [f for f in self.cat_feats if 'hour' not in f]
        if len(cat_feats):
            self.kmeans = KMeans(n_clusters=n_clusters)
            self.labels = pd.Series(self.kmeans.fit_predict(self.X[cat_feats].values), index=self.X.index,
                                    name='labels')

    def fit_combine_method(self, combine_method, pred_methods, y, hor, n_predictors, dates, cluster_name,
                           path_combine_method, cluster_dir):
        if not os.path.exists(os.path.join(path_combine_method, f'{combine_method}_model.pickle')) \
                or self.refit:
            if combine_method == 'bcp':
                print('BCP training')
                model = dict()
                w = bcp_fit(pred_methods.values, y.iloc[:, hor].values.reshape(-1, 1),
                            n_predictors)
                model['w'] = w
            elif combine_method == 'kmeans':
                print('Kmeans training')
                n_clusters = 5 if self.problem_type == 'pv' else 12
                kmeans = KMeans(n_clusters=n_clusters)
                labels = pd.Series(kmeans.fit_predict(self.X.loc[dates].values),
                                   index=dates,
                                   name='labels')
                kmeans_model = kmeans_fit(kmeans, labels.values, pred_methods.values,
                                          y.iloc[:, hor].values.reshape(-1, 1))
                model = kmeans_model

            elif combine_method == 'elastic_net':
                print('elastic_net training')
                model = ElasticNetCV(cv=5, max_iter=200000)
                model.fit(pred_methods.values, y.iloc[:, hor].values)
            else:
                if not os.path.exists(os.path.join(path_combine_method, f'results_{cluster_name}_{combine_method}.csv')) \
                        or self.refit:
                    best_predictor = np.argmin(np.abs(pred_methods.values -
                                                      y.iloc[:, hor].values.reshape(-1, 1)), axis=1).reshape(-1, 1)
                    classes = np.unique(best_predictor)
                    predictors_id = []
                    for cl in classes:
                        count = np.where(best_predictor == cl)[0].shape[0]
                        if count > 8:
                            predictors_id.append(cl)
                    best_predictor = np.argmin(np.abs(pred_methods.values[:, predictors_id] -
                                                      y.iloc[:, hor].values.reshape(-1, 1)), axis=1).reshape(-1, 1)
                    best_predictor = pd.DataFrame(best_predictor, index=dates, columns=['target'])
                    train_classifier(self.static_data, combine_method, cluster_name,
                                     path_combine_method, cluster_dir,
                                     best_predictor, predictors_id, refit=self.refit)
                    model = None
                else:
                    model = None
            if model is not None:
                joblib.dump(model, os.path.join(path_combine_method,
                                                f'{combine_method}_model.pickle'))

    def fit_methods(self):
        self.feed_data()
        if self.static_data['type'] not in {'load', 'fa'}:
            predictions = self.predictions_resampled
        else:
            predictions = self.predictions
        if self.is_Fuzzy:
            for hor in self.horizon:
                for clusterer_method, rules in predictions['clusters'].items():
                    for cluster_name, methods_predictions in rules.items():
                        if cluster_name == 'averages':
                            continue
                        n_predictors = len(methods_predictions)
                        if n_predictors > 1:
                            cv_masks = joblib.load(os.path.join(self.clusters[cluster_name], 'cv_mask.pickle'))
                            cv_mask = cv_masks[1].union(cv_masks[-1])
                            pred_methods = []
                            for method in sorted(methods_predictions.keys()):
                                pred = methods_predictions[method].iloc[:, hor].to_frame()
                                pred.columns = [method]
                                pred_methods.append(pred)
                            pred_methods = pd.concat(pred_methods, axis=1)
                            pred_methods[pred_methods < 0] = 0
                            pred_methods = pred_methods.dropna(axis='index')
                            dates = pred_methods.index.intersection(self.y.index)
                            dates = dates.intersection(cv_mask)
                            pred_methods = pred_methods.loc[dates]
                            y = self.y.loc[dates]
                            for combine_method in self.combine_methods:
                                print(f'Fitting combine method {combine_method} for cluster {cluster_name} '
                                      f'and horizon {hor}')
                                path_combine_method = os.path.join(self.combine_clusters[cluster_name],
                                                                   combine_method)
                                if self.static_data['horizon_type'] == 'multi-output':
                                    path_combine_method = os.path.join(path_combine_method, f'hour_ahead_{hor}')
                                if not os.path.exists(path_combine_method):
                                    os.makedirs(path_combine_method)
                                self.fit_combine_method(combine_method, pred_methods, y, hor,
                                                        n_predictors, dates, cluster_name,
                                                        path_combine_method, self.clusters[cluster_name])

    def fit_methods_for_cluster(self, clusterer_method, cluster_name, trial=None):
        methods_predictions = self.predictions['clusters'][clusterer_method][cluster_name]
        self.feed_data()
        cv_masks = joblib.load(os.path.join(self.clusters[cluster_name], 'cv_mask.pickle'))
        cv_mask = cv_masks[1].union(cv_masks[-1])
        for hor in self.horizon:
            n_predictors = len(methods_predictions)
            if n_predictors > 1:
                pred_methods = []
                for method in sorted(methods_predictions.keys()):
                    pred = methods_predictions[method].iloc[:, hor].to_frame()
                    pred.columns = [method]
                    pred_methods.append(pred)
                pred_methods = pd.concat(pred_methods, axis=1)
                pred_methods[pred_methods < 0] = 0
                pred_methods = pred_methods.dropna(axis='index')
                dates = pred_methods.index.intersection(self.y.index)
                dates = dates.intersection(cv_mask)
                pred_methods = pred_methods.loc[dates]
                y = self.y.loc[dates]
                for combine_method in self.combine_methods:
                    print(f'Fitting combine method {combine_method} for cluster {cluster_name} '
                          f'and horizon {hor}')
                    path_combine_method = os.path.join(self.combine_clusters[cluster_name],
                                                       combine_method)
                    if trial is not None:
                        path_combine_method = os.path.join(path_combine_method, f'trial_{trial}')
                    if self.static_data['horizon_type'] == 'multi-output':
                        path_combine_method = os.path.join(path_combine_method, f'hour_ahead_{hor}')
                    if not os.path.exists(path_combine_method):
                        os.makedirs(path_combine_method)
                    self.fit_combine_method(combine_method, pred_methods, y, hor,
                                            n_predictors, dates, cluster_name,
                                            path_combine_method, self.clusters[cluster_name])

    def fit_models(self, combine_methods):
        cluster_name = 'Distributed'
        cluster_path = os.path.join(self.static_data['path_model'], 'Distributed')
        alias_methods = []
        for cm in self.combine_methods:
            alias_methods.append(f'{cm}_classifier' if cm in self.methods else cm)
        self.feed_data(methods=False)
        for hor in self.horizon:
            pred_models = []
            if 'distributed' in self.predictions.keys():
                for distributed_model, distributed_prediction in self.predictions['distributed'].items():
                    pred_models.append(distributed_prediction.iloc[:, hor].to_frame())
            if 'clusters' in self.predictions.keys():
                for clusterer_method, rules in self.predictions['clusters'].items():
                    for combine_method, combine_prediction in rules['averages'].items():
                        if '_'.join(combine_method.split('_')[:-1]) in alias_methods:
                            if self.static_data['horizon_type'] == 'multi-output':
                                pred_models.append(combine_prediction.iloc[:, hor].
                                                   to_frame(f"{clusterer_method}_"
                                                            f"{'_'.join(combine_method.split('_')[:-1])}"
                                                            f"_{combine_prediction.columns[hor]}"))
                            else:
                                pred_models.append(combine_prediction.iloc[:, hor].
                                                   to_frame(f'{clusterer_method}_{combine_prediction.columns[hor]}'))
            n_predictors = len(pred_models)
            if n_predictors == 0:
                if 'clusters' in self.predictions.keys():
                    for clusterer_method, rules in self.predictions['clusters'].items():
                        for combine_method, combine_prediction in rules['averages'].items():
                            if self.static_data['horizon_type'] == 'multi-output':
                                pred_models.append(combine_prediction.iloc[:, hor].
                                                   to_frame(f"{clusterer_method}_"
                                                            f"{'_'.join(combine_method.split('_')[:-1])}"
                                                            f"_{combine_prediction.columns[hor]}"))
                            else:
                                pred_models.append(combine_prediction.iloc[:, hor].
                                to_frame(
                                    f'{clusterer_method}_{combine_prediction.columns[hor]}'))
                else:
                    raise ValueError('Cannot find clusters or distributed models to combine')
            n_predictors = len(pred_models)
            pred_models = pd.concat(pred_models, axis=1)
            pred_models = pred_models.clip(0, np.inf)
            pred_models = pred_models.dropna(axis='index')
            dates = pred_models.index.intersection(self.y.index)
            pred_models = pred_models.loc[dates]
            y = self.y.loc[dates]
            if n_predictors > 1:
                for combine_method in combine_methods:
                    print(f'Fitting combine method {combine_method} for models and horizon {hor}')
                    path_combine_method = os.path.join(self.static_data['path_model'], 'combine_models',
                                                       combine_method)
                    if self.static_data['horizon_type'] == 'multi-output':
                        path_combine_method = os.path.join(path_combine_method, f'hour_ahead_{hor}')
                    if not os.path.exists(path_combine_method):
                        os.makedirs(path_combine_method)
                    self.fit_combine_method(combine_method, pred_models, y, hor,
                                            n_predictors, dates, cluster_name,
                                            path_combine_method, cluster_path)

    def fit_concat_nets(self):
        cluster_name = 'Distributed'
        cluster_path = os.path.join(self.static_data['path_model'], 'Distributed')
        alias_methods = []
        for cm in self.combine_methods:
            alias_methods.append(f'{cm}_classifier' if cm in self.methods else cm)
        self.feed_data(methods=False)
        for hor in self.horizon:
            pred_models = []
            if 'distributed' in self.predictions.keys():
                for distributed_model, distributed_prediction in self.predictions['distributed'].items():
                    pred_models.append(distributed_prediction.iloc[:, hor].to_frame())
            if 'clusters' in self.predictions.keys():
                for clusterer_method, rules in self.predictions['clusters'].items():
                    for combine_method, combine_prediction in rules['averages'].items():
                        if self.static_data['horizon_type'] == 'multi-output':
                            pred_models.append(combine_prediction.iloc[:, hor].
                                               to_frame(f"{clusterer_method}_"
                                                        f"{'_'.join(combine_method.split('_')[:-1])}"
                                                        f"_{combine_prediction.columns[hor]}"))
                        else:
                            pred_models.append(combine_prediction.iloc[:, hor].
                                               to_frame(f'{clusterer_method}_{combine_prediction.columns[hor]}'))
            n_predictors = len(pred_models)

            pred_models = pd.concat(pred_models, axis=1)
            pred_models = pred_models.clip(0, np.inf)
            pred_models = pred_models.dropna(axis='index')
            dates = pred_models.index.intersection(self.y.index)
            pred_models = pred_models.loc[dates]
            y = self.y.loc[dates]
            data = self.X.loc[dates]
            self.metadata['dates'] = dates
            pred_clusters = dict()
            clusterer = ClusterOrganizer(self.static_data, is_online=False, train=True)
            activations = None
            if 'clusters' in self.predictions.keys():
                activations = dict()
                for clusterer_method, rules in self.predictions['clusters'].items():
                    activations[clusterer_method], _ = clusterer.compute_activations(clusterer_method)
                    pred_clusters[clusterer_method] = dict()
                    for cluster_name, cluster_prediction in rules.items():
                        if cluster_name != 'averages':
                            pred_cluster = []
                            for method, pred_df in cluster_prediction.items():
                                if self.static_data['horizon_type'] == 'multi-output':
                                    pred_cluster.append(pred_df.iloc[:, hor].
                                                        to_frame(f"{clusterer_method}_{method}"
                                                                 f"{'_'.join(cluster_name.split('_')[:-1])}"
                                                                 f"_{pred_df.columns[hor]}"))
                                else:
                                    pred_cluster.append(pred_df.iloc[:, hor].
                                                        to_frame(f'{clusterer_method}_{pred_df.columns[hor]}'))
                            pred_cluster = pd.concat(pred_cluster, axis=1)
                            pred_cluster = pred_cluster.clip(0, np.inf)
                            dates_cluster = dates.intersection(pred_cluster.index)
                            pred_clusters[clusterer_method][cluster_name] = pd.DataFrame(0, index=dates,
                                                                                         columns=pred_cluster.columns)
                            pred_clusters[clusterer_method][cluster_name].loc[dates_cluster] \
                                = pred_cluster.loc[dates_cluster].values
            X = dict()
            X['data'] = data
            X['predictions'] = pred_models
            X.update(pred_clusters)
            file_manager = FilesManager(self.static_data, is_online=False, train=True)
            cv_masks = file_manager.check_if_exists_cv_data()
            cv_mask = cv_masks[1].union(cv_masks[2])
            split_test = int(cv_mask.shape[0] * (1 - self.static_data['val_test_ratio']))
            mask_test = cv_mask[split_test:]
            mask_train = cv_mask[:split_test]

            mask_train, mask_val = train_test_split(mask_train, test_size=self.static_data['val_test_ratio'],
                                                    random_state=42)
            cv_mask = [mask_train, mask_test, mask_val]
            combine_path = os.path.join(self.static_data['path_model'], 'combine_models', 'concat_network')
            if not os.path.exists(combine_path):
                os.makedirs(combine_path)
            params = dict()
            params['name'] = f'concat_net_combine_{self.static_data["project_name"]}'
            params['method'] = 'concat_net_combine'
            params['hold_prob'] = 1
            params['batch_size'] = 32
            params['max_iterations'] = 600
            params['warming_iterations'] = 4
            params['learning_rate'] = 1e-4
            params['n_jobs'] = 1
            for key in X.keys():
                if key in self.static_data['clustering']['prediction_for_method']:
                    params[key] = copy.deepcopy(self.static_data['combining']['params_concat_net_with_act'])
                    params[key]['groups'] = []
                    params[key]['merge'] = self.merge
                    params[key]['is_fuzzy'] = False
                    params[key]['is_global'] = True
                    params[key]['is_for_cluster'] = False
                    params[key]['what_data'] = self.what_data
                    params[key]['conv_dim'] = None
                    params[key]['experiment'] = copy.deepcopy(self.static_data['experiments'][params[key]['experiment_tag']])
                    params[key]['replicate_data'] = False
                elif key == 'data':
                    params[key] = copy.deepcopy(self.static_data['combining']['params_concat_net_for_data'])
                    params[key]['groups'] = []
                    params[key]['merge'] = 'row'
                    params[key]['is_fuzzy'] = False
                    params[key]['is_global'] = False
                    params[key]['is_for_cluster'] = False
                    params[key]['what_data'] = self.what_data
                    params[key]['conv_dim'] = None
                    params[key]['experiment'] = copy.deepcopy(self.static_data['experiments'][params[key]['experiment_tag']])
                    params[key]['replicate_data'] = None
                else:
                    params[key] = copy.deepcopy(self.static_data['combining']['params_concat_net_simple'])
                    params[key]['groups'] = []
                    params[key]['merge'] = 'row'
                    params[key]['is_fuzzy'] = False
                    params[key]['is_global'] = False
                    params[key]['is_for_cluster'] = False
                    params[key]['what_data'] = self.what_data
                    params[key]['conv_dim'] = None
                    params[key]['experiment'] = copy.deepcopy(self.static_data['experiments'][params[key]['experiment_tag']])
                    params[key]['replicate_data'] = None
            network = DeepConcatNetwork(self.static_data, combine_path, net_names=[key for key in X.keys()],
                                        params=params, refit=self.refit)
            network.fit(X, y, cv_mask, self.metadata, activations=activations, gpu_id=0)
