import copy
import os
import joblib
import numpy as np
import pandas as pd

from eforecast.common_utils.eval_utils import compute_metrics
from eforecast.common_utils.dataset_utils import sync_datasets
from eforecast.dataset_creation.data_feeder import DataFeeder
from eforecast.data_preprocessing.data_scaling import Scaler

CategoricalFeatures = ['hour', 'month', 'sp_index']


class Evaluator:
    def __init__(self, static_data, train=True, refit=False):
        self.static_data = static_data
        self.refit = refit
        self.is_Fuzzy = self.static_data['is_Fuzzy']
        if self.is_Fuzzy:
            self.clusters = joblib.load(os.path.join(static_data['path_model'], 'clusters.pickle'))
            if train:
                self.predictions_resampling = joblib.load(os.path.join(self.static_data['path_data'],
                                                                       'predictions_regressors_resampling.pickle'))
        self.is_Global = self.static_data['is_Global']

        self.predictions = joblib.load(os.path.join(self.static_data['path_data'],
                                                    'predictions_regressors_train.pickle'))

        self.predictions_eval = joblib.load(os.path.join(self.static_data['path_data'],
                                                         'predictions_regressors_eval.pickle'))
        self.scaler = Scaler(static_data, recreate=False, online=False, train=True)
        self.scale_target_method = self.static_data['scale_target_method']
        self.rated = self.static_data['rated']
        self.multi_output = True if self.static_data['horizon_type'] == 'multi-output' else False
        self.evaluator_path = os.path.join(self.static_data['path_model'], 'Results')
        if not os.path.exists(self.evaluator_path):
            os.makedirs(self.evaluator_path)

    def feed_target(self, train=False, resampling=False):
        print('Read target for evaluation....')
        data_feeder = DataFeeder(self.static_data, train=train, resampling=resampling)
        y, _ = data_feeder.feed_target(inverse=False)
        return y

    def evaluate_methods_for_cluster(self, clusterer_method, cluster_name, trial=None):
        results_methods = pd.DataFrame()
        y_scaled = self.feed_target(train=True)
        y_eval_scaled = self.feed_target()
        methods_predictions = self.predictions['clusters'][clusterer_method][cluster_name]
        for method in methods_predictions.keys():
            pred_train_scaled = methods_predictions[method]
            if pred_train_scaled.shape[0] == 0:
                return results_methods
            pred_eval_scaled = self.predictions_eval['clusters'][clusterer_method][cluster_name][method]

            pred_train = self.scaler.inverse_transform_data(pred_train_scaled,
                                                            f'target_{self.scale_target_method}')
            if pred_eval_scaled.shape[0] != 0:
                pred_eval = self.scaler.inverse_transform_data(pred_eval_scaled,
                                                               f'target_{self.scale_target_method}')
            else:
                pred_eval = pred_eval_scaled
            cv_masks = joblib.load(os.path.join(self.clusters[cluster_name], 'cv_mask.pickle'))
            cv_names = ['train', 'val', 'test']
            results_methods_temp = pd.DataFrame()
            if pred_eval_scaled.shape[0] != 0:
                cv_mask = pred_eval.index.intersection(y_eval_scaled.index)
                y_eval = self.scaler.inverse_transform_data(y_eval_scaled,
                                                            f'target_{self.scale_target_method}')
                res_eval = compute_metrics(pred_eval.loc[cv_mask], y_eval.loc[cv_mask], self.rated,
                                           f'{cluster_name}_{method}')['mae'].to_frame()
                res_eval.columns = [f'{column}_eval' for column in res_eval.columns]
                results_methods_temp = res_eval
            if self.static_data['horizon_type'] == 'multi-output':
                col = [f'hour_ahead_{i}' for i in range(self.static_data['horizon'])]
            else:
                col = y_scaled.columns[0]
            name_col = 'target'
            y = self.scaler.inverse_transform_data(y_scaled[col],
                                                   f'target_{self.scale_target_method}')
            p_scaled = copy.deepcopy(pred_train_scaled)
            p = copy.deepcopy(pred_train)
            for cv_name, cv_mask in zip(cv_names, cv_masks):
                cv_mask = cv_mask.intersection(p_scaled.index)

                res = compute_metrics(p.loc[cv_mask], y[col].loc[cv_mask],
                                      self.rated,
                                      f'{cluster_name}_{method}')['mae'].to_frame()
                res.columns = [f'{column}_{name_col}_{cv_name}'
                               for column in res.columns]
                results_methods_temp = pd.concat([results_methods_temp, res], axis=1)
            results_methods = pd.concat([results_methods, results_methods_temp])
        if 'mae_eval' in results_methods.columns:
            empty_row = results_methods.corrwith(results_methods['mae_eval']). \
                to_frame(f'corr_of_{cluster_name}').T
        else:
            empty_row = pd.DataFrame(index=[f'corr_of_{cluster_name}'], columns=results_methods.columns)
        results_methods = pd.concat([results_methods, empty_row])
        rows = [row for row in results_methods.index if 'corr' in row]
        empty_row = results_methods.loc[rows].mean(axis=0).to_frame(f'correlation').T
        results_methods = pd.concat([empty_row, results_methods])
        if not os.path.exists(os.path.join(self.evaluator_path, 'clusters')):
            os.makedirs(os.path.join(self.evaluator_path, 'clusters'))
        if trial is None:
            results_methods.to_csv(
                os.path.join(self.evaluator_path, 'clusters',
                             f'results_methods_{cluster_name}_first.csv'),
                float_format="%.2f")
        else:
            results_methods.to_csv(
                os.path.join(self.evaluator_path, 'clusters',
                             f'results_methods_{cluster_name}_{trial}.csv'),
                float_format="%.2f")

    def evaluate_methods(self):
        if (not os.path.exists(os.path.join(self.evaluator_path, 'clusters', 'results_methods_train.csv')) and
            not os.path.exists(os.path.join(self.evaluator_path, 'clusters', 'results_methods_scaled_train.csv'))) or \
                self.refit:
            results_methods = pd.DataFrame()
            results_methods_scaled = pd.DataFrame()
            y_scaled = pd.concat([self.feed_target(train=True), self.feed_target(train=True, resampling=True)], axis=1)
            y_eval_scaled = self.feed_target()
            for clusterer_method, rules in self.predictions['clusters'].items():
                for cluster_name, methods_predictions in rules.items():
                    for method in methods_predictions.keys():
                        pred_train_scaled = self.predictions['clusters'][clusterer_method][cluster_name][method]
                        if pred_train_scaled.shape[0] == 0:
                            continue
                        pred_resampled_scaled = \
                            self.predictions_resampling['clusters'][clusterer_method][cluster_name][method]
                        if pred_resampled_scaled.shape[0] == 0:
                            continue
                        pred_eval_scaled = self.predictions_eval['clusters'][clusterer_method][cluster_name][method]

                        pred_train = self.scaler.inverse_transform_data(pred_train_scaled,
                                                                        f'target_{self.scale_target_method}')
                        pred_resampled = self.scaler.inverse_transform_data(pred_resampled_scaled,
                                                                            f'target_{self.scale_target_method}')
                        if pred_eval_scaled.shape[0] != 0:
                            pred_eval = self.scaler.inverse_transform_data(pred_eval_scaled,
                                                                           f'target_{self.scale_target_method}')
                        else:
                            pred_eval = pred_eval_scaled
                        cv_masks = joblib.load(os.path.join(self.clusters[cluster_name], 'cv_mask.pickle'))
                        cv_names = ['train', 'val', 'test']
                        results_methods_temp = pd.DataFrame()
                        results_methods_temp_scaled = pd.DataFrame()
                        if pred_eval_scaled.shape[0] != 0:
                            cv_mask = pred_eval.index.intersection(y_eval_scaled.index)
                            res_eval_scaled = compute_metrics(pred_eval_scaled.loc[cv_mask], y_eval_scaled.loc[cv_mask],
                                                              1 if self.rated is not None else None,
                                                              f'{cluster_name}_{method}')['mae'].to_frame()
                            y_eval = self.scaler.inverse_transform_data(y_eval_scaled,
                                                                        f'target_{self.scale_target_method}')
                            res_eval = compute_metrics(pred_eval.loc[cv_mask], y_eval.loc[cv_mask], self.rated,
                                                       f'{cluster_name}_{method}')['mae'].to_frame()
                            res_eval.columns = [f'{column}_eval' for column in res_eval.columns]
                            res_eval_scaled.columns = [f'{column}_eval' for column in res_eval_scaled.columns]
                            results_methods_temp = res_eval
                            results_methods_temp_scaled = res_eval_scaled
                        if self.static_data['horizon_type'] == 'multi-output':
                            cols = [f'hour_ahead_{i}' for i in range(self.static_data['horizon'])]
                            columns = 2 * [cols]
                            for r in ['swap', 'kernel_density', 'linear_reg']:
                                columns.append([f'{r}_{c}' for c in cols])
                        else:
                            columns = [y_scaled.columns[0]] + y_scaled.columns.tolist()
                        for col, name_col in zip(columns,
                                                 ['target', 'target1', 'swap', 'kernel_density', 'linear_reg']):
                            y = self.scaler.inverse_transform_data(y_scaled[col],
                                                                   f'target_{self.scale_target_method}')
                            if name_col == 'target':
                                p_scaled = copy.deepcopy(pred_train_scaled)
                                p = copy.deepcopy(pred_train)
                            else:
                                p_scaled = copy.deepcopy(pred_resampled_scaled)
                                p = copy.deepcopy(pred_resampled)
                            cv_mask = np.concatenate(cv_masks)
                            cv_mask = pd.DatetimeIndex(cv_mask).intersection(p_scaled.index)
                            res_total_scaled = \
                                compute_metrics(p_scaled.loc[cv_mask], y_scaled[col].loc[cv_mask],
                                                1 if self.rated is not None else None,
                                                f'{cluster_name}_{method}')['mae'].to_frame()
                            res_total = compute_metrics(p.loc[cv_mask], y[col].loc[cv_mask], self.rated,
                                                        f'{cluster_name}_{method}')['mae'].to_frame()
                            res_total_scaled.columns = [f'{column}_{name_col}_total' if name_col == 'target'
                                                        else f'{column}_{name_col}_total_resampled'
                                                        for column in res_total_scaled.columns]
                            res_total.columns = [f'{column}_{name_col}_total' if name_col == 'target'
                                                 else f'{column}_{name_col}_total_resampled'
                                                 for column in res_total.columns]
                            results_methods_temp = pd.concat([results_methods_temp, res_total], axis=1)
                            results_methods_temp_scaled = pd.concat([results_methods_temp_scaled, res_total_scaled],
                                                                    axis=1)
                            for cv_name, cv_mask in zip(cv_names, cv_masks):
                                cv_mask = cv_mask.intersection(p_scaled.index)
                                res_resampled_scaled = compute_metrics(p_scaled.loc[cv_mask],
                                                                       y_scaled[col].loc[cv_mask],
                                                                       1 if self.rated is not None else None,
                                                                       f'{cluster_name}_{method}')['mae'].to_frame()
                                res_resampled = compute_metrics(p.loc[cv_mask], y[col].loc[cv_mask],
                                                                self.rated,
                                                                f'{cluster_name}_{method}')['mae'].to_frame()
                                res_resampled.columns = [f'{column}_{name_col}_{cv_name}'
                                                         if name_col == 'target'
                                                         else f'{column}_{name_col}_{cv_name}_resampled'
                                                         for column in res_resampled.columns]
                                res_resampled_scaled.columns = [f'{column}_{name_col}_{cv_name}'
                                                                if name_col == 'target'
                                                                else f'{column}_{name_col}_{cv_name}_resampled'
                                                                for column in res_resampled_scaled.columns]
                                res_resampled_scaled.columns = [f'{column}_scaled'
                                                                for column in res_resampled_scaled.columns]
                                results_methods_temp = pd.concat([results_methods_temp, res_resampled], axis=1)
                                results_methods_temp_scaled = pd.concat([results_methods_temp_scaled,
                                                                         res_resampled_scaled],
                                                                        axis=1)
                        results_methods = pd.concat([results_methods, results_methods_temp])
                        results_methods_scaled = pd.concat([results_methods_scaled, results_methods_temp_scaled])
                    if 'mae_eval' in results_methods.columns:
                        empty_row = results_methods.corrwith(results_methods['mae_eval']). \
                            to_frame(f'corr_of_{cluster_name}').T
                        empty_row_scaled = \
                            results_methods_scaled.corrwith(results_methods_scaled['mae_eval']). \
                                to_frame(f'corr_of_{cluster_name}').T
                    else:
                        empty_row = pd.DataFrame(index=[f'corr_of_{cluster_name}'], columns=results_methods.columns)
                        empty_row_scaled = pd.DataFrame(index=[f'corr_of_{cluster_name}'],
                                                        columns=results_methods_scaled.columns)
                    results_methods = pd.concat([results_methods, empty_row])
                    results_methods_scaled = pd.concat([results_methods_scaled, empty_row_scaled])
            rows = [row for row in results_methods.index if 'corr' in row]
            empty_row = results_methods.loc[rows].mean(axis=0).to_frame(f'correlation').T
            empty_row_scaled = results_methods_scaled.loc[rows].mean(axis=0).to_frame(f'correlation').T
            results_methods = pd.concat([empty_row, results_methods])
            results_methods_scaled = pd.concat([empty_row_scaled, results_methods_scaled])
            for cv_name in ['train', 'val', 'test']:
                columns = [col for col in results_methods.columns if cv_name in col or 'eval' in col or 'total' in col]

                if not os.path.exists(os.path.join(self.evaluator_path, 'clusters')):
                    os.makedirs(os.path.join(self.evaluator_path, 'clusters'))
                results_methods[columns].to_csv(
                    os.path.join(self.evaluator_path, 'clusters', f'results_methods_{cv_name}.csv'),
                    float_format="%.2f")
                columns = [col for col in results_methods_scaled.columns
                           if cv_name in col or 'eval' in col or 'total' in col]
                results_methods_scaled[columns].to_csv(os.path.join(self.evaluator_path, 'clusters',
                                                                    f'results_methods_scaled_{cv_name}.csv'),
                                                       float_format="%.2f")

    def evaluate_clusterer(self, pred_dict, y, y_scaled):
        eval_metrics = pd.DataFrame()
        eval_metrics_scaled = pd.DataFrame()
        for clusterer_name, clusterer_pred_scaled in pred_dict.items():
            clusterer_pred_scaled = clusterer_pred_scaled.mean(axis=1).to_frame(f'{clusterer_name}_clusterer')
            clusterer_pred_scaled, y = sync_datasets(clusterer_pred_scaled, y, name1='pred', name2='target')
            y_scaled = y_scaled.loc[y.index]
            clusterer_pred = self.scaler.inverse_transform_data(clusterer_pred_scaled,
                                                                f'target_{self.scale_target_method}')
            eval_metrics = pd.concat([eval_metrics, compute_metrics(clusterer_pred, y, self.rated,
                                                                    f'{clusterer_name}_clusterer')])
            eval_metrics_scaled = pd.concat([eval_metrics_scaled, compute_metrics(clusterer_pred_scaled, y_scaled,
                                                                                  1 if self.rated is not None else None,
                                                                                  f'{clusterer_name}_clusterer')])
            empty_row = pd.DataFrame(columns=eval_metrics.columns, index=[f'{clusterer_name}_clusterer_ends'])
            eval_metrics = pd.concat([eval_metrics, empty_row])
            eval_metrics_scaled = pd.concat([eval_metrics_scaled, empty_row])
        return eval_metrics, eval_metrics_scaled

    def evaluate_cluster_averages(self, pred_dict, y, y_scaled):
        eval_metrics = pd.DataFrame()
        eval_metrics_scaled = pd.DataFrame()
        for clusterer_name, cluster_group_pred in pred_dict.items():
            for method, method_pred_scaled in cluster_group_pred['averages'].items():
                method_pred_scaled, y = sync_datasets(method_pred_scaled, y, name1='pred', name2='target')
                y_scaled = y_scaled.loc[y.index]
                method_pred = self.scaler.inverse_transform_data(method_pred_scaled,
                                                                 f'target_{self.scale_target_method}')
                eval_metrics = pd.concat([eval_metrics, compute_metrics(method_pred, y, self.rated,
                                                                        f'{clusterer_name}_{method}',
                                                                        multi_output=self.multi_output)])
                eval_metrics_scaled = pd.concat([eval_metrics_scaled, compute_metrics(method_pred_scaled, y_scaled,
                                                                                      1 if self.rated is not None else None,
                                                                                      f'{clusterer_name}_{method}',
                                                                                      multi_output=self.multi_output)])
            empty_row = pd.DataFrame(columns=eval_metrics.columns, index=[f'{clusterer_name}_ends'])
            eval_metrics = pd.concat([eval_metrics, empty_row])
            eval_metrics_scaled = pd.concat([eval_metrics_scaled, empty_row])
        return eval_metrics, eval_metrics_scaled

    def evaluate_distributed(self, pred_dict, y, y_scaled):
        eval_metrics = pd.DataFrame()
        eval_metrics_scaled = pd.DataFrame()
        for distributed_name, distributed_pred_scaled in pred_dict.items():
            distributed_pred_scaled, y = sync_datasets(distributed_pred_scaled, y, name1='pred', name2='target')
            y_scaled = y_scaled.loc[y.index]
            distributed_pred = self.scaler.inverse_transform_data(distributed_pred_scaled,
                                                                  f'target_{self.scale_target_method}')
            eval_metrics = pd.concat([eval_metrics, compute_metrics(distributed_pred, y, self.rated,
                                                                    f'{distributed_name}_clusterer',
                                                                    multi_output=self.multi_output)])
            eval_metrics_scaled = pd.concat([eval_metrics_scaled, compute_metrics(distributed_pred_scaled, y_scaled,
                                                                                  1 if self.rated is not None else None,
                                                                                  f'{distributed_name}_model',
                                                                                  multi_output=self.multi_output)])
            empty_row = pd.DataFrame(columns=eval_metrics.columns, index=[f'{distributed_name}_model_ends'])
            eval_metrics = pd.concat([eval_metrics, empty_row])
            eval_metrics_scaled = pd.concat([eval_metrics_scaled, empty_row])
        return eval_metrics, eval_metrics_scaled

    def evaluate_combining_models(self, pred_dict, y, y_scaled):
        eval_metrics = pd.DataFrame()
        eval_metrics_scaled = pd.DataFrame()
        for combining_model_name, combining_model_pred_scaled in pred_dict.items():
            combining_model_pred_scaled, y = sync_datasets(combining_model_pred_scaled, y, name1='pred', name2='target')
            y_scaled = y_scaled.loc[y.index]
            combining_model_pred = self.scaler.inverse_transform_data(combining_model_pred_scaled,
                                                                      f'target_{self.scale_target_method}')
            eval_metrics = pd.concat([eval_metrics, compute_metrics(combining_model_pred, y, self.rated,
                                                                    f'{combining_model_name}_model',
                                                                    multi_output=self.multi_output)])
            eval_metrics_scaled = pd.concat([eval_metrics_scaled, compute_metrics(combining_model_pred_scaled, y_scaled,
                                                                                  1 if self.rated is not None else None,
                                                                                  f'{combining_model_name}_model',
                                                                                  multi_output=self.multi_output)])
            empty_row = pd.DataFrame(columns=eval_metrics.columns, index=[f'{combining_model_name}_model_ends'])
            eval_metrics = pd.concat([eval_metrics, empty_row])
            eval_metrics_scaled = pd.concat([eval_metrics_scaled, empty_row])
        return eval_metrics, eval_metrics_scaled

    def evaluate_models(self):
        if (not os.path.exists(os.path.join(self.evaluator_path, 'results_methods.csv')) and
            not os.path.exists(os.path.join(self.evaluator_path, 'results_methods_scaled.csv'))) or \
                self.refit:
            results = pd.DataFrame()
            results_scaled = pd.DataFrame()
            y_scaled = self.feed_target(train=True)
            y = self.scaler.inverse_transform_data(y_scaled,
                                                   f'target_{self.scale_target_method}')
            y_eval_scaled = self.feed_target()
            y_eval = self.scaler.inverse_transform_data(y_eval_scaled,
                                                        f'target_{self.scale_target_method}')
            for model_name, model_preds in self.predictions.items():
                if model_name == 'clusterer':
                    res, res_scaled = self.evaluate_clusterer(model_preds, y, y_scaled)
                elif model_name == 'clusters':
                    res, res_scaled = self.evaluate_cluster_averages(model_preds, y, y_scaled)
                elif model_name == 'distributed':
                    res, res_scaled = self.evaluate_distributed(model_preds, y, y_scaled)
                elif model_name == 'models':
                    res, res_scaled = self.evaluate_combining_models(model_preds, y, y_scaled)
                else:
                    raise ValueError(f'Unknown model for evaluation {model_name}')
                results = pd.concat([results, res])
                results_scaled = pd.concat([results_scaled, res_scaled])
                results.to_csv(os.path.join(self.evaluator_path, f'results_models_train.csv'), float_format="%.2f")
                results_scaled.to_csv(os.path.join(self.evaluator_path, f'results_models_train_scaled.csv'),
                                      float_format="%.2f")
                results_eval = pd.DataFrame()
                results_eval_scaled = pd.DataFrame()
                for model_name, model_preds in self.predictions_eval.items():
                    if model_name == 'clusterer':
                        res, res_scaled = self.evaluate_clusterer(model_preds, y_eval, y_eval_scaled)
                    elif model_name == 'clusters':
                        res, res_scaled = self.evaluate_cluster_averages(model_preds, y_eval, y_eval_scaled)
                    elif model_name == 'distributed':
                        res, res_scaled = self.evaluate_distributed(model_preds, y_eval, y_eval_scaled)
                    elif model_name == 'models':
                        res, res_scaled = self.evaluate_combining_models(model_preds, y_eval, y_eval_scaled)
                    else:
                        raise ValueError(f'Unknown model for evaluation {model_name}')
                    results_eval = pd.concat([results_eval, res])
                    results_eval_scaled = pd.concat([results_eval_scaled, res_scaled])
                    results_eval.to_csv(os.path.join(self.evaluator_path, f'results_models_eval.csv'),
                                        float_format="%.6f")
                    results_eval_scaled.to_csv(os.path.join(self.evaluator_path, f'results_models_eval_scaled.csv'),
                                               float_format="%.6f")
