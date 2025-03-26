import numpy as np
from short_term.configuration.config_utils import define_n_jobs
from short_term.configuration.config_utils import define_enviroment
from short_term.configuration.config_project import TYPE
from short_term.configuration.config_project import HORIZON_TYPE
from short_term.configuration.experiments_base import experiments

ENV_NAME, PATH_ENV = define_enviroment()

PROJECT_METHODS = {'CNN': False,
                   'LSTM': True,
                   'MLP': True,
                   'RF': False,
                   'CatBoost': True,
                   'lasso': False,
                   'RBFols': False,
                   'GA_RBFols': False,
                   }

FEATURE_SELECTION_METHODS = [None, 'ShapValues']  # Could be None, 'lasso','FeatureImportance', 'ShapValues'

combine_methods = ['kmeans']  # 'bcp', 'elastic_net', 'kmeans', 'CatBoost'

DATA_TYPE_TO_CLUSTER_COMBINE = {'scaling': 'minmax',
                                'merge': 'by_area',
                                'compress': 'load',
                                'what_data': 'row_all'}


def fuzzy_variables():
    if TYPE == 'pv':
        var_imp = [{'hour': [{'flux': {'mfs': 2}},
                             {'flux': {'mfs': 3}},
                             {'flux': {'mfs': 4}},
                             {'flux': {'mfs': 3}},
                             {'flux': {'mfs': 2}},
                             ]}]
    elif TYPE == 'wind':
        var_imp = [{'direction': [{'wind': {'mfs': 6}},
                                  {'wind': {'mfs': 6}},
                                  # {'wind': {'mfs': 3}},
                                  # {'wind': {'mfs': 3}},
                                  ]}]
    elif TYPE == 'load':
        var_imp = [
            {'sp_index': [
                {'hour': {'mfs': 8}, 'month': {'mfs': 3}},
                {'hour': {'mfs': 2}, 'month': {'mfs': 2}},
                {'hour': {'mfs': 2}, 'month': {'mfs': 1}},
            ]
            },
            # {'sp_index': [
            #     {'hour': {'mfs': 4}, 'month': {'mfs': 6}},
            #     {'hour': {'mfs': 2}, 'month': {'mfs': 2}},
            #     {'hour': {'mfs': 1}, 'month': {'mfs': 2}},
            # ]
            # },
            # {'sp_index': [
            #     {'hour': {'mfs': 4}, 'Temp_max': {'mfs': 6}},
            #     {'hour': {'mfs': 2}, 'Temp_max': {'mfs': 2}},
            #     {'hour': {'mfs': 1}, 'Temp_max': {'mfs': 2}},
            # ]
            # }
        ]
    elif TYPE == 'fa':
        var_imp = [
            {'sp_index': [
                {'month': {'mfs': 4}},
                {'month': {'mfs': 1}},
            ]
            },
            {'sp_index': [
                {'temp_max': {'mfs': 4}},
                {'temp_max': {'mfs': 1}},
            ]
            },
            {'sp_index': [
                {'dayweek': {'mfs': 4}},
                {'dayweek': {'mfs': 1}},
            ]
            }
        ]
    else:
        var_imp = []
    return var_imp


def rbf_variables():
    if TYPE == 'pv':
        var_imp = [['hour', 'cloud', 'flux']]
    elif TYPE == 'wind':
        var_imp = [['direction', 'wind']]
    elif TYPE == 'load':
        var_imp = [['sp_index', 'hour', 'month', 'dayweek', 'Temp_max']]
    elif TYPE == 'fa':
        var_imp = [['sp_index', 'month', 'Temp_max', 'dayweek']]
    else:
        var_imp = [[]]
    return var_imp


def config_methods():
    static_data = dict()
    static_data['env_name'] = ENV_NAME
    static_data['path_env'] = PATH_ENV
    static_data['project_methods'] = PROJECT_METHODS
    static_data['feature_selection_methods'] = FEATURE_SELECTION_METHODS
    static_data['experiments'] = experiments
    static_data['resampling'] = True
    static_data['val_test_ratio'] = 0.1

    n_jobs = define_n_jobs()
    static_data['n_jobs'] = n_jobs['n_jobs']
    static_data['n_gpus'] = n_jobs['n_gpus']
    static_data['intra_op'] = n_jobs['intra_op']

    static_data['CNN'] = {'experiment_tag': {'cnn1', 'cnn2', 'cnn3'},
                          'n_trials': 30,
                          'filters': {6, 12, 9},
                          'conv_dim': {2, 3},
                          'batch_size': {64, 32, 128},
                          'max_iterations': 600,
                          'warming_iterations': 5,
                          'learning_rate': {1e-4, 1e-3, 1e-5},
                          'act_func': {'tanh', 'sigmoid', 'elu'},
                          'n_jobs': n_jobs['n_jobs_cnn_3d']
                          }
    static_data['LSTM'] = {'units': 24,
                           'n_trials': 20,
                           'experiment_tag': {'lstm1', 'lstm2', 'lstm3', 'lstm4'},
                           'batch_size': {64, 128, 32},
                           'act_func': {'elu', 'sigmoid', 'tanh'},
                           'max_iterations': 600,
                           'warming_iterations': 5,
                           'learning_rate': {1e-4, 1e-3},
                           'n_jobs': n_jobs['n_jobs_lstm']
                           }
    static_data['MLP'] = {'experiment_tag': {'mlp2', 'mlp3'},
                          'n_trials': 30,
                          'hold_prob': 1,
                          'batch_size': {64, 32},
                          'act_func': {'elu', 'sigmoid', 'tanh'},
                          'max_iterations': 600,
                          'warming_iterations': 5,
                          'learning_rate': {1e-4, 1e-3},
                          'n_jobs': n_jobs['n_jobs_mlp'],
                          }
    static_data['Global'] = {'experiment_tag': {'distributed_cnn1', 'distributed_mlp2'},
                             'keep_n_models': 2,
                             'n_trials': 10,
                             'hold_prob': 1,
                             'batch_size': {32, 64, 128},
                             'filters': 6,
                             'conv_dim': 2,
                             'nwp_data_merge': DATA_TYPE_TO_CLUSTER_COMBINE['merge'],
                             'compress_data': DATA_TYPE_TO_CLUSTER_COMBINE['compress'],
                             'scale_nwp_method': DATA_TYPE_TO_CLUSTER_COMBINE['scaling'],
                             'what_data': DATA_TYPE_TO_CLUSTER_COMBINE['what_data'],
                             'act_func': {'sigmoid'},
                             'max_iterations': 600,
                             'warming_iterations': 4,
                             'learning_rate': {1e-4, 1e-3},
                             'thres_act': 0.001,
                             'min_samples': 200,
                             'max_samples_ratio': 0.8,
                             'train_schedule': 'simple',  # simple or complex
                             'n_rules': {24},
                             'is_fuzzy': {False},  # If True creates its own activations
                             'clustering_method': {'RBF', 'HEBO'},  # None RBF or GA
                             'rbf_var_imp': rbf_variables()[0],
                             'data_type': DATA_TYPE_TO_CLUSTER_COMBINE,  # Data type used for clustering when it is Fuzzy
                             'n_jobs': 1,
                             }
    CLUSTERING_METHOD = ['RBF', 'HEBO']  # RBF or GA or HEBO
    static_data['clustering'] = {'n_jobs': n_jobs['n_jobs'],
                                 'data_type': DATA_TYPE_TO_CLUSTER_COMBINE,
                                 'methods': CLUSTERING_METHOD,
                                 'clusters_for_method': 'both',  # RBF or GA or both
                                 'prediction_for_method': ['RBF', 'HEBO'],  # RBF or GA or both
                                 'thres_act': 0.001,
                                 'Gauss_abbreviations': {'hdd_h', 'temp', 'flux', 'wind', 'temp', 'Temp_max',
                                                         'load', 'power',
                                                         'u10', 'v10' 'gb_h0'},
                                 'fuzzy_var_imp': fuzzy_variables(),
                                 'rbf_var_imp': rbf_variables(),
                                 'explode_clusters': True,
                                 'n_var_lin': 12,
                                 'min_samples': 500,
                                 'max_samples_ratio': 0.6,
                                 'warming_iterations': 4,
                                 'pop': 100,
                                 'gen': 300,
                                 'n_rules': 28,
                                 'params': {'experiment_tag': 'exp_fuzzy1',
                                            'n_trials': 1,
                                            'hold_prob': 1,
                                            'batch_size': 128,
                                            'filters': 12,
                                            'conv_dim': 2,
                                            'train_schedule': 'simple',  # Simple or complex
                                            'act_func': None,
                                            'max_iterations': 1000,
                                            'warming_iterations': 100,
                                            'learning_rate': 0.25e-3,
                                            'n_jobs': 1,
                                            }
                                 }
    static_data['RF'] = {'n_trials': 40,
                         'max_depth': {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 36, 42, 48},
                         'max_features': {'auto', 'sqrt', 'log2', 0.8, 0.7, 0.5, 0.4, 0.3, 0.2},
                         'min_samples_leaf': [2, 200],
                         'min_samples_split': [2, 100],
                         'max_samples': [0.3, 1],
                         'oob_score': True
                         }
    static_data['CatBoost'] = {'n_trials': 40,
                               'iterations': 1000,
                               'learning_rate': {0.01, 0.05, 0.1, 0.2},
                               'l2_leaf_reg': [2, 6],
                               "objective": {"RMSE", "MAE"} if HORIZON_TYPE != 'multi-output' else {'MultiRMSE'},
                               'min_data_in_leaf': [2, 100],
                               "colsample_bylevel": [0.6, 1],
                               "depth": [5, 9],
                               "boosting_type": {"Ordered", "Plain"},
                               "bootstrap_type": {"Bayesian", "Bernoulli", "MVS"},
                               "eval_metric": "MAE" if HORIZON_TYPE != 'multi-output' else 'MultiRMSE',
                               }
    static_data['lasso'] = {'n_trials': 40,
                            'max_iter': 150000,
                            'eps': {0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2},
                            'fit_intercept': {True, False},
                            "selection": {"cyclic", "random"},
                            }
    static_data['RBFols'] = {'n_trials': 50,
                             'warming_iterations': 4,
                             'width': [0.01, 15],
                             'keep': [2, 6]
                             }
    static_data['GA_RBFols'] = {'n_trials': 1000,
                                'warming_iterations': 4,
                                'width': [0.01, 15],
                                'keep': [2, 6]
                                }
    static_data['combining'] = {'methods': combine_methods,
                                'data_type': DATA_TYPE_TO_CLUSTER_COMBINE,
                                'params_concat_net_for_data': {'experiment_tag': 'mlp_for_combine_data',
                                                               'act_func': 'elu',
                                                               },
                                'params_concat_net_with_act': {'experiment_tag': 'distributed_mlp3',
                                                               'act_func': 'sigmoid',
                                                               },
                                'params_concat_net_simple': {'experiment_tag': 'mlp_for_combine_simple',
                                                             'act_func': 'sigmoid',
                                                             },
                                'resampling_method': None}  # Could be 'swap', 'kernel_density', 'linear_reg'
    static_data['Proba'] = {'experiment_tag': 'distributed_mlp2',
                            'method': 'Proba-MLP',
                            'quantiles': np.linspace(0.1, 0.9, 9),
                            'hold_prob': 1,
                            'batch_size': 128,
                            'act_func': 'sigmoid',
                            'max_iterations': 600,
                            'warming_iterations': 0,
                            'learning_rate': 1e-4,
                            'n_jobs': n_jobs['n_jobs_mlp'],
                            'data_type': DATA_TYPE_TO_CLUSTER_COMBINE,
                            'resampling_method': 'swap'  # Could be 'swap', 'kernel_density', 'linear_reg'
                            }
    return static_data
