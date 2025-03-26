import os
from eforecast.init.initialize import initializer
from eforecast.dataset_creation.dataset_creator import DatasetCreator
from eforecast.clustering.clustering_manager import ClusterOrganizer

from eforecast.prediction.predict import Predictor

def create_datasets(static_data, dates):
    dataset = DatasetCreator(static_data, train=False, is_online=True, dates=dates)
    dataset.create_nwp_dataset()
    dataset.merge_nwp_dataset()
    dataset.compress_nwp_datasets()
    dataset.create_row_datasets()
    if static_data['type'] in {'load', 'fa'}:
        dataset.create_lstm_dataset()


def predict_intra(dates, method, model_horizon='intra'):
    if model_horizon == 'intra':
        from intraday.configuration.config import config
    elif model_horizon == 'day_ahead':
        from day_ahead.configuration.config import config
    else:
        raise ValueError('Unknown model horizon type')
    static_data = initializer(config(), online=True)
    cluster_organizer = ClusterOrganizer(static_data, is_online=True, train=False, refit=False)
    cluster_organizer.update_cluster_folders()
    create_datasets(static_data, dates)
    predictor = Predictor(static_data, train=False, online=True)
    predictor.predict_regressors()
    predictor.predict_combine_methods()
    predictor.compute_predictions_averages()
    predictor.predict_combine_models()
    return predictor.predictions['models'][method]