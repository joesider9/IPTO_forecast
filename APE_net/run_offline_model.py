import os
import joblib
import pandas as pd

from intraday.configuration.config import config
from eforecast.init.initialize import initializer
from eforecast.nwp_extraction.nwp_extractor import NwpExtractor
from eforecast.dataset_creation.dataset_creator import DatasetCreator
from eforecast.prediction.predict import Predictor

dates = pd.date_range(start='2017-01-01', end='2023-05-10')

static_data = initializer(config())


def nwp_extraction():
    nwp_extractor = NwpExtractor(static_data, recreate=False, is_online=True, dates=dates)
    nwp_extractor.extract()


def create_datasets():
    dataset = DatasetCreator(static_data, train=False, is_online=True, dates=dates)
    dataset.create_nwp_dataset()
    dataset.merge_nwp_dataset()
    dataset.compress_nwp_datasets()
    dataset.create_row_datasets()
    if static_data['type'] in {'load', 'fa'}:
        dataset.create_lstm_dataset()


def predict():
    predictor = Predictor(static_data, train=False, online=True)
    predictor.predict_regressors()
    predictor.predict_combine_methods()
    predictor.compute_predictions_averages()
    predictor.predict_combine_models()


if __name__ == '__main__':
    # nwp_extraction()
    create_datasets()
    predict()
