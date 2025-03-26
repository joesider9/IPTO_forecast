import sys
import os
print(os.getcwd())
sys.path.append(os.getcwd())
from day_ahead.configuration.config import config
from eforecast.init.initialize import initializer
from eforecast.nwp_extraction.nwp_extractor import NwpExtractor

from eforecast.dataset_creation.dataset_creator import DatasetCreator
from eforecast.data_preprocessing.data_pipeline import DataPipeline
from eforecast.clustering.clustering_manager import ClusterOrganizer
from eforecast.data_preprocessing.data_split import Splitter
from eforecast.feature_selection.feature_selection_fit import FeatureSelector

from eforecast.combine_predictions.combine_predictions_fit import CombinerFit
from eforecast.prediction.predict import Predictor
from eforecast.prediction.evaluate import Evaluator
from eforecast.proba_model.proba_model import ProbaNetwork

import traceback
from eforecast.common_utils.train_utils import send_predictions

static_data = initializer(config())
RECREATE_DATASETS = False
BACKEND = 'command_line'  #command_line, threads

if BACKEND == 'command_line':
    from eforecast.training_on_cmd.train_manager import fit_clusters
elif BACKEND == 'threads':
    from eforecast.training.train_manager import fit_clusters
else:
    raise ValueError('Unknown backend')

def nwp_extraction():
    nwp_extractor = NwpExtractor(static_data, recreate=RECREATE_DATASETS, is_online=False, dates=None)
    nwp_extractor.extract()


def create_datasets():
    dataset = DatasetCreator(static_data, recreate=RECREATE_DATASETS, train=True)
    dataset.create_nwp_dataset()
    dataset.merge_nwp_dataset()
    dataset.compress_nwp_datasets()
    dataset.create_row_datasets()
    if static_data['type'] in {'load', 'fa'}:
        dataset.create_lstm_dataset()
    dataset.create_target()


def preprocess_data():
    pipeline = DataPipeline(static_data, recreate=RECREATE_DATASETS, online=False, train=True)
    pipeline.fit_pipe()


def cluster_and_split():
    splitter = Splitter(static_data, is_online=False, train=True)
    splitter.split(refit=RECREATE_DATASETS)
    cluster_organizer = ClusterOrganizer(static_data, is_online=False, train=True, refit=RECREATE_DATASETS)
    cluster_organizer.fit()
    cluster_organizer.create_clusters_and_cvs()


def feature_selection():
    feature_selector = FeatureSelector(static_data, recreate=RECREATE_DATASETS, online=False, train=True)
    feature_selector.fit()


def resample_datasets(refit=False):
    dataset = DatasetCreator(static_data, recreate=refit, train=True, resampling=True)
    dataset.resample_data()


def predict_methods(train=True, resampling=False):
    predictor = Predictor(static_data, train=train, resampling=resampling)
    predictor.predict_regressors()


def evaluate_methods():
    evaluator = Evaluator(static_data, refit=True)
    evaluator.evaluate_methods()


def combine_methods():
    combiner = CombinerFit(static_data, refit=RECREATE_DATASETS)
    combiner.fit_methods()


def predict_combine_methods(train=True, resampling=False):
    predictor = Predictor(static_data, train=train, resampling=resampling)
    predictor.predict_combine_methods()


def compute_averages_methods(train=True, resampling=False):
    predictor = Predictor(static_data, train=train, resampling=resampling)
    predictor.compute_predictions_averages()


def combine_models():
    combiner = CombinerFit(static_data, refit=RECREATE_DATASETS)
    combiner.fit_models()


def predict_combine_models(train=True, resampling=False):
    predictor = Predictor(static_data, train=train, resampling=resampling)
    predictor.predict_combine_models()


def evaluate_models():
    evaluator = Evaluator(static_data, refit=RECREATE_DATASETS)
    evaluator.evaluate_models()


def train_proba_model():
    proba_model = ProbaNetwork(static_data, train=True, refit=True)
    proba_model.fit()


def eval_proba_model():
    proba_model = ProbaNetwork(static_data, train=False)
    proba_model.predict()


if __name__ == '__main__':
    try:
        nwp_extraction()
        create_datasets()
        preprocess_data()
        cluster_and_split()
        feature_selection()
    except Exception as e:
        tb = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
        print("".join(tb))
        send_predictions(" ".join(tb))
        raise e
    fit_clusters(static_data)
    try:
        resample_datasets()
        predict_methods(train=True, resampling=False)
        predict_methods(train=True, resampling=True)
        predict_methods(train=False, resampling=False)
        evaluate_methods()
        combine_methods()
        predict_combine_methods(train=True, resampling=False)
        compute_averages_methods(train=True, resampling=False)
        predict_combine_methods(train=False, resampling=False)
        compute_averages_methods(train=False, resampling=False)
        combine_models()
        predict_combine_models(train=True, resampling=False)
        predict_combine_models(train=False, resampling=False)
        evaluate_models()
        resample_datasets(refit=True)
        predict_methods(train=True, resampling=True)
        predict_combine_methods(resampling=True)
        compute_averages_methods(resampling=True)
        predict_combine_models(resampling=True)
    except Exception as e:
        tb = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
        print("".join(tb))
        send_predictions(" ".join(tb))
        raise e
    # train_proba_model()
    # eval_proba_model()
