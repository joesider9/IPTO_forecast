import sys
import os
print(os.getcwd())
sys.path.append(os.getcwd())
from eforecast.init.initialize import initializer

from eforecast.dataset_creation.dataset_creator import DatasetCreator
from eforecast.combine_predictions.combine_predictions_fit import CombinerFit
from eforecast.prediction.predict import Predictor
from eforecast.prediction.evaluate import Evaluator
from eforecast.proba_model.proba_model import ProbaNetwork
from eforecast.data_preprocessing.data_split import Splitter


import traceback
from eforecast.common_utils.train_utils import send_predictions
RECREATE_DATASETS = False


def resample_datasets(static_data, refit=False):
    dataset = DatasetCreator(static_data, recreate=refit, train=True, resampling=True)
    dataset.resample_data()


def predict_methods(static_data, train=True, resampling=False):
    predictor = Predictor(static_data, train=train, resampling=resampling)
    predictor.predict_regressors()


def evaluate_methods(static_data):
    evaluator = Evaluator(static_data, refit=True)
    evaluator.evaluate_methods()


def evaluate_averages(static_data, train=True):
    evaluator = Evaluator(static_data, train=train, refit=True)
    evaluator.evaluate_averages()

def combine_methods(static_data):
    combiner = CombinerFit(static_data, refit=True)
    combiner.fit_methods()


def predict_combine_methods(static_data, train=True, resampling=False):
    predictor = Predictor(static_data, train=train, resampling=resampling)
    predictor.predict_combine_methods()


def compute_averages_methods(static_data, train=True, resampling=False, only_methods=False, only_combine_methods=False):
    predictor = Predictor(static_data, train=train, resampling=resampling)
    predictor.compute_predictions_averages(only_methods=only_methods, only_combine_methods=only_combine_methods)


def combine_models(static_data, combine_methods=None):
    combiner = CombinerFit(static_data, refit=True)
    if combine_methods is not None:
        combiner.fit_models(combine_methods)
    combiner.fit_concat_nets()


def predict_combine_models(static_data, train=True, resampling=False, combine_methods=None):
    predictor = Predictor(static_data, train=train, resampling=resampling)
    predictor.predict_combine_models(combine_methods)
    predictor.predict_concat_nets()


def evaluate_models(static_data):
    evaluator = Evaluator(static_data, refit=RECREATE_DATASETS)
    evaluator.evaluate_models()


def train_proba_model(static_data):
    proba_model = ProbaNetwork(static_data, train=True, refit=True)
    proba_model.fit()


def eval_proba_model(static_data):
    proba_model = ProbaNetwork(static_data, train=False, refit=RECREATE_DATASETS)
    proba_model.predict()


def run_combine_models(static_data):
    # try:
    # if static_data['type'] not in {'load', 'fa'}:
    #     resample_datasets(static_data)
    #     predict_methods(static_data, train=True, resampling=True)
    # predict_methods(static_data, train=True, resampling=False)
    # predict_methods(static_data, train=False, resampling=False)
    # evaluate_methods(static_data)
    # compute_averages_methods(static_data, train=True, resampling=False, only_methods=True, only_combine_methods=False)
    # compute_averages_methods(static_data, train=False, resampling=False, only_methods=True, only_combine_methods=False)
    # evaluate_averages(static_data)
    # combine_methods(static_data)
    # predict_combine_methods(static_data, train=True, resampling=False)
    # compute_averages_methods(static_data, train=True, resampling=False, only_methods=False, only_combine_methods=False)
    # predict_combine_methods(static_data, train=False, resampling=False)
    # compute_averages_methods(static_data, train=False, resampling=False, only_methods=False, only_combine_methods=False)
    # evaluate_averages(static_data)
    combine_models(static_data, combine_methods=['kmeans'])
    predict_combine_models(static_data, train=True, resampling=False, combine_methods=['kmeans'])
    predict_combine_models(static_data, train=False, resampling=False, combine_methods=['kmeans'])
    evaluate_models(static_data)

    # except Exception as e:
    #     tb = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
    #     print("".join(tb))
    #     send_predictions(" ".join(tb))
    #     return
    # try:
    #     splitter = Splitter(static_data, is_online=False, train=True)
    #     splitter.split(refit=True)
    #     predict_methods(static_data, train=True, resampling=True)
    #     predict_combine_methods(static_data, train=True, resampling=True)
    #     compute_averages_methods(static_data, train=True, resampling=True)
    #     predict_combine_models(static_data, train=True, resampling=True)
    #     train_proba_model(static_data)
    #     eval_proba_model(static_data)
    #
    # except Exception as e:
    #     tb = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
    #     print("".join(tb))
    #     send_predictions(" ".join(tb))
    #     return

if __name__ == '__main__':
    # from day_ahead.configuration.config import config
    # static_data = initializer(config())
    # run_combine_models(static_data)
    # from intra_day.configuration.config import config
    # static_data = initializer(config())
    # run_combine_models(static_data)
    from short_term.configuration.config import config
    static_data = initializer(config())
    run_combine_models(static_data)