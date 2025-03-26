import sys
import os
print(os.getcwd())
sys.path.append(os.getcwd())
from eforecast.init.initialize import initializer
from eforecast.nwp_extraction.nwp_extractor import NwpExtractor

from eforecast.dataset_creation.dataset_creator import DatasetCreator
from eforecast.data_preprocessing.data_pipeline import DataPipeline
from eforecast.clustering.clustering_manager import ClusterOrganizer
from eforecast.data_preprocessing.data_split import Splitter
from eforecast.feature_selection.feature_selection_fit import FeatureSelector

import traceback
from eforecast.common_utils.train_utils import send_predictions

RECREATE_DATASETS = False

def nwp_extraction(static_data):
    nwp_extractor = NwpExtractor(static_data, recreate=False, is_online=False, dates=None)
    nwp_extractor.extract()


def create_datasets(static_data):
    dataset = DatasetCreator(static_data, recreate=RECREATE_DATASETS, train=True)
    dataset.create_nwp_dataset()
    dataset.merge_nwp_dataset()
    dataset.compress_nwp_datasets()
    dataset.create_row_datasets()
    if static_data['type'] in {'load', 'fa'}:
        dataset.create_lstm_dataset()
    dataset.create_target()


def preprocess_data(static_data):
    pipeline = DataPipeline(static_data, recreate=RECREATE_DATASETS, online=False, train=True)
    pipeline.fit_pipe()


def cluster_and_split(static_data):
    splitter = Splitter(static_data, is_online=False, train=True)
    splitter.split(refit=RECREATE_DATASETS)
    cluster_organizer = ClusterOrganizer(static_data, is_online=False, train=True, refit=RECREATE_DATASETS)
    cluster_organizer.fit()
    cluster_organizer.create_clusters_and_cvs()


def feature_selection(static_data):
    feature_selector = FeatureSelector(static_data, recreate=RECREATE_DATASETS, online=False, train=True)
    feature_selector.fit()


def run_datasets(static_data):
    try:
        nwp_extraction(static_data)
        create_datasets(static_data)
        preprocess_data(static_data)
        cluster_and_split(static_data)
        feature_selection(static_data)
    except Exception as e:
        tb = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
        print("".join(tb))
        send_predictions(" ".join(tb))
        return
    send_predictions("Done")

if __name__ == '__main__':
    from day_ahead.configuration.config import config
    static_data = initializer(config())
    run_datasets(static_data)
    from intra_day.configuration.config import config
    static_data = initializer(config())
    run_datasets(static_data)
    from short_term.configuration.config import config
    static_data = initializer(config())
    run_datasets(static_data)
