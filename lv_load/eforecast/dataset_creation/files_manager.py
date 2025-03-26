import os
import joblib
import numpy as np
import pandas as pd

from eforecast.common_utils.dataset_utils import get_slice


class FilesManager:
    def __init__(self, static_data, is_online=False, train=True):
        self.static_data = static_data
        self.is_online = is_online
        self.train = train
        self.path_data = self.static_data['path_data']

    def file_target(self, resampling=False):
        if self.is_online:
            raise ValueError('Cannot create target files online')
        else:
            if resampling:
                dataset_file = 'dataset_target_data_resampling.csv'
            else:
                dataset_file = 'dataset_target_data.csv'
        return os.path.join(self.path_data, dataset_file)

    def check_if_exists_target(self, resampling=False):
        file = self.file_target(resampling=resampling)
        if not os.path.exists(file):
            return None
        else:
            data = pd.read_csv(file, index_col=0, header=0, parse_dates=True)
            if not self.is_online:
                data, _ = self.split(data, data.index)
            return data

    def save_target(self, row_data, resampling=False):
        file = self.file_target(resampling=resampling)
        row_data.to_csv(file)

    def remove_target_files(self, resampling=False):
        file = self.file_target(resampling=resampling)
        if os.path.exists(file):
            os.remove(file)

    def file_row_data(self, resampling=False):
        if self.is_online:
            dataset_file = 'dataset_row_data_online.csv'
        else:
            if resampling:
                dataset_file = 'dataset_row_data_resampling.csv'
            else:
                dataset_file = 'dataset_row_data.csv'
        return os.path.join(self.path_data, dataset_file)

    def check_if_exists_row_data(self, resampling=False):
        file = self.file_row_data(resampling=resampling)
        if not os.path.exists(file):
            return None
        else:
            data = pd.read_csv(file, index_col=0, header=0, parse_dates=True)
            if not self.is_online:
                data, _ = self.split(data, data.index)
            return data

    def save_row_data(self, row_data, resampling=False):
        file = self.file_row_data(resampling=resampling)
        row_data.to_csv(file)

    def remove_row_data_files(self, resampling=False):
        file = self.file_row_data(resampling=resampling)
        if os.path.exists(file):
            os.remove(file)

    def file_lstm_data(self, resampling=False):
        if self.is_online:
            dataset_file = 'dataset_lstm_data_online.pickle'
        else:
            if resampling:
                dataset_file = 'dataset_lstm_data_resampling.pickle'
            else:
                dataset_file = 'dataset_lstm_data.pickle'
        return os.path.join(self.path_data, dataset_file)

    def check_if_exists_lstm_data(self, resampling=False):
        file = self.file_lstm_data(resampling=resampling)
        if not os.path.exists(file):
            return {'data': None, 'metadata': None}
        else:
            lstm_data_dict = joblib.load(file)
            lstm_data, metadata = lstm_data_dict['data'], lstm_data_dict['metadata']
            if not self.is_online:
                lstm_data, dates = self.split(lstm_data, metadata['dates'])
                metadata['dates'] = dates
            return {'data': lstm_data, 'metadata': metadata}

    def save_lstm_data(self, lstm_data, metadata, resampling=False):
        file = self.file_lstm_data(resampling=resampling)
        joblib.dump({'data': lstm_data, 'metadata': metadata}, file)

    def remove_lstm_data_files(self, resampling=False):
        file = self.file_lstm_data(resampling=resampling)
        if os.path.exists(file):
            os.remove(file)

    def file_nwps(self, resampling=False):
        if self.is_online:
            dataset_file = 'dataset_nwps_online.pickle'
        else:
            if resampling:
                dataset_file = 'dataset_nwps_resampling.pickle'
            else:
                dataset_file = 'dataset_nwps.pickle'
        return os.path.join(self.path_data, dataset_file)

    def check_if_exists_nwp_data(self, resampling=False):
        file = self.file_nwps(resampling=resampling)
        if not os.path.exists(file):
            return None
        else:
            return joblib.load(file)

    def save_nwps(self, nwp_data, resampling=False):
        file = self.file_nwps(resampling=resampling)
        joblib.dump(nwp_data, file)

    def remove_nwps(self, resampling=False):
        file = self.file_nwps(resampling=resampling)
        if os.path.exists(file):
            os.remove(file)

    def file_nwps_merged(self, merge, resampling=False):
        if self.is_online:
            dataset_file = f'dataset_nwps_{merge}_online.pickle'
        else:
            if resampling:
                dataset_file = f'dataset_nwps_{merge}_resampling.pickle'
            else:
                dataset_file = f'dataset_nwps_{merge}.pickle'
        return os.path.join(self.path_data, dataset_file)

    def check_if_exists_nwps_merged(self, merge, get_all=False, resampling=False):
        file = self.file_nwps_merged(merge, resampling=resampling)
        if not os.path.exists(file):
            return {'data': None, 'nwp_metadata': None}
        else:
            data_merged = joblib.load(file)
            nwp_data_merged, nwp_metadata = data_merged['data'], data_merged['nwp_metadata']
            if not self.is_online and not get_all:
                nwp_data_merged, dates = self.split(nwp_data_merged, nwp_metadata['dates'])
                nwp_metadata['dates'] = dates
            return {'data': nwp_data_merged, 'nwp_metadata': nwp_metadata}

    def save_nwps_merged(self, merge, nwp_data_merged, nwp_metadata, resampling=False):
        file = self.file_nwps_merged(merge, resampling=resampling)
        joblib.dump({'data': nwp_data_merged, 'nwp_metadata': nwp_metadata}, file)

    def remove_nwps_merged(self, merge, resampling=False):
        file = self.file_nwps_merged(merge, resampling=resampling)
        if os.path.exists(file):
            os.remove(file)

    def file_nwps_compressed(self, merge, compress, resampling=False):
        if self.is_online:
            dataset_file = f'dataset_nwps_{merge}_{compress}_online.pickle'
        else:
            if resampling:
                dataset_file = f'dataset_nwps_{merge}_{compress}_resampling.pickle'
            else:
                dataset_file = f'dataset_nwps_{merge}_{compress}.pickle'
        return os.path.join(self.path_data, dataset_file)

    def check_if_exists_nwps_compressed(self, merge, compress, get_all=False, resampling=False):
        file = self.file_nwps_compressed(merge, compress, resampling=resampling)
        if not os.path.exists(file):
            return {'data_compressed_all': None, 'data_compressed': None, 'data_compressed_distributed': None,
                    'nwp_metadata': None}
        else:
            data_compressed = joblib.load(file)
            nwp_compressed_all, nwp_compressed, nwp_compressed_distributed, metadata = \
                data_compressed['data_compressed_all'], \
                data_compressed['data_compressed'], \
                data_compressed['data_compressed_distributed'], \
                data_compressed['nwp_metadata']
            if not self.is_online and not get_all:
                nwp_compressed_all, _ = self.split(nwp_compressed_all, metadata['dates'])
                nwp_compressed, _ = self.split(nwp_compressed, metadata['dates'])
                nwp_compressed_distributed, dates = self.split(nwp_compressed_distributed, metadata['dates'])
                metadata['dates'] = dates
            return {'data_compressed_all': nwp_compressed_all, 'data_compressed': nwp_compressed,
                    'data_compressed_distributed': nwp_compressed_distributed, 'nwp_metadata': metadata}

    def save_nwps_compressed(self, merge, compress, nwp_compressed_all, nwp_compressed, nwp_compressed_distributed,
                             metadata, resampling=False):
        file = self.file_nwps_compressed(merge, compress, resampling=resampling)
        joblib.dump({'data_compressed_all': nwp_compressed_all, 'data_compressed': nwp_compressed,
                     'data_compressed_distributed': nwp_compressed_distributed, 'nwp_metadata': metadata}, file)

    def remove_nwps_compressed(self, merge, compress, resampling=False):
        file = self.file_nwps_compressed(merge, compress, resampling=resampling)
        if os.path.exists(file):
            os.remove(file)

    def split(self, data, dates):
        if self.train:
            ind = np.where(dates <= self.static_data['Evaluation_start'])[0]
        else:
            ind = np.where(dates > self.static_data['Evaluation_start'])[0]
        return get_slice(data, ind), dates[ind]

    def file_cv_data(self, fuzzy=False):
        dataset_file = 'cv_mask.pickle'if not fuzzy else 'cv_mask_fuzzy.pickle'
        return os.path.join(self.path_data, dataset_file)

    def check_if_exists_cv_data(self, fuzzy=False):
        file = self.file_cv_data(fuzzy=fuzzy)
        if not os.path.exists(file):
            return None
        else:
            data = joblib.load(file)
            return data

    def save_cv_data(self, cv_mask, fuzzy=False):
        file = self.file_cv_data(fuzzy=fuzzy)
        joblib.dump(cv_mask, file)

    def remove_cv_data_files(self, fuzzy=False):
        file = self.file_cv_data(fuzzy=fuzzy)
        if os.path.exists(file):
            os.remove(file)
