import os

import joblib
import pvlib
import pandas as pd
import numpy as np

from eforecast.dataset_creation.nwp_data.dataset_nwp_creator import DatasetNWPsCreator
from eforecast.dataset_creation.nwp_data.dataset_nwp_organizer import DatasetNWPsOrganizer
from eforecast.dataset_creation.nwp_data.dataset_nwp_compressor import DatasetNWPsCompressor
from eforecast.dataset_creation.data_transformations import DataTransformer
from eforecast.dataset_creation.files_manager import FilesManager

from eforecast.dataset_creation.resampling.data_sampler import DataSampler

from eforecast.common_utils.date_utils import sp_index
from eforecast.common_utils.date_utils import last_year_lags
from eforecast.common_utils.dataset_utils import fix_timeseries_dates


class DatasetCreator:
    def __init__(self, static_data, recreate=False, train=True, is_online=False, dates=None, resampling=False):
        self.static_data = static_data
        self.is_online = is_online
        self.train = train
        self.resampling = resampling
        self.dates = dates
        if self.is_online and self.dates is None:
            raise ValueError('If it runs online you should provide dates')
        self.path_data = self.static_data['path_data']
        self.horizon_type = static_data['horizon_type']
        self.nwp_models = static_data['NWP']
        self.nwp_data_merge = self.static_data['nwp_data_merge']
        self.nwp_data_compress = self.static_data['compress_data']
        self.transformer = DataTransformer(self.static_data, recreate=recreate, is_online=self.is_online,
                                           train=self.train)
        self.files_manager = FilesManager(self.static_data, is_online=self.is_online, train=self.train)
        if recreate or is_online:
            self.files_manager.remove_row_data_files(resampling=self.resampling)
            if self.static_data['type'] in {'load', 'fa'}:
                self.files_manager.remove_lstm_data_files(resampling=self.resampling)
            self.files_manager.remove_nwps(resampling=self.resampling)
            if not is_online:
                self.files_manager.remove_target_files(resampling=self.resampling)
            for merge in self.nwp_data_merge:
                self.files_manager.remove_nwps_merged(merge, resampling=self.resampling)
                for compress in self.nwp_data_compress:
                    self.files_manager.remove_nwps_compressed(merge, compress, resampling=self.resampling)

    def create_nwp_dataset(self):
        if self.nwp_models[0]['model'] is not None:
            nwp_data_creator = DatasetNWPsCreator(self.static_data, self.transformer, dates=self.dates,
                                                  is_online=self.is_online)
            nwp_data = self.files_manager.check_if_exists_nwp_data()
            if nwp_data is None:
                nwp_data = nwp_data_creator.make_dataset()
                self.files_manager.save_nwps(nwp_data)

    def resample_data(self):
        if self.static_data['type'] in {'load', 'fa'}:
            print('No resampling needed for load or fa')
            return
        if self.nwp_models[0]['model'] is not None:
            nwp_data = self.files_manager.check_if_exists_nwp_data()
            nwp_data_sampled = self.files_manager.check_if_exists_nwp_data(resampling=True)
            if nwp_data_sampled is None:
                data_sampler = DataSampler(self.static_data)
                nwp_data_sampled = data_sampler.sampling(nwp_data, dataset_type='nwp')
                self.files_manager.save_nwps(nwp_data_sampled, resampling=True)
            self.merge_nwp_dataset()
            self.compress_nwp_datasets()
        data_row = self.files_manager.check_if_exists_row_data()
        data_row_sampled = self.files_manager.check_if_exists_row_data(resampling=True)
        if data_row_sampled is None:
            data_sampler = DataSampler(self.static_data)
            if self.static_data['type'] in {'load', 'fa'}:
                data_lstm_dict = self.files_manager.check_if_exists_lstm_data()
                data_row_sampled,  X_lstm_dict = data_sampler.sampling(data_row, dataset_type='row',
                                                                       X_lstm_dict=data_lstm_dict)
                self.files_manager.save_lstm_data(X_lstm_dict['data'], X_lstm_dict['metadata'], resampling=True)
            else:
                data_row_sampled = data_sampler.sampling(data_row, dataset_type='row')
            self.files_manager.save_row_data(data_row_sampled, resampling=True)
        target = self.files_manager.check_if_exists_target()
        target_sampled = self.files_manager.check_if_exists_target(resampling=True)
        if target_sampled is None:
            data_sampler = DataSampler(self.static_data)
            target_sampled = data_sampler.sampling(target, dataset_type='target')
            self.files_manager.save_target(target_sampled, resampling=True)

    def merge_nwp_dataset(self):
        if self.nwp_models[0]['model'] is not None:
            nwp_data = self.files_manager.check_if_exists_nwp_data(resampling=self.resampling)
            if nwp_data is None:
                raise ImportError('Merge NWP dataset failed due to nwp_data is None, data_nwp_creator seems to fail')
            for merge in self.nwp_data_merge:
                data_merged = self.files_manager.check_if_exists_nwps_merged(merge, get_all=True,
                                                                             resampling=self.resampling)
                nwp_data_merged, nwp_metadata = data_merged['data'], data_merged['nwp_metadata']
                if nwp_data_merged is None:
                    nwp_data_organizer = DatasetNWPsOrganizer(self.static_data, nwp_data)
                    nwp_data_merged, nwp_metadata = nwp_data_organizer.merge(merge)
                    self.files_manager.save_nwps_merged(merge, nwp_data_merged, nwp_metadata, resampling=self.resampling)

    def compress_nwp_datasets(self):
        if self.nwp_models[0]['model'] is not None:
            print(f"Dataset NWP compressing started for project {self.static_data['_id']}")
            for merge in self.nwp_data_merge:
                data_merged = self.files_manager.check_if_exists_nwps_merged(merge, get_all=True,
                                                                             resampling=self.resampling)
                nwp_data_merged, nwp_metadata = data_merged['data'], data_merged['nwp_metadata']
                for compress in self.nwp_data_compress:
                    data_compressed = self.files_manager.check_if_exists_nwps_compressed(merge, compress, get_all=True,
                                                                                         resampling=self.resampling)
                    nwp_compressed_all, nwp_compressed, nwp_compressed_distributed, metadata = \
                        data_compressed['data_compressed_all'], \
                        data_compressed['data_compressed'], \
                        data_compressed['data_compressed_distributed'], \
                        data_compressed['nwp_metadata']
                    if nwp_compressed is None:
                        nwp_data_compressor = DatasetNWPsCompressor(self.static_data, nwp_data_merged, nwp_metadata,
                                                                    compress)
                        nwp_compressed_all, nwp_compressed, nwp_compressed_distributed = nwp_data_compressor.compress()
                        self.files_manager.save_nwps_compressed(merge, compress, nwp_compressed_all, nwp_compressed,
                                                                nwp_compressed_distributed, nwp_metadata,
                                                                resampling=self.resampling)

    def concat_lagged_data(self, data, var_name, var_data, use_diff_between_lags=False):
        data = data.sort_index()
        data = fix_timeseries_dates(data, freq=self.static_data['ts_resolution'])
        dates = pd.date_range(data.index[0], data.index[-1],
                              freq=self.static_data['ts_resolution'])
        data_temp = pd.DataFrame(index=dates, columns=data.columns)
        data_temp.loc[data.index] = data
        for lag in var_data['lags']:
            if var_data['source'] == 'target':
                col = self.static_data['project_name']
            elif var_data['source'] in {'nwp_dataset', 'index'}:
                col = var_name
            elif var_data['source'].endswith('.csv'):
                if var_name in data.columns:
                    col = var_name
                else:
                    col = data.columns[0]
            else:
                col = var_data['source']
            if isinstance(lag, int):
                if isinstance(col, list):
                    for c in col:
                        data_temp[f'{c}_lag_{lag}'] = data_temp[c].shift(-lag)
                else:
                    data_temp[f'{var_name}_lag_{lag}'] = data_temp[col].shift(-lag)
            else:
                lylags = pd.DataFrame()
                for d in data_temp.index:
                    try:
                        lags = [d - pd.DateOffset(hours=i) for i in last_year_lags(d, self.static_data['country'])]
                        loads = data_temp[col].iloc[lags]
                        loads = pd.DataFrame(loads, index=[d], columns=[f'lylags_{i}' for i in range(8)])
                        lylags = pd.concat([lylags, loads])
                    except:
                        continue
                data_temp = pd.concat([data_temp, lylags], axis=1)
        if use_diff_between_lags:
            for lag1 in var_data['lags']:
                for lag2 in var_data['lags']:
                    if lag1 > lag2:
                        data_temp[f'Diff_{var_name}_lag{lag1}_lag{lag2}'] = \
                            data_temp[f'{var_name}_lag_{lag1}'] - data_temp[f'{var_name}_lag_{lag2}']
                        data_temp[f'Diff2_{var_name}_lag{lag1}_lag{lag2}'] = np.square(
                            data_temp[f'{var_name}_lag_{lag1}'] - data_temp[f'{var_name}_lag_{lag2}'])
        return data_temp

    def create_autoregressive_dataset(self, remove_nans=True, lag_lstm=0):
        variables = dict([(var_data['name'], var_data) for var_data in self.static_data['variables']
                          if var_data['type'] == 'timeseries' and var_data['source'] == 'target'])
        if len(variables) == 0:
            return None
        data_arma = pd.DataFrame()
        for var_name, var_data in variables.items():
            data = self.static_data['data'].copy(deep=True)
            if self.is_online:
                max_lag = min(var_data['lags'] + [lag_lstm])
                dates_pred = pd.date_range(self.dates[0] + pd.DateOffset(hours=max_lag),
                                           self.dates[-1] + pd.DateOffset(hours=47), freq='H')
                data_nan = pd.DataFrame(index=pd.date_range(data.index[-1] + pd.DateOffset(hours=1),
                                                            dates_pred[-1], freq='H'),
                                        columns=data.columns)
                data = pd.concat([data, data_nan])
            data = self.concat_lagged_data(data, var_name, var_data,
                                           use_diff_between_lags=self.static_data['use_diff_between_lags'])
            data = data.drop(columns=[self.static_data['project_name']])
            if remove_nans:
                data = data.dropna(axis='index')
            data_arma = pd.concat([data_arma, data], axis=1)
        if self.is_online:
            dates_online = pd.date_range(self.dates[0] + pd.DateOffset(hours=lag_lstm),
                                         self.dates[-1] +
                                         pd.DateOffset(hours=47), freq='H').intersection(data_arma.index)

            data_arma = data_arma.loc[dates_online]
        return data_arma

    def create_calendar_dataset(self, remove_nans=True, lag_lstm=0):
        # sin_transformer = lambda x: np.sin(x / period * 2 * np.pi)
        variables_index = dict([(var_data['name'], var_data) for var_data in self.static_data['variables']
                                if var_data['type'] == 'calendar' and var_data['source'] == 'index'])
        if self.static_data['data'] is not None:
            data = self.static_data['data'].copy(deep=True)
        else:
            data = None
        if not self.is_online:
            index = data.index
        else:
            if data is not None:
                index = pd.date_range(data.index[0], self.dates[-1] + pd.DateOffset(hours=47), freq='H')
            else:
                max_lag = [lag_lstm] + [min(var_data['lags']) for var_data in self.static_data['variables']
                           if var_data['type'] == 'calendar' and var_data['source'] == 'index']
                if len(max_lag) > 0:
                    max_lag = min(max_lag)
                index = pd.date_range(self.dates[0] + pd.DateOffset(hours=max_lag), self.dates[-1] + pd.DateOffset(hours=47), freq='H')
        for var_name, var_data in variables_index.items():
            if var_name == 'hour':
                values = index.hour.values
                period = 24
            elif var_name == 'month':
                values = index.month.values
                period = 12
            elif var_name == 'dayweek':
                values = index.dayofweek.values
                period = 7
            elif var_name == 'sp_index':
                values = [sp_index(d, country=self.static_data['country']) for d in index]
            else:
                raise ValueError(f'Unknown variable {var_name} for index and calendar')

            values = pd.DataFrame(values, columns=[var_name], index=index)
            if len(var_data['lags']) > 1:
                lag_values = self.concat_lagged_data(values, var_name, var_data)
                cols = [var_name] + [col for col in data.columns if 'lag_0' in col]
                lag_values = lag_values.drop(columns=cols)
                values = pd.concat([values, lag_values], axis=1)
            data = pd.concat([data, values], axis=1)
        variables_astral = dict([(var_data['name'], var_data) for var_data in self.static_data['variables']
                                 if var_data['type'] == 'calendar' and var_data['source'] == 'astral'])
        if len(variables_astral) == 0 and len(variables_index) == 0:
            return None
        for var_name, var_data in variables_astral.items():
            solpos = pvlib.solarposition.get_solarposition(index, self.static_data['coord'][0]
                                                           , self.static_data['coord'][1])
            if var_name not in {'azimuth', 'zenith'}:
                raise ValueError(f'Unknown variable {var_name} for astral and calendar. '
                                 f'Accepted values are azimuth, zenith')
            data = pd.concat([data, solpos[var_name].to_frame()], axis=1)
        if self.static_data['project_name'] in data.columns:
            data = data.drop(columns=[self.static_data['project_name']])
        if remove_nans:
            data = data.dropna(axis='index')
        return data

    def create_extra_ts_datasets(self, remove_nans=True, exclude_nwps=False, lag_lstm=0):
        if exclude_nwps:
            variables_extra = dict([(var_data['name'], var_data) for var_data in self.static_data['variables']
                                    if var_data['type'] == 'timeseries' and var_data['source'] != 'target'
                                    and var_data['source'] != 'nwp_dataset'])
        else:
            variables_extra = dict([(var_data['name'], var_data) for var_data in self.static_data['variables']
                                    if var_data['type'] == 'timeseries' and var_data['source'] != 'target'])
        if len(variables_extra) == 0:
            return None
        data_extra = pd.DataFrame()
        for var_name, var_data in variables_extra.items():
            name = var_name
            if var_data['source'].endswith('.csv'):
                if os.path.exists(var_data['source']):
                    data = pd.read_csv(var_data['source'], index_col=0, header=0, parse_dates=True)
                    if name in data.columns:
                        data = data[name].to_frame()
                else:
                    raise ImportError(f"{var_data['source']} does not exists")
            elif var_data['source'] == 'nwp_dataset':
                data_compressed = self.files_manager.check_if_exists_nwps_compressed(self.nwp_data_merge[0], 'load', get_all=True,
                                                                                     resampling=self.resampling)
                nwp_compressed_all, nwp_compressed, nwp_compressed_distributed, metadata = \
                    data_compressed['data_compressed_all'], \
                        data_compressed['data_compressed'], \
                        data_compressed['data_compressed_distributed'], \
                        data_compressed['nwp_metadata']

                if nwp_compressed is None:
                    raise ImportError('Can not find nwp data for load to get their lags')
                if self.static_data['project_name'] == 'kythnos_ems' and \
                        self.static_data['type'] == 'load' and\
                        self.static_data['horizon_type'] == 'multi-output':

                    name = var_name if var_name not in {'Temp'} \
                        else [n for n in nwp_compressed_all.columns if 'kythnos_ems_Temperature_0_ecmwf_0' in n]
                else:
                    name = var_name if var_name not in {'Temp'} \
                        else [n for n in nwp_compressed_all.columns if 'Temperature' in n]
                data = nwp_compressed_all[name]
                if not isinstance(data, pd.DataFrame):
                    data = data.to_frame()
            else:
                data = pd.read_csv(self.static_data['filename'], index_col=0, header=0, parse_dates=True)
                if var_data['source'] not in data.columns:
                    raise ValueError(f"{var_data['source']} does not exists in main file columns. "
                                     f"Filename is {self.static_data['filename']}")
                data = data[var_data['source']].to_frame()
            if self.is_online:
                max_lag = min(var_data['lags'] + [lag_lstm])
                dates_pred = pd.date_range(self.dates[0] + pd.DateOffset(hours=max_lag), self.dates[-1] + pd.DateOffset(hours=47), freq='H')
                data_nan = pd.DataFrame(index=pd.date_range(data.index[-1] + pd.DateOffset(hours=1),
                                                            dates_pred[-1], freq='H'),
                                        columns=data.columns)
                data = pd.concat([data, data_nan])
            data = self.concat_lagged_data(data, name, var_data)

            if var_data['source'] == 'nwp_dataset':
                if isinstance(name, list):
                    cols = name + [col for col in data.columns if 'lag_0' in col]
                else:
                    cols = [name] + [col for col in data.columns if 'lag_0' in col]
            elif var_data['source'].endswith('.csv'):
                if name in data.columns:
                    cols = [name]
                else:
                    cols = [data.columns[0]]
            else:
                cols = [var_data['source']]
            data = data.drop(columns=cols)
            if remove_nans:
                data = data.dropna(axis='index')
            else:
                data.iloc[data.isna().values] = 0
            data_extra = pd.concat([data_extra, data], axis=1)
        data_extra = data_extra.dropna(axis='index')
        if self.is_online:
            dates_online = pd.date_range(self.dates[0] + pd.DateOffset(hours=lag_lstm),
                                         self.dates[-1] + pd.DateOffset(hours=47), freq='H')
            dates_online = dates_online.intersection(data_extra.index)
            data_extra = data_extra.loc[dates_online]
        return data_extra

    def create_row_datasets(self):
        data_row = self.files_manager.check_if_exists_row_data()
        if data_row is None:
            data_arma = self.create_autoregressive_dataset()
            data_extra = self.create_extra_ts_datasets()
            data_calendar = self.create_calendar_dataset()
            data_row = pd.DataFrame()
            for data in [data_arma, data_extra, data_calendar]:
                if data is not None:
                    data_row = pd.concat([data_row, data], axis=1)
            data_row = data_row.dropna(axis='index')
            self.files_manager.save_row_data(data_row)

    def create_target(self):
        data = self.files_manager.check_if_exists_target()
        if data is None:
            data = self.static_data['data'].copy(deep=True)
            if self.static_data['horizon_type'] == 'multi-output':
                for hor in range(self.static_data['horizon']):
                    data[f'hour_ahead_{hor}'] = data[self.static_data['project_name']].shift(-hor)
                data = data.dropna(axis='index')
                data = data.drop(columns=[self.static_data['project_name']])
            else:
                data.columns = ['target']
            self.files_manager.save_target(data)

    @staticmethod
    def get_lags_from_df(df, lags, col):
        df = fix_timeseries_dates(df)
        df1 = pd.DataFrame(index=df.index)
        for lag in lags:
            df1[f'{col}_lag_{lag}'] = df.shift(-lag)
        return df1

    def create_lstm_dataset(self):
        data_lstm_dict = self.files_manager.check_if_exists_lstm_data()
        data_lstm = data_lstm_dict['data']
        metadata = data_lstm_dict['metadata']
        if self.is_online:
            path_lstm = os.path.join(self.path_data, 'dataset_lstm_data.pickle')
            data_train = joblib.load(path_lstm)
            variables_train = data_train['metadata']['variables']
        if data_lstm is None:
            metadata = dict()
            metadata['groups'] = []
            lags = []
            for var_data in self.static_data['variables']:
                if var_data['name'] == 'load':
                    lags = var_data['lags']
            if len(lags) == 0:
                lags = [-i for i in range(1, self.static_data['global_lags'])]
            if len(lags) <= 2:
                raise ValueError('Load problem with very few lags')

            if self.horizon_type == 'day-ahead':
                lags = [0, -24] + lags
            elif self.horizon_type == 'intra-ahead':
                lags = [0] + lags
            metadata['lags'] = lags
            metadata['variables'] = []
            data_lagged = []

            data_ts = self.create_autoregressive_dataset(remove_nans=False, lag_lstm=min(lags))
            if data_ts is not None:
                lags_arma = [f'load_lag_{lag}' for lag in lags]
                for lag in lags:
                    if f'load_lag_{lag}' not in data_ts.columns:
                        data_ts[f'load_lag_{lag}'] = 0
                data_ts = data_ts[lags_arma]

            data_extra = self.create_extra_ts_datasets(remove_nans=False, exclude_nwps=True, lag_lstm=min(lags))
            if data_ts is None and data_extra is None:
                raise ValueError('Cannot find a variable named load')
            if data_ts is None and data_extra is not None and self.static_data['global_lags'] is None:
                lags_load_extra = sorted(list(set([col for col in data_extra.columns if 'load' in col and 'estimation' not in col])))
                if len(lags_load_extra) > 0:
                    data_ts = data_extra[lags_load_extra]
                    data_extra = data_extra.drop(columns=lags_load_extra)
                    lags_arma = [f'load_lag_{lag}' for lag in lags]
                    for lag in lags:
                        if f'load_lag_{lag}' not in data_ts.columns:
                            data_ts[f'load_lag_{lag}'] = 0
                    data_ts = data_ts[lags_arma]
            if data_ts is not None:
                data_lagged.append(data_ts)
                dates = data_ts.index
                metadata['variables'].append('load')
            else:
                dates = None

            if data_extra is not None:
                if dates is None:
                    dates = data_extra.index
                if data_extra.shape[1] > 0:
                    cols_extra = sorted(list(set([col for col in data_extra.columns if 'lag' not in col])))
                    if len(cols_extra) == 0:
                        cols_extra = sorted(list(set([col for col in data_extra.columns if 'lag_0' in col])))
                    data_extra = data_extra[cols_extra]
                    for col in data_extra.columns:
                        if 'rated' not in col:
                            data_lagged.append(self.get_lags_from_df(data_extra[col].to_frame(), lags, col))
                            metadata['variables'].append(col)

            data_calendar = self.create_calendar_dataset(remove_nans=False, lag_lstm=min(lags))
            if data_calendar is not None:
                if dates is None:
                    dates = data_extra.index
                cols_calendar = sorted(col for col in data_calendar.columns if 'lag' not in col)
                if len(cols_calendar) == 0:
                    cols_calendar = sorted(list(set([col for col in data_calendar.columns if 'lag_0' in col])))
                data_calendar = data_calendar[cols_calendar]
                if data_calendar.shape[1] > 0:
                    for col in data_calendar.columns:
                        data_lagged.append(self.get_lags_from_df(data_calendar[col].to_frame(), lags, col))
                        metadata['variables'].append(col)
            data_type = self.static_data['clustering']['data_type']
            data_compressed = self.files_manager.check_if_exists_nwps_compressed(data_type['merge'],
                                                                                 data_type['compress'], get_all=True,
                                                                                 resampling=self.resampling)
            nwp_compressed_all, nwp_compressed, nwp_compressed_distributed, nwp_metadata = \
                data_compressed['data_compressed_all'], \
                data_compressed['data_compressed'], \
                data_compressed['data_compressed_distributed'], \
                data_compressed['nwp_metadata']

            if nwp_compressed_all is not None:
                cols_nwp = [col for col in nwp_compressed_all.columns if 'lag' not in col]
                nwp_compressed_all = nwp_compressed_all[cols_nwp]
                if nwp_compressed_all.shape[1] > 0:
                    for col in nwp_compressed_all.columns:
                        data_lagged.append(self.get_lags_from_df(nwp_compressed_all[col].to_frame(), lags, col))
                        metadata['variables'].append(col)
            if self.is_online:
                order_vars = []
                for v_train in variables_train:
                    for ind, v in enumerate(metadata['variables']):
                        if v.split('_lag')[0] == v_train:
                            order_vars.append(ind)
                data_lagged = [data_lagged[o] for o in order_vars]
            for data in data_lagged:
                data = data.dropna(axis='index')
                dates = dates.intersection(data.index)

            metadata['dates'] = dates
            data_lstm = np.array([])
            for data in data_lagged:
                data_np = np.expand_dims(data.loc[dates].values, axis=-1)
                if data_lstm.shape[0] == 0:
                    data_lstm = data_np.astype('float')
                else:
                    data_lstm = np.concatenate([data_lstm, data_np.astype('float')], axis=-1)
            self.files_manager.save_lstm_data(data_lstm, metadata)
