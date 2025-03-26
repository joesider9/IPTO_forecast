import numpy as np
import pandas as pd


class LoadCompressor:
    def __init__(self, static_data, nwp_data, nwp_metadata):
        self.static_data = static_data
        self.horizon = self.static_data['horizon']
        self.use_data_before_and_after_target = self.static_data['use_data_before_and_after_target']
        self.type = self.static_data['type']
        self.nwp_metadata = nwp_metadata
        self.nwp_data = nwp_data
        self.extra_temp_vars = [var['name'] for var in self.static_data['variables']
                                if var['name'] in {'Temp_max', 'Temp_min', 'Temp_mean'}]

    def load_compressor(self, data):
        shape = data.shape
        data = data.reshape(-1, np.prod(shape[1:]))
        return np.mean(data, axis=1)

    def perform_load_compress(self, i, ax, nwp_data, group_name=None):
        if self.horizon == 0:
            ax_name = '_'.join(ax)
            data = self.load_compressor(nwp_data[:, :, :, :, i])
            data = pd.DataFrame(data, index=self.nwp_metadata['dates'], columns=[ax_name])
            if 'Temperature' in ax_name:
                if 'Temp_max' in self.extra_temp_vars:
                    col = 'Temp_max' if group_name is None else '_'.join(['Temp_max', group_name])
                    data[col] = data.groupby(by=pd.DatetimeIndex(data.index.date))[ax_name].max()
                if 'Temp_min' in self.extra_temp_vars:
                    col = 'Temp_min' if group_name is None else '_'.join(['Temp_min', group_name])
                    data[col] = data.groupby(by=pd.DatetimeIndex(data.index.date))[ax_name].min()
                if 'Temp_mean' in self.extra_temp_vars:
                    col = 'Temp_mean' if group_name is None else '_'.join(['Temp_mean', group_name])
                    data[col] = data.groupby(by=pd.DatetimeIndex(data.index.date))[ax_name].mean()
                data = data.fillna(method='ffill')
            nwp_compressed = data
        else:
            nwp_compressed = pd.DataFrame()
            for hor in range(self.horizon):
                ax_name = '_'.join(ax + [str(hor)])
                data = self.load_compressor(nwp_data[:, hor, :, :, i])
                data = pd.DataFrame(data, index=self.nwp_metadata['dates'], columns=[ax_name])
                if 'Temperature' in ax_name:
                    if 'Temp_max' in self.extra_temp_vars:
                        if hor == 0:
                            col = 'Temp_max' if group_name is None else '_'.join(['Temp_max', group_name])
                        else:
                            col = f'Temp_max_hor_{hor}' if group_name is None \
                                else '_'.join(['Temp_max', group_name, f'hor_{hor}'])
                        data[col] = data.groupby(by=pd.DatetimeIndex(data.index.date))[ax_name].max()
                    if 'Temp_min' in self.extra_temp_vars:
                        if hor == 0:
                            col = 'Temp_min' if group_name is None else '_'.join(['Temp_min', group_name])
                        else:
                            col = f'Temp_min_hor_{hor}' if group_name is None \
                                else '_'.join(['Temp_min', group_name, f'hor_{hor}'])
                        data[col] = data.groupby(by=pd.DatetimeIndex(data.index.date))[ax_name].min()
                    if 'Temp_mean' in self.extra_temp_vars:
                        if 'Temp_mean' in self.extra_temp_vars:
                            if hor == 0:
                                col = 'Temp_mean' if group_name is None else '_'.join(['Temp_mean', group_name])
                            else:
                                col = f'Temp_mean_hor_{hor}' if group_name is None \
                                    else '_'.join(['Temp_mean', group_name, f'hor_{hor}'])
                    data[col] = data.groupby(by=pd.DatetimeIndex(data.index.date))[ax_name].mean()
                    data = data.fillna(method='ffill')
                nwp_compressed = pd.concat([nwp_compressed, data], axis=1)
        return nwp_compressed

    def load_compress(self):
        groups = self.nwp_metadata['groups']
        axis = self.nwp_metadata['axis']
        if len(groups) == 0:
            nwp_compressed = pd.DataFrame()
            nwp_compressed_all = pd.DataFrame()
            for i, ax in enumerate(axis):
                data = self.perform_load_compress(i, ax, self.nwp_data)
                nwp_compressed = pd.concat([nwp_compressed, data], axis=1)
            nwp_compressed_all = nwp_compressed
            nwp_compressed_distributed = np.mean(self.nwp_data, axis=(2, 3))
        else:
            nwp_compressed = dict()
            nwp_compressed_distributed = dict()
            for group in groups:
                group_name = '_'.join(group) if isinstance(group, tuple) else group
                nwp_compressed[group_name] = pd.DataFrame()
                for i, ax in enumerate(axis[group_name]):
                    data = self.perform_load_compress(i, ax, self.nwp_data[group_name], group_name=group_name)
                    nwp_compressed[group_name] = pd.concat([nwp_compressed[group_name], data], axis=1)
                nwp_compressed_distributed[group_name] = np.mean(self.nwp_data[group_name], axis=(2, 3))
            nwp_compressed_all = pd.DataFrame()
            for group_name, data in nwp_compressed.items():
                nwp_compressed_all = pd.concat([nwp_compressed_all, data], axis=1)
            for extra_var in self.extra_temp_vars:
                cols = []
                for group_name in nwp_compressed.keys():
                    cols += [col for col in nwp_compressed_all.columns if extra_var in col and group_name in col]
                nwp_compressed_all[extra_var] = nwp_compressed_all[cols].mean(axis=1)

        return nwp_compressed_all, nwp_compressed, nwp_compressed_distributed
