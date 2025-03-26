import numpy as np
import pandas as pd
import re
import requests
import os, datetime

from eforecast.common_utils.date_utils import convert_timezone_dates

from download_load_data_online import DataDownloader


class Reader():
    def __init__(self, data_dir, date=None):
        self.date = date
        self.data_dir = data_dir

    def load_files(self):
        filename = os.path.join(self.data_dir, 'scada_ts.csv')
        if not os.path.exists(filename):
            raise ImportError('SCADA input file does not exists')
        scada = pd.read_csv(filename, index_col=0, header=0, parse_dates=True)
        scada = scada

        filename = os.path.join(self.data_dir, 'res_mv_ts.csv')
        if not os.path.exists(filename):
            raise ImportError('res_mv_ts input file does not exists')
        res_mv_ts = pd.read_csv(filename, index_col=0, header=0, parse_dates=True)
        res_mv_ts = res_mv_ts

        filename = os.path.join(self.data_dir, 'inst_cap.csv')
        if not os.path.exists(filename):
            raise ImportError('Installed RES capacity input file does not exists')
        inst_cap = pd.read_csv(filename, index_col=0, header=0, parse_dates=True)

        filename = os.path.join(self.data_dir, 'ape_net.csv')
        if not os.path.exists(filename):
            raise ImportError('RES Forecast input file does not exists')
        APE_net = pd.read_csv(filename, index_col=0, header=0, parse_dates=True)

        filename = os.path.join(self.data_dir, 'extra_data.csv')
        if not os.path.exists(filename):
            raise ImportError('extra_data input file does not exists')
        extra_data = pd.read_csv(filename, index_col=0, header=0, parse_dates=True)
        return scada, res_mv_ts, inst_cap, APE_net, extra_data

    def insert_data(self, df, row):
        df = df[~df.index.duplicated(keep='last')]
        row = row[~row.index.duplicated(keep='last')]
        ind_df = row.index.intersection(df.index)
        if ind_df.shape[0] > 0:
            df.loc[ind_df] = row.loc[ind_df]
        else:
            dates_diff = row.index.difference(df.index)
            df = pd.concat([df, row.loc[dates_diff]])
            df = df.sort_index()
        return df
    def read_scada(self, scada):
        scada_path = os.path.join(self.data_dir, 'scada')
        file = os.path.join(scada_path, 'scada' + self.date.strftime('%Y%m%d') + '.xls')

        dts = pd.date_range(start=self.date + pd.DateOffset(hours=int(0)), end=self.date + pd.DateOffset(hours=int(23)),
                            freq='H')
        dts = pd.DatetimeIndex(convert_timezone_dates(dts, timezone2='CET'))
        if os.path.exists(file):
            try:
                data = pd.read_excel(file, index_col=[1])
                data = data.loc['ΣΥΝΟΛΙΚΟ ΦΟΡΤΙΟ'].values[1:-1]
                rows = pd.DataFrame(data.astype(float), index=dts, columns=['scada'])
                scada = self.insert_data(scada, rows)
            except:
                try:
                    data = pd.read_excel(file, index_col=[0])
                    data = data.loc[self.date.strftime('%d-%m-%Y')].values[:-1]
                    rows = pd.DataFrame(data.astype(float), index=dts, columns=['scada'])
                    scada = self.insert_data(scada, rows)
                except:
                    try:
                        data = pd.read_excel(file, index_col=[0])
                        data = data[~data.index.duplicated(keep='first')]
                        data = data.loc[self.date.strftime('%d-%m-%Y')].values[:-1]
                        if dts.shape[0] != data.shape[0]:
                            dts = pd.date_range(dts[0], dts[-1], freq='H')
                        rows = pd.DataFrame(data.astype(float), index=dts, columns=['scada'])
                        scada = self.insert_data(scada, rows)
                    except:
                        raise ImportError('Cannot find SCADA')
        else:
            file = os.path.join(scada_path, 'report' + self.date.strftime('%Y%m%d') + '.xls')
            try:
                data = pd.read_excel(file, index_col=[0])
                data = data.loc['ΚΑΘΑΡΟ ΦΟΡΤΙΟ'].iloc[1].values[:-2]
                rows = pd.DataFrame(data.astype(float), index=dts, columns=['scada'])
                scada = self.insert_data(scada, rows)
            except:
                print(file)
                raise ImportError('Cannot find SCADA')
        return scada

    def read_scada_real_time(self, scada):
        try:
            data = requests.get(f"https://www.admie.gr/services/fortiosysthmatos.php?date={self.date.strftime('%Y%m%d')}")
            data = data.json()['items']
            if data[0]['itemname'] == 'SCADA_LOAD':
                data = data[0]
            else:
                data = data[1]
            data = {self.date + pd.DateOffset(hours=int(re.findall(r'\d+', h)[0])): value for h, value in data.items()
                        if 'hour' in h and h != 'hour25' and value > 1000}
            rows = pd.DataFrame().from_dict(data, orient='index')
            rows.columns = scada.columns

            dates, indices = convert_timezone_dates(rows.index, timezone1='Europe/Athens', timezone2='CET',
                                                    return_indices=True)
            rows = rows.iloc[indices]
            rows.index = dates
            scada = self.insert_data(scada, rows)
        except:
            pass
        return scada

    def read_res_mv_non_verified(self, res_mv_ts):

        file = self.data_dir + '/res_mv/res' + self.date.strftime('%Y%m%d') + '.xls'
        dts = pd.date_range(start=self.date + pd.DateOffset(hours=int(0)), end=self.date + pd.DateOffset(hours=int(23)),
                            freq='H')
        dts = pd.DatetimeIndex(convert_timezone_dates(dts, timezone2='CET'))
        if os.path.exists(file):

            try:
                data = pd.read_excel(file, index_col=[1], skiprows=[i for i in range(25, 34)])
                data = data.iloc[:, 1:].astype(float)
                data.columns = ['bio', 'bio_cap', 'hydro', 'hydro_cap', 'sithia', 'sithia_cap', 'pv', 'pv_cap']
                if dts.shape[0] != data.shape[0]:
                    dts = pd.date_range(dts[0], dts[-1], freq='H')
                data.index = dts
                data['pv'] = data['pv'] / 1000000
                res_mv_ts = self.insert_data(res_mv_ts, data)
            except:
                raise ImportError('Cannot find RES_MV_non_verified')
        return res_mv_ts

    def read_load_forecastISP(self, LoadForecastISP):
        if self.date >= pd.to_datetime('01082020', format='%d%m%Y'):

            LoadForecast_path = os.path.join(self.data_dir, 'LoadForecastIPTO/new')
            file = os.path.join(LoadForecast_path, 'load_forecast' + self.date.strftime('%Y%m%d') + '.xlsx')
            dts = pd.date_range(start=self.date + pd.DateOffset(hours=int(1)),
                                end=self.date + pd.DateOffset(hours=int(24)),
                                freq='H')

            if os.path.exists(file):
                try:
                    data = pd.read_excel(file, index_col=[0], engine='openpyxl')
                    data = data.loc['Non-Dispatcheble Load']
                    data = data.dropna()
                    data = data.values[1:-1:2]
                    rows = pd.DataFrame(data.astype(float), index=dts, columns=['LoadForecastISP'])
                    LoadForecastISP = self.insert_data(LoadForecastISP, rows)
                except:
                    try:
                        data = pd.read_excel(file, index_col=[0], engine='openpyxl')
                        data = data.loc['Load Forecast']
                        if data.shape[0] >= 2:
                            data = data.iloc[0]
                        data = data.dropna()

                        if data.values[1::2].shape[0] == 23:
                            data = np.hstack([data.values[1::2], data.values[-1]])
                        else:
                            data = data.values[1::2]
                        rows = pd.DataFrame(data.astype(float), index=dts, columns=['LoadForecastISP'])
                        LoadForecastISP = self.insert_data(LoadForecastISP, rows)
                    except:
                        try:
                            data = pd.read_excel(file, index_col=[0], engine='openpyxl')
                            data = data.loc['Load Forecast']
                            if data.shape[0] >= 2:
                                data = data.iloc[0]
                            data = data.dropna()
                            data = data.values[2::2]
                            rows = pd.DataFrame(data.astype(float), index=dts, columns=['LoadForecastISP'])
                            LoadForecastISP = self.insert_data(LoadForecastISP, rows)
                        except:

                            raise ImportError('Cannot read LoadForcastISP')
            else:
                raise ImportError('File not exists', file)
        return LoadForecastISP

    def merge_data(self, scada, loadfor, res):
        filename = os.path.join(self.data_dir, 'load_ts_online.csv')
        dates = pd.date_range(scada.index[0], scada.index[-1], freq='H')
        data = pd.DataFrame(index=dates, columns=['scada', 'LoadForecast', 'APE_net'])
        data.loc[scada.index, 'scada'] = scada.values.ravel()
        data.loc[loadfor.index, 'LoadForecast'] = loadfor.values.ravel()
        data.loc[res.index, 'APE_net'] = res.values.ravel()
        data.to_csv(filename)

    def save_data(self, scada, res_mv):
        filename = os.path.join(self.data_dir, 'scada_ts.csv')
        scada.to_csv(filename)
        filename = os.path.join(self.data_dir, 'res_mv_ts.csv')
        res_mv.to_csv(filename)



if __name__ == '__main__':
    data_dir = 'D:\Dropbox\current_codes\PycharmProjects\LV_System\data'.replace('\\', '/')
    reader = Reader(data_dir)
    scada, LoadForecastISP, APE_net = reader.load_files()
    dt = [scada.index[-1], LoadForecastISP.index[-1], APE_net.index[-1]]
    t = dt[0]
    for t1 in dt:
        if t1 < t:
            t1 = t
    now = pd.to_datetime(datetime.datetime.now().strftime('%d%m%y'), format='%d%m%y')
    dates = pd.date_range(start=t1, end=now, freq='D')

    for d in dates:
        print(d)
        downloader = DataDownloader([d])
        downloader.download_scada()
        downloader.download_load_forecast()
        reader = Reader(data_dir, d)
        scada = reader.read_scada(scada)
        LoadForecastISP = reader.read_load_forecastISP(LoadForecastISP)
    reader.merge_data(scada, LoadForecastISP, APE_net)
    reader.save_data(scada, LoadForecastISP, APE_net)
