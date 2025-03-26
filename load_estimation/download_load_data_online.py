import pandas as pd
from requests import get
from tqdm import tqdm
from joblib import Parallel
from joblib import delayed
import time, os

class DataDownloader():
    def __init__(self, dates, path_data):
        self.dates = dates
        self.path_data =path_data

    def download(self, urls):
        file_name, url = urls
        # open in binary mode
        try:
            # get request
            response = get(url)
            # write to file
            if response.ok:
                with open(file_name, "wb") as file:


                    file.write(response.content)
            else:
                print(file_name)
        except:
            print(file_name)

    def download_scada(self):
        for d in self.dates:
            for id in range(1,34):
                response = self.scada_download(d, id)
                if response:
                    print(f'{d} downloaded')
                    break

    def download_res(self):
        for d in tqdm(self.dates):
            self.download_res_mv(d)
        # Parallel(n_jobs=4)(delayed(self.download_res_mv)(d) for d in tqdm(self.dates))

    def download_res_mv(self, d):
        for id in range(1, 4):
            response = self.res_mv_download(d, id)
            if response:
                print(f'{d} downloaded')
                break
            else:
                print(f'Cannot download id {id} of date {d}')

    def download_load_forecast(self):
        for d in self.dates:
            for id in range(6, 1, -1):
                response = self.load_forecast_download(d, id)
                if response:
                    break

    def res_mv_download(self, d, id):
        # for SCADA
        file_name = self.path_data + '/res_mv/res' + d.strftime(
            '%Y%m%d') + '.xls'

        d = d + pd.DateOffset(days=1)
        months = [d.month, 4, 3, (d - pd.DateOffset(months=1)).month, (d + pd.DateOffset(months=1)).month]
        years = [d.year, 2020, (d - pd.DateOffset(months=12)).year, (d + pd.DateOffset(months=12)).year]
        if not os.path.exists(file_name):
            date_str =  d.strftime('%Y-%m-%d')
            url = f'https://www.admie.gr/getOperationMarketFile?dateStart={date_str}&dateEnd={date_str}&FileCategory=RESMV'
            response_file = get(url)

            info = response_file.json() if response_file.ok else ''
            info = info[0] if len(info) > 0 else ''
            file_path = info['file_path'] if info != '' else ''

            for year in years:
                for month in months:
                    urls = [file_path,
                            'https://www.admie.gr/sites/default/files/attached-files/type-file/' + str(
                                year) + '/' + str(
                                month).zfill(2) + '/' + d.strftime('%Y%m%d') + '_RESMV_0' + str(
                                id) + '.xlsx',
                            'https://www.admie.gr/sites/default/files/attached-files/type-file/' + str(
                                year) + '/' + str(
                                month).zfill(2) + '/' + d.strftime('%Y%m%d') + '_RESMV_0' + str(
                                id) + '.xls',

                            ]

                    for url in urls:
                        try:
                            # get request
                            response = get(url)
                        except:
                            print(url)
                            continue

                        if response.ok:
                            with open(file_name, "wb") as file:
                                file.write(response.content)
                                file.close()
                            break
                    if response.ok:
                        break
                if response.ok:
                    break
            if response.ok:
                return response.ok
            else:
                print('Cannot download ', file_name)

        else:
            return True


    def scada_download(self, d, id):
        # for SCADA
        file_name = self.path_data + '/scada/scada' + d.strftime('%Y%m%d') + '.xls'
        months = [d.month, 4, 3, (d - pd.DateOffset(months=1)).month, (d + pd.DateOffset(months=1)).month]
        years = [d.year, 2020, (d - pd.DateOffset(months=12)).year, (d + pd.DateOffset(months=12)).year]
        if not os.path.exists(file_name):
            date_str =  d.strftime('%Y-%m-%d')
            url = f'https://www.admie.gr/getOperationMarketFile?dateStart={date_str}&dateEnd={date_str}' \
                  f'&FileCategory=RealTimeSCADASystemLoad'
            response_file = get(url)

            info = response_file.json() if response_file.ok else ''
            info = info[0] if len(info) > 0 else ''
            file_path = info['file_path'] if info != '' else ''
            for year in years:
                for month in months:
                    urls = [file_path,
                            'https://www.admie.gr/sites/default/files/attached-files/type-file/' + str(
                                year) + '/' + str(
                                month).zfill(2) + '/' + d.strftime('%Y%m%d') + '_RealTimeSCADASystemLoad_01.xls',
                            'https://www.admie.gr/sites/default/files/attached-files/type-file/' + str(
                                year) + '/' + str(
                                month).zfill(2) + '/' + d.strftime('%Y%m%d') + '_SystemRealizationSCADA_0' + str(
                                id) + '.xls',
                            ]
                    for url in urls:
                        try:
                            # get request
                            response = get(url)
                        except:
                            print(url)
                            continue
                        if response.ok:
                            with open(file_name, "wb") as file:
                                file.write(response.content)
                                file.close()
                            break
                    if response.ok:
                        break
                if response.ok:
                    break
            if response.ok:
                return response.ok
            else:
                print('Cannot download ', file_name)

        else:
            return True

    def weekly_load_forecast_download_new_files(self, d, id):
        file_name = self.path_data + '/LoadForecastIPTO/week_load_forecast' + d.strftime(
            '%Y%m%d') + '.xlsx'
        months = [d.month, (d - pd.DateOffset(months=1)).month, (d + pd.DateOffset(months=1)).month]
        years = [d.year, (d - pd.DateOffset(months=12)).year, (d + pd.DateOffset(months=12)).year]
        if not os.path.exists(file_name):
            for year in years:
                for month in months:
                    urls = ['https://www.admie.gr/sites/default/files/attached-files/type-file/' + str(d.year) + '/' + str(
                        month).zfill(2) + '/' + d.strftime('%Y%m%d') + '_ISPWeekAheadLoadForecast_01.xlsx',
                            'https://www.admie.gr/sites/default/files/attached-files/type-file/' + str(d.year) + '/' + str(
                                month).zfill(2) + '/' + (d - pd.DateOffset(days=1)).strftime('%Y%m%d') + '_ISPWeekAheadLoadForecast_01.xlsx'
                            ]

                    for url in urls:
                        try:
                            # get request
                            response = get(url)
                        except:
                            print(file_name)
                            pass

                        if response.ok:
                            with open(file_name, "wb") as file:
                                file.write(response.content)
                                file.close()
                            break
                    if response.ok:
                        break
                if response.ok:
                    break
            if response.ok:
                return response.ok
            else:
                print('Cannot download ', file_name)

        else:
            return True

    def load_forecast_download(self, d, id, extensions=None):

        # for Load forecast IPTO
        id=1
        if d>=pd.to_datetime('01082020', format='%d%m%Y'):
            file_name = self.path_data + '/LoadForecastIPTO/new/load_forecast' + d.strftime(
                '%Y%m%d') + '.xlsx'
            months = [d.month, (d - pd.DateOffset(months=1)).month, (d + pd.DateOffset(months=1)).month]
            years = [d.year, (d - pd.DateOffset(months=12)).year, (d + pd.DateOffset(months=12)).year]
            for year in years:
                for month in months:

                    urls = ['https://www.admie.gr/sites/default/files/attached-files/type-file/' + str(year) + '/' + str(
                        month).zfill(2) + '/' + d.strftime('%Y%m%d') + '_ISP3IntraDayLoadForecast_01.xlsx',
                            'https://www.admie.gr/sites/default/files/attached-files/type-file/' + str(
                                year) + '/' + str(
                                month).zfill(2) + '/' + d.strftime('%Y%m%d') + '_ISP2DayAheadLoadForecast_01.xlsx',
                            'https://www.admie.gr/sites/default/files/attached-files/type-file/' + str(
                                year) + '/' + str(
                                month).zfill(2) + '/' + d.strftime('%Y%m%d') + '_ISP1DayAheadLoadForecast_02.xlsx',
                            'https://www.admie.gr/sites/default/files/attached-files/type-file/' + str(
                                year) + '/' + str(
                                month).zfill(2) + '/' + d.strftime('%Y%m%d') + '_ISP1DayAheadLoadForecast_01.xlsx']

                    for url in urls:
                        try:
                            # get request
                            response = get(url)
                        except:
                            print(url)
                            pass

                        if response.ok:
                            with open(file_name, "wb") as file:
                                file.write(response.content)
                                file.close()
                                print(url)
                            break
                        else:
                            print('bad response')
                            print(url)
                    if response.ok:
                        break
                if response.ok:
                    break
            if response.ok:
                return response.ok
            else:
                print('Cannot download ', file_name)

        else:
            return True

if __name__ == '__main__':
    # dates = pd.date_range(start=pd.to_datetime('05/09/2020', format='%d/%m/%Y'), end=pd.to_datetime('05/10/2020', format='%d/%m/%Y'), freq='D')
    # downloader = DataDownloader(dates)
    # downloader.download_files()
    dates = pd.date_range('2022-03-01', '2022-05-03')
    path_data = '/home/smartrue/Dropbox/current_codes/PycharmProjects/LV_System_ver3/data'
    dl = DataDownloader(dates, path_data)
    dl.download_scada()