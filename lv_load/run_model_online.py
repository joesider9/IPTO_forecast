import os
import joblib
import datetime
import astral

from astral.sun import sun

import pandas as pd
import numpy as np

from intra_day.configuration.config import config
from eforecast.init.initialize import initializer
from eforecast.nwp_extraction.nwp_extractor import NwpExtractor
from eforecast.common_utils.date_utils import convert_timezone_dates


from run_model_script import predict_intra



def impute_missing_values(df):
    dates = pd.date_range(df.index[0], df.index[-1], freq='H')
    df_new = pd.DataFrame(index=dates, columns=df.columns)
    df_new.loc[df.index] = df
    ind_nan = np.where(df_new.isna().any(axis=1))
    if np.size(ind_nan) > 0:
        ind_nan = ind_nan[0]
        dates_nan = df_new.index[ind_nan]
        for dt in dates_nan:
            dts = pd.DatetimeIndex([dt - pd.DateOffset(days=1), dt - pd.DateOffset(days=7)])
            df_new.loc[dt] = df_new.loc[dts].mean(axis=0, skipna=True)
    return df_new


def find_dates(df, last_date):
    dates = pd.date_range(df.index[0], last_date + pd.DateOffset(hours=23), freq='H')
    dates_found = pd.DatetimeIndex(dates.difference(df.index).date).unique()
    return dates_found


def nwp_extraction(static_data, dates):
    max_lag = [min(var_data['lags']) for var_data in static_data['variables']
               if var_data['source'] == 'nwp_dataset']
    if len(max_lag) > 0:
        max_lag = min(max_lag)
        dates = dates.sort_values()
        dates = pd.date_range(dates[0] + pd.DateOffset(hours=max_lag), dates[-1])

    nwp_extractor = NwpExtractor(static_data, recreate=False, is_online=True, dates=dates)
    nwp_extractor.extract()


if __name__ == '__main__':
    static_data = initializer(config(), online=True)
    date_h = pd.to_datetime(datetime.datetime.now().strftime('%d%m%y %H'), format='%d%m%y %H')
    if static_data['Docker'] and not ('smartrue' in os.getcwd() or 'joesider' in os.getcwd()):
        date_h = convert_timezone_dates(pd.DatetimeIndex([date_h]), timezone1='UTC', timezone2='Europe/Athens')[0]
    print(date_h)
    date = pd.to_datetime(date_h.strftime('%d%m%y'), format='%d%m%y')
    path_pred = os.path.join(static_data['path_group'], 'predictions')
    if not os.path.exists(path_pred):
        os.makedirs(path_pred)
    print(f'LV load model start at {date}')
    if not static_data['Docker'] or date_h.hour == 10:
        dates = pd.DatetimeIndex([date])
        if len(dates) > 0:
            print(f'LV load Data are collected  initialize successfully ')

            pred_day_ahead, proba_ahead= predict_intra(dates, 'kmeans', model_horizon='day_ahead')
            pred_day_ahead.columns = ['LV_load_day_ahead']
            pred_day_ahead.to_csv(os.path.join(path_pred, f"LV_load_day_ahead_hist_{date.strftime('%d_%m_%Y')}.csv"))
            proba_ahead.to_csv(
                os.path.join(path_pred, f"Proba_LV_load_day_ahead_hist_{date.strftime('%d_%m_%Y')}.csv"))
            pred_intra, proba_intra = predict_intra(dates, 'bcp', model_horizon='intra')
            pred_intra.columns = ['LV_load_intra']
            pred_intra.to_csv(os.path.join(path_pred, f"LV_load_Intra_ahead_hist_{date.strftime('%d_%m_%Y')}.csv"))
            proba_intra.to_csv(
                os.path.join(path_pred, f"Proba_LV_load_Intra_ahead_hist_{date.strftime('%d_%m_%Y')}.csv"))
    date_h = convert_timezone_dates(pd.DatetimeIndex([date_h]), timezone1='Europe/Athens', timezone2='CET')[0]
    dates_short = [date_h]
    pred_short = predict_intra(pd.DatetimeIndex(dates_short), 'kmeans', model_horizon='short_term')
    pred_short.to_csv(os.path.join(path_pred, f"LV_load_short_term_{date_h.strftime('%d_%m_%Y %H')}.csv"))
    print(f'LV_load Intra-head model predictions are saved successfully')
