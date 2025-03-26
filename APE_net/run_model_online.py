import os
import joblib
import datetime

import pandas as pd
import numpy as np

from intraday.configuration.config import config
from eforecast.init.initialize import initializer
from eforecast.nwp_extraction.nwp_extractor import NwpExtractor
from eforecast.common_utils.date_utils import convert_timezone_dates

from run_model_script import predict_intra

date = pd.to_datetime(datetime.datetime.now().strftime('%d%m%y'), format='%d%m%y')



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


def find_missing_dates(static_data):
    last_date = date
    path_group = static_data['path_group']
    APE_predictions_file = os.path.join(path_group, 'DATA', 'APE_predictions.csv')
    APE_predictions = pd.read_csv(APE_predictions_file, index_col=0, header=0, parse_dates=True)
    dates_found = find_dates(APE_predictions, last_date)
    return APE_predictions, dates_found


def nwp_extraction(static_data, dates):
    nwp_extractor = NwpExtractor(static_data, recreate=False, is_online=True, dates=dates)
    nwp_extractor.extract()


def save_predictions(predictions, pred, filename):
    dates_update = predictions.index.union(pred.index).sort_values()
    predictions_new = pd.DataFrame(index=dates_update, columns=predictions.columns)
    predictions_new.loc[predictions.index] = predictions
    predictions_new.loc[pred.index] = pred
    predictions_new = impute_missing_values(predictions_new)
    predictions_new.to_csv(filename)


if __name__ == '__main__':
    static_data = initializer(config(), online=True)
    date = pd.to_datetime(datetime.datetime.now().strftime('%d%m%y %H'), format='%d%m%y %H')
    if static_data['Docker'] and not ('smartrue' in os.getcwd() or 'joesider' in os.getcwd()):
        date = convert_timezone_dates(pd.DatetimeIndex([date]), timezone1='UTC', timezone2='Europe/Athens')[0]
    date = pd.to_datetime(date.strftime('%d%m%y'), format='%d%m%y')
    print(f'APE_net Intra-head model start at {date}')
    APE_predictions, dates = find_missing_dates(static_data)
    dates = dates[dates > date - pd.DateOffset(months=3)]
    pred = pd.DataFrame(columns=APE_predictions.columns)
    dates_for_extract = pd.DatetimeIndex(pd.date_range(date - pd.DateOffset(days=18), date))
    dates_for_extract = dates_for_extract.union(dates)
    try:
        nwp_extraction(static_data, dates_for_extract)
    except:
        pass
    if len(dates) > 0:
        print(f'APE_net Data are collected  initialize successfully for day-head model')
        pred = predict_intra(dates, 'CatBoost_classifier', model_horizon='intra')
        pred.columns = APE_predictions.columns
    dates_ahead = [date]
    if date not in pred.index and pred.shape[0] > 0:
        dates_ahead.append(date - pd.DateOffset(days=1))
    pred_ahead = predict_intra(pd.DatetimeIndex(dates_ahead), 'kmeans', model_horizon='day_ahead')
    pred_ahead.columns = APE_predictions.columns
    pred_ahead = pd.concat([pred, pred_ahead])

    pred_ahead = pred_ahead.sort_index()
    print(f'APE_net Intra-head model predictions are made successfully')
    if pred.shape[0] > 0:
        path_group = static_data['path_group']
        pred = pred.sort_index()
        APE_predictions_file = os.path.join(path_group, 'DATA', 'APE_predictions.csv')
        save_predictions(APE_predictions, pred, APE_predictions_file)
    path_group = static_data['path_group']
    APE_predictions_file = os.path.join(path_group, 'DATA', 'ape_net.csv')
    save_predictions(APE_predictions, pred_ahead, APE_predictions_file)
    print(f'APE_net Intra-head model predictions are saved successfully')
