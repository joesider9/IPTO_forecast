import os
import joblib
import datetime
import astral

from astral.sun import sun

import pandas as pd
import numpy as np

from intraday.configuration.config import config
from eforecast.init.initialize import initializer
from eforecast.nwp_extraction.nwp_extractor import NwpExtractor
from eforecast.common_utils.date_utils import convert_timezone_dates

from download_load_data_online import DataDownloader
from online_reader_for_load import Reader

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


def find_missing_dates(static_data, date):
    last_date = date
    path_group = static_data['path_group']
    load_estimation_file = os.path.join(path_group, 'DATA', 'load_estimation_intra.csv')
    load_estimation_intra = pd.read_csv(load_estimation_file, index_col=0, header=0, parse_dates=True)
    dates_found = find_dates(load_estimation_intra, last_date)
    return load_estimation_intra, dates_found


def feature_engineering(static_data, res_mv, ape_net, scada_ts, inst_cap_ts, extra_data):
    path_group = static_data['path_group']
    dates = ape_net.index.difference(extra_data.index)
    dates = scada_ts.index.intersection(dates)
    dates = inst_cap_ts.index.intersection(dates)
    dates_res = res_mv.index.intersection(dates)
    res_tot = 0.95 * res_mv.loc[dates_res]
    pred = 0.95 * ape_net.loc[dates]
    scada_tot = scada_ts.loc[dates]
    inst_cap = 0.95 * inst_cap_ts.loc[dates]
    data = pd.concat([res_tot, pred, inst_cap, scada_tot], axis=1)
    data['ape_net_up'] = data.ape_net * data.pv_rated
    data['ape_net_fix'] = data['ape_net']
    data.pv_cap = data.pv_cap.clip(10, np.inf)
    data.pv_rated = data.pv_rated.clip(10, np.inf)
    pv_up = data.pv / data.pv_cap
    pv_up = pv_up.clip(0, 0.9)
    data['pv_up'] = (pv_up * data.pv_rated).clip(0, 4)
    data['rate_pv_cap'] = data.pv_cap / data.pv_rated
    data_max = data.groupby(data.index.date).agg({'ape_net_up': np.max, 'pv': np.max})
    data_max.columns = ['ape_net_up_max', 'pv_max']
    data_max.index = pd.DatetimeIndex([pd.to_datetime(d) for d in data_max.index])
    df_max = pd.DataFrame(index=pd.date_range(data_max.index[0], data.index[-1] + pd.DateOffset(hours=23), freq='H')
                          , columns=['ape_net_up_max', 'pv_max'])
    df_max.loc[data_max.index] = data_max
    df_max = df_max.fillna(method='ffill')
    data = pd.concat([data, df_max], axis=1)
    l = astral.LocationInfo('Custom Name', 'My Region', 'CET',
                            37.415839, 22.893832)
    for date in dates:
        sun_attr = sun(l.observer, date=date)
        sunrise = pd.to_datetime(sun_attr['dawn'].strftime('%Y%m%d %H:%M'), format='%Y%m%d %H:%M')
        sunset = pd.to_datetime(sun_attr['dusk'].strftime('%Y%m%d %H:%M'), format='%Y%m%d %H:%M')
        if not sunrise < date < sunset:
            data.loc[date, 'ape_net_fix'] = 0

    data['load_estimation'] = data.scada + data.pv_up
    data['load_estimation_pred'] = data.scada + data.ape_net_up

    file_extra = os.path.join(path_group, 'DATA', 'extra_data.csv')
    file_res_tot = os.path.join(path_group, 'DATA', 'res_mv_ts.csv')
    file_pred = os.path.join(path_group, 'DATA', 'ape_net.csv')
    file_scada_tot = os.path.join(path_group, 'DATA', 'scada_ts.csv')
    file_inst_cap = os.path.join(path_group, 'DATA', 'inst_cap.csv')
    cols = [col for col in data.columns
            if col not in res_tot.columns and col not in pred.columns and col not in scada_tot.columns and col not in inst_cap.columns]
    extra_data = pd.concat([extra_data, data[cols]])
    col_nan = ['pv_up', 'rate_pv_cap', 'load_estimation']
    for col in col_nan:
        ind_nan = np.where(extra_data[col].isna())[0]
        extra_data[col].iloc[ind_nan] = extra_data[col].iloc[ind_nan - 24].values
    extra_data.to_csv(file_extra)
    res_mv.to_csv(file_res_tot)
    scada_ts.to_csv(file_scada_tot)
    inst_cap_ts.to_csv(file_inst_cap)

def receive_data(static_data, date):
    last_date = date
    path_data = os.path.join(static_data['path_group'], 'DATA')
    reader = Reader(path_data, date=last_date)
    scada_ts, res_mv, inst_cap, ape_pred, extra_data = reader.load_files()
    scada_ts = scada_ts[scada_ts.index < last_date]
    extra_data = extra_data[extra_data.index < last_date - pd.DateOffset(days=10)]
    dfs_new = []
    for df in [scada_ts, ape_pred, res_mv, inst_cap, extra_data]:
        dates_found = find_dates(df, last_date)
        dates_found = dates_found[dates_found > date - pd.DateOffset(months=3)]
        if 'scada' in df.columns or 'pv' in df.columns:
            downloader = DataDownloader(dates_found[:-1], path_data)
            downloader.download_scada() if 'scada' in df.columns else downloader.download_res()
        for dt in dates_found:
            if 'scada' in df.columns or 'pv' in df.columns:
                reader_temp = Reader(path_data, date=dt)
                if 'scada' in df.columns:
                    if dt == last_date:
                        df = reader_temp.read_scada_real_time(df)
                    else:
                        try:
                            df = reader_temp.read_scada(df)
                        except:
                            df = reader_temp.read_scada_real_time(df)
                    df[df < 1000] = np.nan
                else:
                    if dt != last_date:
                        df = reader_temp.read_res_mv_non_verified(df)

        df = impute_missing_values(df)
        dfs_new.append(df)
    scada_ts, ape_pred, res_mv, inst_cap, extra_data = dfs_new
    feature_engineering(static_data, res_mv, ape_pred, scada_ts, inst_cap, extra_data)

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
    date_h = pd.to_datetime(datetime.datetime.now().strftime('%d%m%y %H'), format='%d%m%y %H')
    if static_data['Docker'] and not ('smartrue' in os.getcwd() or 'joesider' in os.getcwd()):
        date_h = convert_timezone_dates(pd.DatetimeIndex([date_h]), timezone1='UTC', timezone2='Europe/Athens')[0]
    print(date_h)
    date = pd.to_datetime(date_h.strftime('%d%m%y'), format='%d%m%y')
    receive_data(static_data, date)
    print(f'Load Estimation Intra-head model start at {date}')
    load_estimation_intra, dates = find_missing_dates(static_data, date)
    pred = pd.DataFrame(columns=load_estimation_intra.columns)
    if not static_data['Docker'] or date_h.hour == 10:
        dates = dates[dates > date - pd.DateOffset(months=3)]
        if len(dates) > 0:
            print(f'Load Estimation Data are collected  initialize successfully ')
            pred = predict_intra(dates[:-1], 'CatBoost_classifier', model_horizon='intra')
            pred.columns = load_estimation_intra.columns
    dates_ahead = [date + pd.DateOffset(hours=h) for h in range(date_h.hour)]
    pred_ahead = predict_intra(pd.DatetimeIndex(dates_ahead), 'CatBoost_classifier', model_horizon='short_term')
    pred_ahead.columns = load_estimation_intra.columns
    pred_ahead = pd.concat([pred, pred_ahead])

    pred_ahead = pred_ahead.sort_index()
    print(f'Load Estimation Sort-term model predictions are made successfully')
    if pred.shape[0] > 0:
        path_group = static_data['path_group']
        pred = pred.sort_index()
        load_estimation_file = os.path.join(path_group, 'DATA', 'load_estimation_intra.csv')
        save_predictions(load_estimation_intra, pred, load_estimation_file)
    path_group = static_data['path_group']
    load_estimation_file = os.path.join(path_group, 'DATA', 'load_estimation_short.csv')
    save_predictions(load_estimation_intra, pred_ahead, load_estimation_file)
    print(f'Load Estimation Intra-head model predictions are saved successfully')
