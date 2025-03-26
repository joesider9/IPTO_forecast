import os
import datetime
import pandas as pd
import yagmail

from credentials import Credentials, JsonFileBackend

from day_ahead.configuration.config import config
from eforecast.init.initialize import initializer



static_data_group = config()

date = pd.to_datetime(datetime.datetime.now().strftime('%d%m%y'), format='%d%m%y')

file_cred='filemail.json'

credobj = Credentials([JsonFileBackend(file_cred)])

static_data = initializer(config(), online=True)


def read_lv_load(path_pred):
    date_pred = (date + pd.DateOffset(days=1)).strftime('%d_%m_%Y')
    file_pred = os.path.join(path_pred, f"LV_load_day_ahead_hist_{date.strftime('%d_%m_%Y')}.csv")
    file_proba = os.path.join(path_pred, f"Proba_LV_load_day_ahead_hist_{date.strftime('%d_%m_%Y')}.csv")
    pred = pd.read_csv(file_pred, index_col=0, header=0, parse_dates=True)
    pred_proba = pd.read_csv(file_proba, index_col=0, header=0, parse_dates=True)
    pred_nan = pd.DataFrame(index=pred.index)
    pred = pd.concat([pred, pred_nan], axis=1)
    pred_proba = pd.concat([pred_proba, pred_nan], axis=1)
    return {f'LV_load_DA_{date_pred}': pred}, {f'LV_load_DA_proba_{date_pred}': pred_proba}


def read_total_load(path_pred):
    date_pred = (date + pd.DateOffset(days=1)).strftime('%d_%m_%Y')
    file_pred = os.path.join(path_pred, f"Total_load_day_ahead_hist_{date.strftime('%d_%m_%Y')}.csv")
    file_proba = os.path.join(path_pred, f"Proba_Total_load_day_ahead_hist_{date.strftime('%d_%m_%Y')}.csv")
    pred = pd.read_csv(file_pred, index_col=0, header=0, parse_dates=True)
    pred_proba = pd.read_csv(file_proba, index_col=0, header=0, parse_dates=True)
    pred_nan = pd.DataFrame(index=pred.index)
    pred = pd.concat([pred, pred_nan], axis=1)
    pred_proba = pd.concat([pred_proba, pred_nan], axis=1)
    return {f'Total_load_DA_{date_pred}': pred}, {f'Total_load_DA_proba_{date_pred}': pred_proba}


def read_lv_load_intra(path_pred):
    date_pred = date.strftime('%d_%m_%Y')
    file_pred = os.path.join(path_pred, f"LV_load_Intra_ahead_hist_{date.strftime('%d_%m_%Y')}.csv")
    file_proba = os.path.join(path_pred, f"Proba_LV_load_Intra_ahead_hist_{date.strftime('%d_%m_%Y')}.csv")
    pred = pd.read_csv(file_pred, index_col=0, header=0, parse_dates=True)
    pred_proba = pd.read_csv(file_proba, index_col=0, header=0, parse_dates=True)
    pred_nan = pd.DataFrame(index=pred.index)
    pred = pd.concat([pred, pred_nan], axis=1)
    pred_proba = pd.concat([pred_proba, pred_nan], axis=1)
    return {f'LV_load_INTRA_{date_pred}': pred}, {f'LV_load_INTRA_proba_{date_pred}': pred_proba}


def read_total_load_intra(path_pred):
    date_pred = date.strftime('%d_%m_%Y')
    file_pred = os.path.join(path_pred, f"total_load_Intra_ahead_hist_{date.strftime('%d_%m_%Y')}.csv")
    file_proba = os.path.join(path_pred, f"Proba_total_load_Intra_ahead_hist_{date.strftime('%d_%m_%Y')}.csv")
    pred = pd.read_csv(file_pred, index_col=0, header=0, parse_dates=True)
    pred_proba = pd.read_csv(file_proba, index_col=0, header=0, parse_dates=True)
    pred_nan = pd.DataFrame(index=pred.index)
    pred = pd.concat([pred, pred_nan], axis=1)
    pred_proba = pd.concat([pred_proba, pred_nan], axis=1)

    return {f'Total_load_INTRA_{date_pred}': pred}, {f'Total_load_INTRA_proba_{date_pred}': pred_proba}



def reform_dict(dictionary, t=tuple(), reform=[]):
    for key, val in dictionary.items():
        t = t + (key,)
        if isinstance(val, dict):
            reform_dict(val, t, reform)
        else:
            columns = [t + (col,) for col in val.columns]
            val.columns = pd.MultiIndex.from_tuples(columns)
            reform.append(val)
        t = t[:-1]
    return reform


def send_files(files):
    contents = ' '
    # The mail addresses and password
    sender_address = "joesider9@gmail.com"
    sender_pass = "eupzzyzjzamlkbcm"
    yag_smtp_connection = yagmail.SMTP(user=sender_address, password=sender_pass, host='smtp.gmail.com', smtp_ssl=False)

    subject = 'Load predictions based only on historical data for ' + date.strftime('%Y%m%d')
    receivers = ['joesider@power.ece.ntua.gr', 'supply@fysikoaerioellados.gr']
    for receiver_address in receivers:
        yag_smtp_connection.send(to=receiver_address, subject=subject, contents=contents, attachments=files)

    print(f"Mail Sent for {(date).strftime('%d_%m_%Y')}")


def send_error():
    contents = ' '
    # The mail addresses and password
    sender_address = "joesider9@gmail.com"
    sender_pass = "eupzzyzjzamlkbcm"
    yag_smtp_connection = yagmail.SMTP(user=sender_address, password=sender_pass, host='smtp.gmail.com', smtp_ssl=False)

    subject = 'Error at ' + date.strftime('%d_%m_%Y %H')
    receivers = ['joesider@power.ece.ntua.gr']
    for receiver_address in receivers:
        yag_smtp_connection.send(to=receiver_address, subject=subject, contents=contents)


def send_intra_day_ahead():
    try:
        path_pred = os.path.join(static_data['path_group'], 'predictions')

        pred_lv_da, proba_lv_da = read_lv_load(path_pred)
        pred_lv_intra, proba_lv_intra = read_lv_load_intra(path_pred)
        pred_total_da, proba_total_da = read_total_load(path_pred)
        pred_total_intra, proba_total_intra = read_total_load_intra(path_pred)
        preds_da = {f"Day-Ahead-predictions": {**pred_lv_da, **pred_total_da}}
        preds_intra = {f"Intra-Day-predictions": {**pred_lv_intra, **pred_total_intra}}
        proba_da = {f"Proba_Day-Ahead-predictions": {**proba_lv_da, **proba_total_da}}
        proba_intra = {f"Proba_Intra-Day-predictions": {**proba_lv_intra, **proba_total_intra}}

        pred_da = {**preds_da, **proba_da}
        pred_intra = {**preds_intra, **proba_intra}
        reform_da = reform_dict(pred_da, t=tuple(), reform=[])
        reform_intra = reform_dict(pred_intra, t=tuple(), reform=[])
        pred_df_da = pd.concat(reform_da, axis=1)
        pred_df_intra = pd.concat(reform_intra, axis=1)

        files = [os.path.join(path_pred,
                                       f"Day_Ahead_forecasts_{(date + pd.DateOffset(days=1)).strftime('%d_%m_%Y')}.xlsx"),
                 os.path.join(path_pred,
                              f"Intra_Day_forecasts_{(date + pd.DateOffset(days=1)).strftime('%d_%m_%Y')}.xlsx")
                 ]
        pred_df_da.to_excel(files[0])
        pred_df_intra.to_excel(files[1])

        send_files(files)
    except:
        send_error()