import os
import datetime
import pandas as pd
import yagmail

from credentials import Credentials, JsonFileBackend
from eforecast.common_utils.date_utils import convert_timezone_dates

from day_ahead.configuration.config import config
from eforecast.init.initialize import initializer



static_data_group = config()


static_data = initializer(config(), online=True)

file_cred='filemail.json'

credobj = Credentials([JsonFileBackend(file_cred)])


def read_files(path_pred, date):
    date_pred = date.strftime('%d_%m_%Y %H')
    file_pred_total = os.path.join(path_pred, f"total_load_short_term_{date.strftime('%d_%m_%Y %H')}.csv")
    file_pred_lv = os.path.join(path_pred, f"LV_load_short_term_{date.strftime('%d_%m_%Y %H')}.csv")
    pred_total = pd.read_csv(file_pred_total, index_col=0, header=0, parse_dates=True)
    pred_lv = pd.read_csv(file_pred_lv, index_col=0, header=0, parse_dates=True)
    return {f'Total_load_ST_{date_pred}': pred_total, f'LV_load_ST_{date_pred}': pred_lv}


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


def send_files(files, date):
    contents = ' '
    # The mail addresses and password
    sender_address = "joesider9@gmail.com"
    sender_pass = "eupzzyzjzamlkbcm"
    yag_smtp_connection = yagmail.SMTP(user=sender_address, password=sender_pass, host='smtp.gmail.com', smtp_ssl=False)

    subject = 'Short-term predictions for ' + date.strftime('%d_%m_%Y %H')
    receivers = ['joesider@power.ece.ntua.gr', 'supply@fysikoaerioellados.gr']
    for receiver_address in receivers:
        yag_smtp_connection.send(to=receiver_address, subject=subject, contents=contents, attachments=files)

    print(f"Mail Sent for {(date).strftime('%d_%m_%Y %H')}")


def send_error(date):
    contents = ' '
    # The mail addresses and password
    sender_address = "joesider9@gmail.com"
    sender_pass = "eupzzyzjzamlkbcm"
    yag_smtp_connection = yagmail.SMTP(user=sender_address, password=sender_pass, host='smtp.gmail.com', smtp_ssl=False)

    subject = 'Error short term at ' + date.strftime('%d_%m_%Y %H')
    receivers = ['joesider@power.ece.ntua.gr']
    for receiver_address in receivers:
        yag_smtp_connection.send(to=receiver_address, subject=subject, contents=contents)


def send_short_term():
    date = pd.to_datetime(datetime.datetime.now().strftime('%d%m%y %H'), format='%d%m%y %H')
    try:
        if static_data['Docker'] and not ('smartrue' in os.getcwd() or 'joesider' in os.getcwd()):
            date = convert_timezone_dates(pd.DatetimeIndex([date]), timezone1='UTC', timezone2='CET')[0]
    except:
        pass
    date = convert_timezone_dates(pd.DatetimeIndex([date]), timezone1='Europe/Athens', timezone2='CET')[0]
    date = date + pd.DateOffset(hours=1)
    print(f'Send predictions for {date}')

    try:
        path_pred = os.path.join(static_data['path_group'], 'predictions')

        preds = read_files(path_pred, date)
        reform_st = reform_dict(preds, t=tuple(), reform=[])
        pred_df_st = pd.concat(reform_st, axis=1)

        files = [os.path.join(path_pred,
                                       f"Short_term_forecasts_{date.strftime('%d_%m_%Y %H')}.xlsx"),
                 ]
        pred_df_st.to_excel(files[0])

        send_files(files, date)
    except:
        send_error(date)