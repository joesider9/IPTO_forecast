import os
import datetime
import pandas as pd


from day_ahead.configuration.config import config
from eforecast.init.initialize import initializer
from eforecast.common_utils.date_utils import convert_timezone_dates
from sent_predictions_intra import send_intra_day_ahead
from sent_predictions_short_term import send_short_term


static_data_group = config()

date = pd.to_datetime(datetime.datetime.now().strftime('%d%m%y'), format='%d%m%y')

static_data = initializer(config(), online=True)

if __name__ == '__main__':
    static_data = initializer(config(), online=True)
    date_h = pd.to_datetime(datetime.datetime.now().strftime('%d%m%y %H'), format='%d%m%y %H')
    if static_data['Docker'] and not ('smartrue' in os.getcwd() or 'joesider' in os.getcwd()):
        date_h = convert_timezone_dates(pd.DatetimeIndex([date_h]), timezone1='UTC', timezone2='Europe/Athens')[0]
    print(date_h)
    date = pd.to_datetime(date_h.strftime('%d%m%y'), format='%d%m%y')
    if date_h.hour == 10 and not ('smartrue' in os.getcwd() or 'joesider' in os.getcwd()):
        send_intra_day_ahead()
    send_short_term()