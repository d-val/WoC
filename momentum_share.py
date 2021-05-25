

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
global true_value, dic, accuracy, ind_list
from scipy.stats import pearsonr
from scipy.stats import linregress
import os
from collections import Counter
from time import time
import readline
from dashboard import *
import copy
import pickle as pkl
import datetime as dt
from dateutil import parser
from sklearn import linear_model
import time
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

start = ['2016-06-01', '2016-06-28', '2016-07-16','2016-09-17','2016-10-15','2016-11-12','2016-11-28']
deadline = ['2016-06-17', '2016-07-08', '2016-07-29','2016-09-30','2016-10-21','2016-11-25','2016-12-12']
end = ['2016-06-24', '2016-07-15', '2016-08-05','2016-10-07','2016-10-28','2016-12-02','2016-12-19']
instru = ['ESY00', 'CLY00', 'GCY00','ESY00','ESY00','ESY00','ESY00']

final_error = pd.DataFrame()
for i in range(len(start)):
    print('>>>>>>>>>', i)
    freq = 'daily'
    fetch_online = True

    futures_data = dl_futures_data(start[i], end[i], instru[i], freq, True, fetch_online, i)

    futures_data['timestamp'] = futures_data['timestamp'].apply(parser.parse)
    try:
        futures_data['timestamp'] = (futures_data['timestamp'] - parser.parse('1970-01-01T00:00:00-05:00')).dt.total_seconds()
    except ValueError:
        print('timezone')
        futures_data['timestamp'] = (futures_data['timestamp'] - parser.parse('1970-01-01T00:00:00-06:00')).dt.total_seconds()


    deadline_val = ( parser.parse(deadline[i]) - parser.parse('1970-01-01T00:00:00')).total_seconds()

    final_close = list(futures_data['close'])[-1]

    target = 'high'
    data = futures_data[[target, 'timestamp']]

    data_train = data.loc[futures_data['timestamp'] < deadline_val ]
    data_test  = data

    X_train = data_train['timestamp']
    X_train = X_train.reshape(-1,1)

    X_test  = data_test['timestamp']
    X_test = X_test.reshape(-1,1)

    y_train = data_train[target]
    y_test = data_test[target]

    lm =  sm.OLS
    model = lm(y_train, X_train).fit()
    print(model.summary() )

    predictions = lm.predict(X_test)
    error_pct = (predictions[-1] - final_close)/final_close*100.0
    error = (predictions[-1] - final_close)/final_close

    final_error = final_error.append({
                                    'i': i,
                                    'instrument': instru[i],
                                    'start': start[i],
                                    'real_price': final_close,
                                    'predict': predictions[-1],
                                    'error': abs(error),
                                    'error_pct': abs(error_pct)
                                    }, ignore_index=True)


print(final_error)
final_error.to_csv('final_error.csv', index = False)


















































