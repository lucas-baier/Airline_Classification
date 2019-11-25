import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost.sklearn import XGBRegressor
import time
from datetime import datetime
import pickle
import gc

import joblib
from XGBoostModel import XGBoostModel

print('Data loading started: ', datetime.now())
start_time = time.time()

data = joblib.load("data_ORD_date_class.joblib")

#
#
#
# Local processing
# data = joblib.load("sample_data_ORD_date_class.joblib")

print('Duration Loading: ', (time.time() - start_time))



# Data starts with 1990-01-01, last entry is 2008-10-31, last prediction for Q3 2008, training from Q3/2006 - Q2/2008
xgboost_model = XGBoostModel(strategy_name='Quarterly_Update')

start_train_date = pd.Timestamp('1989-10-01')
no_retraining = (len(data.Year.unique())-2)*4
#no_retraining = 5

for i in range(no_retraining):

    start_train_date = start_train_date + pd.DateOffset(months = 3)
    end_train_date = start_train_date + pd.DateOffset(years = 2)

    start_test_date = end_train_date
    end_test_date = start_test_date + pd.DateOffset(months = 3)

    # print(start_train_date, end_train_date, start_test_date, end_test_date)


    X_train, y_train, X_test, y_test = xgboost_model.generate_data(data, start_train_date, end_train_date,
                                                                   start_test_date, end_test_date, verbose=1)

    xgboost_model.update_model(X_train, y_train)

    results_dict = xgboost_model.compute_predictions(X_test, y_test)


joblib.dump(results_dict, '{}_results_all.joblib'.format(xgboost_model.strategy_name), compress=3)