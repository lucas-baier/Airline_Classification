import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import time
from datetime import datetime
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
#
print('Duration Loading: ', (time.time() - start_time))



# Data starts with 1990-01-01, last entry is 2008-10-31, last prediction for Q3 2008, training from Q3/2006 - Q2/2008
xgboost_model = XGBoostModel(strategy_name='Yearly_Training')

start_train_date = pd.Timestamp('1989-01-01')


for i in range(1990, 2007):

    start_train_date = start_train_date + pd.DateOffset(years = 1)
    end_train_date = start_train_date + pd.DateOffset(years = 2)

    start_test_date = end_train_date
    end_test_date = start_test_date + pd.DateOffset(years = 1)


    X_train, y_train, X_test, y_test = xgboost_model.generate_data(data, start_train_date, end_train_date,
                                                                   start_test_date, end_test_date, verbose=1)

    xgboost_model.fit_model(X_train, y_train)

    results_dict = xgboost_model.compute_predictions(X_test, y_test)

xgboost_model.save_results(results_dict)