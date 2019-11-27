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
from custom_drift_detectors import HDDDM, MannKendall, STEPD

from skmultiflow.drift_detection.adwin import ADWIN

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
# print('Local file loaded')
#
print('Duration Loading: ', (time.time() - start_time))

# delta = 23 leads to 56 drifts!
adwin = ADWIN(delta=23)

#Local ADWIN
# adwin = ADWIN(delta = 15)


#
#
# Start Retraining
#
#


xgboost_model = XGBoostModel(strategy_name='ADWIN_Retraining')
list_drift = []
training_flag = True

start_train_date = pd.Timestamp('1990-01-01')
start_test_date = pd.Timestamp('1992-01-01')

print('ADWIN parameters: ', adwin.delta)

while start_test_date < pd.Timestamp('2008-10-01'):

    end_train_date = start_train_date + pd.DateOffset(years = 2)
    end_test_date = start_test_date + pd.DateOffset(months = 3)

    X_train, y_train, X_test, y_test = xgboost_model.generate_data(data, start_train_date, end_train_date,
                                                                       start_test_date, end_test_date, verbose=1)

    if training_flag:
        xgboost_model.fit_model(X_train, y_train)

    results_dict = xgboost_model.compute_predictions(X_test, y_test)

    temp_drifts = []

    df_results = pd.DataFrame({'y_true': results_dict['y_true'][-1], 'y_pred': results_dict['Predictions'][-1]})
    df_results['Correct'] = (df_results['y_true'] == df_results['y_pred'])

    for i in range(df_results.shape[0]):
        adwin.add_element(df_results['Correct'].iloc[i])
        if adwin.detected_change():
            print('Change detected ADWIN in data: ' + str(df_results['Correct'].iloc[i]) + ' - at date: ' + str(results_dict['Date'][-1].iloc[i]))
            temp_drifts.append(results_dict['Date'][-1].iloc[i])
            adwin.reset()

    if not temp_drifts:
        print('No Drift Detected - Predict next three months')
        start_test_date = start_test_date + pd.DateOffset(months = 3)
        training_flag = False

    if temp_drifts:
        print('Drift detected - Retrain model')
        list_drift.append(temp_drifts[0])
        start_train_date = temp_drifts[0] - pd.DateOffset(years = 2)
        start_test_date = start_train_date + pd.DateOffset(years =2)
        training_flag = True

# Save drift dates to results file
results_dict['Drifts'] = list_drift

# Save results
xgboost_model.save_results(results_dict)




#
# Start Update
#




# Data starts with 1990-01-01, last entry is 2008-10-31, last prediction for Q3 2008, training from Q3/2006 - Q2/2008
xgboost_model = XGBoostModel(strategy_name='ADWIN_Update')
list_drift = []
training_flag = True

start_train_date = pd.Timestamp('1990-01-01')
start_test_date = pd.Timestamp('1992-01-01')

print('ADWIN parameters: ', adwin.delta)

while start_test_date < pd.Timestamp('2008-10-01'):

    end_train_date = start_train_date + pd.DateOffset(years = 2)
    end_test_date = start_test_date + pd.DateOffset(months = 3)

    X_train, y_train, X_test, y_test = xgboost_model.generate_data(data, start_train_date, end_train_date,
                                                                       start_test_date, end_test_date, verbose=1)

    if training_flag:
        xgboost_model.update_model(X_train, y_train)

    results_dict = xgboost_model.compute_predictions(X_test, y_test)

    temp_drifts = []


    df_results = pd.DataFrame({'y_true': results_dict['y_true'][-1], 'y_pred': results_dict['Predictions'][-1]})
    df_results['Correct'] = (df_results['y_true'] == df_results['y_pred'])

    for i in range(df_results.shape[0]):
        adwin.add_element(df_results['Correct'].iloc[i])
        if adwin.detected_change():
            print('Change detected ADWIN in data: ' + str(df_results['Correct'].iloc[i]) + ' - at date: ' + str(results_dict['Date'][-1].iloc[i]))
            temp_drifts.append(results_dict['Date'][-1].iloc[i])
            adwin.reset()

    if not temp_drifts:
        print('No Drift Detected - Predict next three months')
        start_test_date = start_test_date + pd.DateOffset(months = 3)
        training_flag = False

    if temp_drifts:
        print('Drift detected - Retrain model')
        list_drift.append(temp_drifts[0])
        start_train_date = temp_drifts[0] - pd.DateOffset(years = 2)
        start_test_date = start_train_date + pd.DateOffset(years =2)
        training_flag = True

# Save drift dates to results file
results_dict['Drifts'] = list_drift

# Save results
xgboost_model.save_results(results_dict)










