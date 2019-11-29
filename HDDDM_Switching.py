import pandas as pd
import time
from datetime import datetime, timedelta
from custom_drift_detectors import HDDDM
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

print('Duration Loading: ', (time.time() - start_time))


# Data starts with 1990-01-01, last entry is 2008-10-31, last prediction for Q3 2008, training from Q3/2006 - Q2/2008
xgboost_model = XGBoostModel(strategy_name='HDDDM_Switching')
list_drift = []
training_flag = True
update_flag = False

start_train_date = pd.Timestamp('1990-01-01')
start_test_date = pd.Timestamp('1992-01-01')

# delta = 23 leads to 56 drifts!
hdddm = HDDDM(batch_size=20000, gamma=0.8)

#Local ADWIN
# hdddm = HDDDM(batch_size=500, gamma=1)

print('HDDDM parameters: ', hdddm.batch_size, hdddm.gamma)

while start_test_date < pd.Timestamp('2008-10-01'):

    end_train_date = start_train_date + pd.DateOffset(years = 2)
    end_test_date = start_test_date + pd.DateOffset(months = 3)

    X_train, y_train, X_test, y_test = xgboost_model.generate_data(data, start_train_date, end_train_date,
                                                                       start_test_date, end_test_date, verbose=1)

    if training_flag:
        xgboost_model.fit_model(X_train, y_train)

        # Add date of fitting to train_dates
        xgboost_model.results['Training'].append(end_train_date)


    if update_flag:
        xgboost_model.update_model(X_train, y_train)

        # Add date of updating to update_dates
        xgboost_model.results['Update'].append(end_train_date)


    results_dict = xgboost_model.compute_predictions(X_test, y_test)

    temp_drifts = []

    df_results = pd.DataFrame({'y_true': results_dict['y_true'][-1], 'y_pred': results_dict['Predictions'][-1]})
    df_results['Correct'] = (df_results['y_true'] == df_results['y_pred'])

    for i in range(df_results.shape[0]):
        hdddm.add_element(df_results['Correct'].iloc[i])
        if hdddm.detected_change():
            print('Change detected HDDDM in data: ' + str(df_results['Correct'].iloc[i]) + ' - at date: ' + str(results_dict['Date'][-1].iloc[i]))
            temp_drifts.append(results_dict['Date'][-1].iloc[i])
            hdddm.reset()

    if not temp_drifts:
        print('No Drift Detected - Predict next three months')
        start_test_date = start_test_date + pd.DateOffset(months = 3)
        training_flag = False
        update_flag = False

    if temp_drifts:
        print('Drift detected - Choice on model')
        list_drift.append(temp_drifts[0])
        start_train_date = temp_drifts[0] - pd.DateOffset(years = 2)
        start_test_date = start_train_date + pd.DateOffset(years =2)

        if (temp_drifts[0] - datetime.date(xgboost_model.results['Training'][-1])) > timedelta(days = 365):
            training_flag = True
            print('Retraining of model')
            print('Date today: ', temp_drifts[0])
            print('Date last training: ', datetime.date(xgboost_model.results['Training'][-1]))

        else:
            update_flag = True
            print('Update of model')
            print('Date today: ', temp_drifts[0])
            print('Date last training: ', datetime.date(xgboost_model.results['Training'][-1]))

# Save drift dates to results file
results_dict['Drifts'] = list_drift

# Save results
xgboost_model.save_results(results_dict)