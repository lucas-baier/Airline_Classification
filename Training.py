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

print('Data loading started: ', datetime.now())
start_time = time.time()
data = joblib.load("data_ORD.joblib")
print('Duration Loading: ', (time.time() - start_time))

data = data.reset_index(drop = True)

data.loc[data['ArrDelay'] < 0, 'ArrDelay'] = 0

y = data.loc[:,'ArrDelay']
X = data.drop(['ArrDelay'], axis=1)

print(y.shape)

train_range = 1989

# X_train = X[(X['Year'] < train_range) & (X['Month'] < 11)]
X_train = X[(X['Year'] < train_range)]
print(X_train.shape)
y_train = y[y.index.isin(X_train.index)]

def fit_model(X, y):
    print('Model Fitting started: ', datetime.now())
    start_time = time.time()

    classifier = XGBRegressor(objective='reg:squarederror', n_jobs=8, n_estimators= 1000, verbosity= 3)
    classifier.fit(X_train, y_train)
    pickle.dump(classifier, open("staticModel.pickle.dat", 'wb'))

    print('Duration Fitting: ', (time.time() - start_time))

    return classifier

model = fit_model(X_train, y_train)

loaded_model = pickle.load(open("staticModel.pickle.dat",'rb'))

def RMSE(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse

def sMAPE(y_true, y_pred):
    smape = 100 * np.average(np.abs(y_pred - y_true) / ((np.abs(y_pred) + np.abs(y_true))/2))
    return smape

def compute_predictions(model, X_test, y_test, results):
    y_predicted = model.predict(X_test)

    results_dict['RMSE'].append(RMSE(y_test, y_predicted))
    print('RMSE {}: '.format(X_test.iloc[0,0]), RMSE(y_test, y_predicted))

    results_dict['MSE'].append(mean_squared_error(y_test, y_predicted))
    print('MSE {}: '.format(X_test.iloc[0,0]), mean_squared_error(y_test, y_predicted))

    results_dict['SMAPE'].append(sMAPE(y_test, y_predicted))
    print('SMAPE {}: '.format(X_test.iloc[0,0]), sMAPE(y_test, y_predicted))


results_dict = {'RMSE': [], 'MSE': [], 'SMAPE': []}

for i in range(train_range, 2009):
    X_test = X[X['Year'] == i]
    y_test = y[y.index.isin(X_test.index)]
    compute_predictions(loaded_model, X_test, y_test, results_dict)

joblib.dump(results_dict, 'results.joblib', compress=3)