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

train_range = 1990

### Local
#X_train = X.iloc[:5000,:]

### For server
X_train = X[(X['Year'] < train_range)]
print(X_train.shape)
y_train = y[y.index.isin(X_train.index)]

def fit_model(X, y, model_name):
    print('Model Fitting started: ', datetime.now())
    start_time = time.time()

    classifier = XGBRegressor(objective='reg:squarederror', n_jobs=8, n_estimators= 1000, verbosity= 3)
    classifier.fit(X, y)
    pickle.dump(classifier, open("models/{}.pickle.dat".format(model_name), 'wb'))

    print('Duration Fitting: ', (time.time() - start_time))

    return classifier


def RMSE(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse

def sMAPE(y_true, y_pred):
    smape = 100 * np.average(np.abs(y_pred - y_true) / ((np.abs(y_pred) + np.abs(y_true))/2))
    return smape

def compute_predictions(model, X_test, y_test, results):
    y_predicted = model.predict(X_test)

    year = X_test.iloc[0,0]
    rmse = RMSE(y_test, y_predicted)
    mse = mean_squared_error(y_test, y_predicted)
    smape = sMAPE(y_test, y_predicted)

    results['Year'].append((year))

    results['RMSE'].append(rmse)
    print('RMSE {}: '.format(year), rmse)

    results['MSE'].append(mse)
    print('MSE {}: '.format(year), mse)

    results['SMAPE'].append(smape)
    print('SMAPE {}: '.format(year), smape)


model_name = 'static_model'

print('Fit model with data from {} to {}'.format(X_train['Year'].min(), X_train['Year'].max()))
model = fit_model(X_train, y_train, model_name)

loaded_model = pickle.load(open("models/{}.pickle.dat".format(model_name),'rb'))


results_dict = {'Year': [], 'RMSE': [], 'MSE': [], 'SMAPE': []}

for i in range(train_range, 2009):
    X_test = X[X['Year'] == i]
    y_test = y[y.index.isin(X_test.index)]
    compute_predictions(loaded_model, X_test, y_test, results_dict)

joblib.dump(results_dict, 'results.joblib', compress=3)