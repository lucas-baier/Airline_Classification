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

print('Model loading started: ', datetime.now())
start_time = time.time()
X_train = joblib.load("X_train.joblib")
y_train = joblib.load("y_train.joblib")
print('Duration Loading: ', (time.time() - start_time))


def fit_model(X, y):
    print('Model Fitting started: ', datetime.now())
    start_time = time.time()

    classifier = XGBRegressor(objective='reg:squarederror', n_jobs=8, n_estimators= 50)
    classifier.fit(X_train, y_train)
    pickle.dump(classifier, open("staticModel.pickle.dat", 'wb'))

    print('Duration Fitting: ', (time.time() - start_time))

    return classifier

model = fit_model(X_train, y_train)

loaded_model = pickle.load(open("staticModel.pickle.dat",'rb'))
y_predicted = loaded_model.predict(X_train)
print(loaded_model.predict(X_train).shape)
print(mean_squared_error(y_train, y_predicted))