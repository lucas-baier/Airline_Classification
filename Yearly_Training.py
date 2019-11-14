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
data = joblib.load("data_ORD.joblib")
print('Duration Loading: ', (time.time() - start_time))

data = data.reset_index(drop = True)

data.loc[data['ArrDelay'] < 0, 'ArrDelay'] = 0

### Local
#data = data.iloc[:5000,:]
#data = data.iloc[60000:80000,]


xgboost_model = XGBoostModel()

# X_train, y_train, X_test, y_test = xgboost_model.generate_data(data, 1987, 1987, 1987, 1987, verbose=1)
#
# xgboost_model.fit_model(X_train, y_train)
#
# results_dict = xgboost_model.compute_predictions(X_test, y_test)
# print(results_dict)

for i in range(1987, 2007):

    print(i, i+1, i+2, i+2)

    X_train, y_train, X_test, y_test = xgboost_model.generate_data(data, i, i+1, i+2, i+2, verbose = 1 )

    xgboost_model.fit_model(X_train, y_train)

    results_dict = xgboost_model.compute_predictions(X_test, y_test)
    print(results_dict)


joblib.dump(results_dict, 'yearly_static_results_all.joblib', compress=3)