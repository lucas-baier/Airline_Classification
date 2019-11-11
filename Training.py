import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import time
from datetime import datetime
import pickle
import gc

start_time = time.time()

data = pd.read_csv('airline_14col.data', delimiter=',', header=None)

end_time = time.time()
duration = end_time - start_time

print('Duration LoadingData: ', (end_time-start_time))

column_names = ['Year','Month','DayofMonth','DayofWeek','CRSDepTime','CRSArrTime','UniqueCarrier',
           'FlightNum','ActualElapsedTime','Origin','Dest','Distance','Diverted','ArrDelay']

data.columns = column_names

data_short = data[data['Year'] < 1989]


def preprocessing(data):
    start_time = time.time()

    # One-Hot Encoding
    data_encoded = pd.get_dummies(data[['UniqueCarrier', 'Origin', 'Dest']])

    data_reduced = data.drop(['UniqueCarrier', 'Origin', 'Dest', 'FlightNum', 'Diverted'], axis=1)
    data_reduced['ArrDelay'][data_reduced['ArrDelay'] < 0] = 0
    data_full = pd.concat([data_reduced, data_encoded], axis=1)

    y = np.array(data_full['ArrDelay'])
    X = np.array(data_full.drop(['ArrDelay'], axis=1))

    end_time = time.time()
    duration = end_time - start_time

    print(data_encoded.info())
    print(data_full.info())
    print(data_reduced.info())

    del data_reduced
    del data_full
    del data_encoded

    gc.collect()

    print('Duration Preprocessing: ', duration)

    return X, y


X_train, y_train = preprocessing(data_short)


def fit_model(X, y):
    print('Model Fitting started: ', datetime.now())
    start_time = time.time()

    classifier = XGBRegressor(objective='reg:squarederror', n_jobs=-1)
    classifier.fit(X_train, y_train)
    pickle.dump(classifier, open("staticModel.pickle.dat", 'wb'))

    end_time = time.time()
    duration = end_time - start_time

    print('Duration Fitting: ', (end_time - start_time))

    return classifier

model = fit_model(X_train, y_train)

loaded_model = pickle.load(open("staticModel.pickle.dat",'rb'))
loaded_model.predict(X_train).shape