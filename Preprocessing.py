import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import time
from datetime import datetime
import pickle
import gc
# from guppy import hpy
import dask
import dask.dataframe as dd
import joblib

# h = hpy()

start_time = time.time()

data = dd.read_csv('airline_14col.data', delimiter=',', header=None)
#data = pd.read_csv('airline_14col.data', delimiter=',', header=None)

end_time = time.time()
duration = end_time - start_time

print('Duration LoadingData: ', (end_time-start_time))

column_names = ['Year','Month','DayofMonth','DayofWeek','CRSDepTime','CRSArrTime','UniqueCarrier',
           'FlightNum','ActualElapsedTime','Origin','Dest','Distance','Diverted','ArrDelay']

data.columns = column_names
print(data.shape)

data_short = data[data['Year'] < 1989]

data_ORD = data_short[data_short['Origin'] == 'ORD']
#print(data_ORD.shape)

#Pring current memory usage
#print(h.heap())

def preprocessing(data):
    print('Preprocessing started!')
    start_time = time.time()

    # One-Hot Encoding
    data_encoded = dd.get_dummies(data[['DayofWeek','UniqueCarrier', 'Origin', 'Dest']].categorize()).compute()
    print('Data enocded: ', (time.time()-start_time))

    data_reduced = data.drop(['UniqueCarrier', 'Origin', 'Dest', 'FlightNum', 'Diverted', 'DayofWeek'], axis=1).compute()
    y = data_reduced['ArrDelay']
    y[y<0] = 0
    print(y.shape)

    data_reduced = data_reduced.drop(['ArrDelay'], axis=1)
    print('Data reduced: ', (time.time() - start_time))

    X = pd.concat([data_reduced, data_encoded], axis=1)
    print('Data concatenated: ', (time.time() - start_time))


    #y[y<0] = 0


    end_time = time.time()
    duration = end_time - start_time

    # print(data_encoded.info())
    # print(data_full.info())
    # print(data_reduced.info())
    #
    # print(h.heap())

    del data_reduced
    del data_encoded

    gc.collect()

    #print('Afer Deletion:', h.heap())

    print('Duration Preprocessing: ', duration)

    return X, y


X_train, y_train = preprocessing(data_ORD)


#Save files

joblib.dump(X_train, 'X_train.joblib', compress=3)
joblib.dump(y_train, 'y_train.joblib', compress=3)

