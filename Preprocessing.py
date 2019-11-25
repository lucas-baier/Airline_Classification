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

# h = hpy()

# start_time = time.time()
#
# data = dd.read_csv('airline_14col.data', delimiter=',', header=None)
# #data = pd.read_csv('airline_14col.data', delimiter=',', header=None)
#
# end_time = time.time()
# duration = end_time - start_time
#
# print('Duration LoadingData: ', (end_time-start_time))
#
# column_names = ['Year','Month','DayofMonth','DayofWeek','CRSDepTime','CRSArrTime','UniqueCarrier',
#            'FlightNum','ActualElapsedTime','Origin','Dest','Distance','Diverted','ArrDelay']
#
# data.columns = column_names
# print(data.shape)
#
# #data_short = data[data['Year'] < 1989]
# data_short = data
#
# data_ORD = data_short[data_short['Origin'] == 'ORD']
# #print(data_ORD.shape)
#
# #Pring current memory usage
# #print(h.heap())
#
# def preprocessing(data):
#     print('Preprocessing started!')
#     start_time = time.time()
#
#     # One-Hot Encoding
#     data['DayofWeek'] = data['DayofWeek'].astype('category')
#     data_encoded = dd.get_dummies(data[['UniqueCarrier', 'Origin', 'Dest', 'DayofWeek']].categorize()).compute()
#     print('Data enocded: ', (time.time()-start_time))
#
#     data_reduced = data.drop(['UniqueCarrier', 'Origin', 'Dest', 'FlightNum', 'Diverted','DayofWeek'], axis=1).compute()
#     print('Data reduced: ', (time.time() - start_time))
#
#     X = pd.concat([data_reduced, data_encoded], axis=1)
#     print('Data concatenated: ', (time.time() - start_time))
#
#
#     #y[y<0] = 0
#
#
#     end_time = time.time()
#     duration = end_time - start_time
#
#     # print(data_encoded.info())
#     # print(data_full.info())
#     # print(data_reduced.info())
#     #
#     # print(h.heap())
#
#     del data_reduced
#     del data_encoded
#
#     gc.collect()
#
#     #print('Afer Deletion:', h.heap())
#
#     print('Duration Preprocessing: ', duration)
#
#     return X
#
#
# data_all = preprocessing(data_ORD)
#
#
# #Save files
#
# joblib.dump(data_all, 'data_ORD.joblib', compress=3)
#
# data.loc[data['ArrDelay'] < 0, 'ArrDelay'] = 0
#
# # Use data after 1990 since sparse data for years 1987 and 1989
# data = data[data.Year >= 1990]
#
# datetime_list = []
#
# for i in range(data.shape[0]):
#     stamp = pd.Timestamp(year=data['Year'].iloc[i], month=data['Month'].iloc[i], day=data['DayofMonth'].iloc[i])
#     datetime_list.append(pd.to_datetime(stamp))
#
# data['Date'] = datetime_list
#
# joblib.dump(data, 'data_ORD_date.joblib', compress=3)
#
#
# #data = data.reset_index(drop = True)
#
#
# ### Sample for local processing
# #data = data.sample(n = 25000, replace = False, random_state=42)
# #data = data.sort_values(by = ['Date'])



print('Data loading started: ', datetime.now())
start_time = time.time()

data = joblib.load("data_ORD_date.joblib")

#
#
#
# Local processing
# data = joblib.load("sample_data_ORD_date.joblib")

print('Duration Loading: ', (time.time() - start_time))

# Set all delays to 1
data.loc[data['ArrDelay'] > 0, 'ArrDelay'] = 1

joblib.dump(data, "data_ORD_date_class.joblib", compress=3)

data_sample = data.sample(n = 25000, replace = False, random_state=42)
data_sample = data_sample.sort_values(by = ['Date'])

joblib.dump(data_sample, "sample_data_ORD_date_class.joblib", compress=3)