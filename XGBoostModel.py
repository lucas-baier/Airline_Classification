import pandas as pd
import numpy as np
 

import sklearn.preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

import itertools

import os

import math

import gc

import pickle
import time
from datetime import datetime

import xgboost as xgb

#import SKlearn Wrapper:
from xgboost.sklearn import XGBRegressor



class XGBoostModel():
    
    
    def __init__(self, objective ='reg:squarederror', n_estimators = 1000, n_jobs=8, random_state = 123, verbosity=0, retraining_memory_save_mode = False):
        

        
        '''
        Variable explanation:
            >> objective = learning objective / objective function to use for learning task
            
            >> n_estimators = Number of trees to fit
        
            >> learning_rate: Boosting learning_rate for training the model 
        
            
            >> reg_alpha: weight for L1 Regularization

            
        '''        

                        
        self.objective = objective
        
        self.n_estimators = n_estimators

        self.n_jobs = n_jobs
        self.random_state = random_state

        self.verbosity = verbosity
        
        self.prediction_model = None
        self.actuals = None
        
        self.model_name = None
        self.training_history = None

        self.results = {'Year': [], 'RMSE': [], 'MSE': [], 'SMAPE': [], 'Predictions': []}

        
     

    def get_params(self):
        '''
        Returns all parameters of model
        '''
        
        param_dict = {"objective" : self.objective,
                      "n_estimators" : self.n_estimators,
                      "n_jobs" : self.n_jobs,
                      "random_state" : self.random_state,
                      "verbosity" : self.verbosity,
                      "model_name" : self.model_name,
                      "prediction_model" : self.prediction_model,
                      "actuals" : self.actuals
                     }
                     
        return param_dict
    
  
   
    def load_model(self, model_to_load, model_name = '_no_name_given'):
        
        '''
        function "loads" pre-trained model which was stored on disk into Class
        
        Note: this only works if params of Class are the same as params used to train "model_to_load"
        
        '''
        
        self.prediction_model = model_to_load
        self.model_name = model_name

    
    def generate_data(self, data, start_train_year, last_train_set_year, start_test_set_year,
                      end_test_set_year, verbose=0):
        
        
        '''
        function creates training data for model:
            several functions are called to get correct shape of data and corresponding features
        
            after that, data is split into training data/ valid data / test data based on given input dates/"years"
            
        '''

        
        if verbose == 1:
            print('generate data..')
            print('start_train_year: ', start_train_year)
            print('last_train_set_year: ', last_train_set_year)

            print('start_test_set_year: ', start_test_set_year)
            print('end_test_set_year: ', end_test_set_year)



        # 2) get Train/Test-Split f


        # create train/test split

        data_train = data.loc[(data['Year'] >= start_train_year) & (data['Year'] <= last_train_set_year)]
        data_test = data.loc[(data['Year'] >= start_test_set_year) & (data['Year'] <= end_test_set_year)]

        print(data_train.shape)

        print(data_test.shape)

        y_train = data_train.loc[:,'ArrDelay']
        y_test = data_test.loc[:,'ArrDelay']

        X_train = data_train.drop(['ArrDelay'], axis=1)
        X_test = data_test.drop(['ArrDelay'], axis=1)

        gc.collect()
        
        return X_train, y_train, X_test, y_test




    def fit_model(self, X_train, y_train):

        print('Model Fitting started: ', datetime.now())
        print('Fit model with data from {} to {}'.format(X_train['Year'].min(), X_train['Year'].max()))

        model_name = 'static_model_{}_{}'.format(X_train['Year'].min(), X_train['Year'].max())

        start_time = time.time()

        regressor = XGBRegressor(objective='reg:squarederror', n_jobs=8, n_estimators= 1000, verbosity= 3)
        regressor.fit(X_train, y_train)
        pickle.dump(regressor, open("models/{}.pickle.dat".format(model_name), 'wb'))

        print('Duration Fitting: ', (time.time() - start_time))

        self.prediction_model = regressor





    def RMSE(self, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return rmse

    def sMAPE(self, y_true, y_pred):
        smape = 100 * np.average(np.abs(y_pred - y_true) / ((np.abs(y_pred) + np.abs(y_true)) / 2))
        return smape



    def compute_predictions(self, X_test, y_test):

        y_predicted = self.prediction_model.predict(X_test)

        results = self.results

        year = X_test.iloc[0, 0]
        rmse = self.RMSE(y_test, y_predicted)
        mse = mean_squared_error(y_test, y_predicted)
        smape = self.sMAPE(y_test, y_predicted)

        results['Year'].append((year))

        results['RMSE'].append(rmse)
        print('RMSE {}: '.format(year), rmse)

        results['MSE'].append(mse)
        print('MSE {}: '.format(year), mse)

        results['SMAPE'].append(smape)
        print('SMAPE {}: '.format(year), smape)

        results['Predictions'].append(y_predicted)

        self.results = results

        return results

