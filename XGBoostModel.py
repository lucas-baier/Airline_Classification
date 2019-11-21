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
    
    
    def __init__(self, objective ='reg:squarederror', n_estimators = 1000, n_jobs=8, random_state = 123, verbosity=0,
                 strategy_name = 'Static'):
        

        
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
        
        self.strategy_name = strategy_name
        self.training_history = None

        self.results = {'Year': [], 'Start_Test': [], 'End_Test': [], 'Date': [],
                        'RMSE': [], 'MSE': [], 'SMAPE': [], 'Predictions': [], 'y_true': []}

        
     

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

    
    def generate_data(self, data, start_train_date, last_train_date, start_test_date,
                      end_test_date, verbose=0):
        
        
        '''
        function creates training data for model:
            several functions are called to get correct shape of data and corresponding features
        
            after that, data is split into training data/ valid data / test data based on given input dates/"years"
            
        '''
        end_train_date = start_train_date + pd.DateOffset(years=2)


        if verbose == 1:
            print('generate data..')
            print('start_train_year: ', start_train_date)
            print('last_train_set_year: ', end_train_date)

            print('start_test_set_year: ', start_test_date)
            print('end_test_set_year: ', end_test_date)



        # 2) get Train/Test-Split f


        # create train/test split

        data_train = data.loc[(data['Date'] >= start_train_date) & (data['Date'] < end_train_date)]
        data_test = data.loc[(data['Date'] >= start_test_date) & (data['Date'] < end_test_date)]

        print(data_train.shape)

        print(data_test.shape)

        y_train = data_train.loc[:,'ArrDelay']
        y_test = data_test.loc[:,'ArrDelay']

        X_train = data_train.drop(['ArrDelay', 'Date'], axis=1)
        X_test = data_test.drop(['ArrDelay', 'Date'], axis=1)

        gc.collect()
        
        return X_train, y_train, X_test, y_test




    def fit_model(self, X_train, y_train):

        print('Model Fitting started: ', datetime.now())

        start_train_date = pd.Timestamp(year=X_train['Year'].iloc[0], month=X_train['Month'].iloc[0],
                                       day=X_train['DayofMonth'].iloc[0]).date()

        end_train_date = pd.Timestamp(year=X_train['Year'].iloc[-1], month=X_train['Month'].iloc[-1],
                                       day=X_train['DayofMonth'].iloc[-1]).date()

        print('Fit model with data from {} to {}'.format(start_train_date, end_train_date))

        model_name = '{}_{}_{}_{}_{}'.format(self.strategy_name, start_train_date.year, start_train_date.month,
                                             end_train_date.year, end_train_date.month)

        start_time = time.time()

        regressor = XGBRegressor(objective='reg:squarederror', n_jobs=8, n_estimators= 1000, verbosity= 1)
        regressor.fit(X_train, y_train)
        pickle.dump(regressor, open("models/{}.pickle.dat".format(model_name), 'wb'))

        print('Duration Fitting: ', (time.time() - start_time))

        self.prediction_model = regressor


    def update_model(self, X_train, y_train):

        print('Model Updating started: ', datetime.now())

        start_train_date = pd.Timestamp(year=X_train['Year'].iloc[0], month=X_train['Month'].iloc[0],
                                       day=X_train['DayofMonth'].iloc[0]).date()

        end_train_date = pd.Timestamp(year=X_train['Year'].iloc[-1], month=X_train['Month'].iloc[-1],
                                       day=X_train['DayofMonth'].iloc[-1]).date()

        print('Update model with data from {} to {}'.format(start_train_date, end_train_date))

        model_name = '{}_{}_{}_{}_{}'.format(self.strategy_name, start_train_date.year, start_train_date.month,
                                             end_train_date.year, end_train_date.month)

        Save_PATH = 'models/temp_boosting_models/'

        start_time = time.time()

        if self.prediction_model == None:
            print('self.predicion_model = None')

            # load existing model from disc:

            # load model:
            file_to_load = Save_PATH + "Quarterly_Retraining_1990_1_1991_12" + '.pickle.dat'
            # load model into dict:
            prediction_model = pickle.load(open(file_to_load, "rb"))
            self.prediction_model = prediction_model

        else:
            # store current model:
            final_model_name = model_name + '.pickle.dat'
            file_to_save = Save_PATH + final_model_name
            # save model on disk:
            pickle.dump(self.prediction_model, open(file_to_save, "wb"))

            print('self.predicion_model != None')
            # delete model to release memory:
            # set self.prediction_model to None:
            print('Delete model')
            self.prediction_model._Booster.__del__()
            self.prediction_model = None

            print('>> prediciton model is loaded from disk')
            print('Model to load: ', model_name)
            # load existing model from disc:

            file_to_load = Save_PATH + model_name + '.pickle.dat'  # self.model_name stores name of previous model!
            # load model into dict:
            prediction_model = pickle.load(open(file_to_load, "rb"))

            current_booster_obj = prediction_model.get_booster()
            print('Multivar XGBoost Model is updated..')
            updated_model = prediction_model.fit(X_train, y_train, xgb_model=current_booster_obj)
            self.prediction_model = updated_model

            # Delete model to release memory
            #updated_model._Booster.__del__()
            #del updated_model
            #gc.collect()

            # store current model:
            final_model_name = model_name + '.pickle.dat'
            file_to_save = Save_PATH + final_model_name
            # save model on disk:
            pickle.dump(self.prediction_model, open(file_to_save, "wb"))

            print('Duration Fitting: ', (time.time() - start_time))



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

        start_test_date = pd.Timestamp(year=X_test['Year'].iloc[0], month=X_test['Month'].iloc[0],
                                       day=X_test['DayofMonth'].iloc[0]).date()

        end_test_date = pd.Timestamp(year=X_test['Year'].iloc[-1], month=X_test['Month'].iloc[-1],
                                       day=X_test['DayofMonth'].iloc[-1]).date()

        def create_date(row_year, row_month, row_day):
            data = pd.Timestamp(year=row_year, month=row_month, day=row_day).date()
            return data

        date = X_test.apply(lambda row: create_date(row['Year'], row['Month'], row['DayofMonth']), axis=1)

        results['Year'].append((year))
        results['Start_Test'].append(start_test_date)
        results['End_Test'].append(end_test_date)
        results['Date'].append(date)
        results['y_true'].append(y_test)

        results['RMSE'].append(rmse)
        print('RMSE from {} to {}: '.format(start_test_date, end_test_date), rmse)

        results['MSE'].append(mse)
        print('MSE from {} to {}: '.format(start_test_date, end_test_date), mse)

        results['SMAPE'].append(smape)
        print('SMAPE from {} to {}: '.format(start_test_date, end_test_date), smape)

        results['Predictions'].append(y_predicted)

        self.results = results

        return results

