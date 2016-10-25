# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 17:07:48 2016

@author: Mangarella
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
from sklearn.cross_validation import KFold
from sklearn.grid_search import ParameterSampler
from multiprocessing import Pool

from randomforest_general import *
from dataloader import *
from featureloader import *
from trigger_percent_change import *




def parallel_search_rf(X_data, y_data, ls_features, n_iter):
        
    
    param_dist = {"criterion" : ["gini", "entropy"],
                  "min_samples_leaf" : randint(1, 50),
                  "min_samples_split" : randint(2, 50),
                  "max_depth" : randint(5, 30),
                  "max_features" : ["sqrt", "log2", None],
                  "n_estimators" : randint(200, 600),
                  "bootstrap" : [True, False]}
    param_list = list(ParameterSampler(param_dist, n_iter=n_iter))
    
    #Change cores as necessary, one less than total cores is preferable to perform some other functions
    pool = Pool(processes = 4)
    results = [pool.apply_async(randomized_search_rf, args = (X_data, y_data, ls_features, params)) for params in param_list]
    answer = [p.get() for p in results]
    return answer

    
def randomized_search_rf(X_data, y_data, ls_features, params):  
    
        
    #Pick hyperparameters
    criterion = params["criterion"]
    max_depth = params["max_depth"]
    max_features = params["max_features"]
    min_samples_leaf = params["min_samples_leaf"]
    min_samples_split = params["min_samples_split"]
    n_estimators = params["n_estimators"]
    bootstrap = params["bootstrap"]
    
    number_of_events = len(X_data)    
    kf = KFold(number_of_events, n_folds = 3, shuffle = True)
    precision = []
    recall = []

    #randomforest function state to restrict printing of extra metrics
    print_state = False 
    
    for train_index, test_index in kf:
        X_train, X_test = X_data.ix[train_index], X_data.ix[test_index]
        y_train, y_test = y_data.ix[train_index], y_data.ix[test_index]
        Rf, test_prediction, prediction_recall, correct_pos_rate, error_matrix = random_forest_generalized(X_train, X_test, y_train, y_test, ls_features, criterion, n_estimators, max_depth, max_features, min_samples_leaf, min_samples_split, bootstrap, print_state)
        
        #Collect all precision and custom recall stats     
        precision.append(correct_pos_rate)
        recall.append(prediction_recall)

    #Store metrics for each hyperparameter
    ave_precision = np.mean(precision)
    ave_recall = np.mean(recall)
    randomizedsearch_scores = (ave_precision, ave_recall, params)
    return randomizedsearch_scores
    
if __name__ == '__main__':
    search_params = parallelSearch_RF(X_data, y_data, ls_features, 20)
    