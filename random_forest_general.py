# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 17:32:14 2016

@author: Mangarella
"""

import datetime as dt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier




def random_forest_generalized(X_train, X_test, y_train, y_test, ls_features, criterion, n_estimators, max_depth, max_features, min_samples_leaf, min_samples_split, bootstrap, print_state):

    
    Rf = RandomForestClassifier(criterion = criterion, n_estimators = n_estimators, max_depth = max_depth, min_samples_leaf = min_samples_leaf, min_samples_split = min_samples_split, max_features = max_features, bootstrap = bootstrap)
    Rf.fit(X_train[ls_features], y_train)
       
    #Evaluation on train data set
    train_prediction_rate = Rf.score(X_train[ls_features], y_train)
    train_prediction = Rf.predict(X_train[ls_features])
    if print_state == True:    
        print "Positive Response % in train data = " + str(round(np.mean(y_train)*100,3))
        print "Train Prediction Rate = " + str(round(train_prediction_rate, 3))
      
    #Actual stats for train data
    number_of_pos = np.floor(np.mean(y_train)*float(len(y_train)))
    number_of_neg = np.floor((1-np.mean(y_train))*float(len(y_train)))
    
    #Stats for train prediction 
    number_of_correct = train_prediction_rate*float(len(X_train))
    number_of_incorrect = (1-train_prediction_rate)*float(len(X_train))
    
    #Number of predicted events
    number_of_pred_pos_train = float(np.mean(train_prediction))*float(len(train_prediction))
    number_of_pred_neg_train = (1-float(np.mean(train_prediction)))*float(len(train_prediction))
    
    #Create confusion matrix, I find this easier than using the confusion matrix in scikit
    error_matrix = np.floor(train_prediction*2 - y_train)
    correct_neg_count_train = float(len(error_matrix[error_matrix == 0]))
    correct_pos_count_train = float(len(error_matrix[error_matrix == 1]))
    false_neg_count_train = float(len(error_matrix[error_matrix == -1]))
    false_pos_count_train = float(len(error_matrix[error_matrix == 2]))
    if print_state == True:    
        print "Positive Response % in train pred = " + str(round(np.mean(train_prediction)*100,3))
        print "Correct Negative   % = " + str(round(correct_neg_count_train/number_of_pred_neg_train*100,3))
        print "Correct Positive   % = " + str(round(correct_pos_count_train/number_of_pred_pos_train*100,3))
        print "Type I error  (FP) % = " + str(round(false_pos_count_train/number_of_pred_pos_train*100,3))
        print "Type II error (FN) % = " + str(round(false_neg_count_train/number_of_pred_neg_train*100,3))
    
    #Now validate with test data
    test_prediction_rate = Rf.score(X_test[ls_features], y_test)
    test_prediction = Rf.predict(X_test[ls_features])
    if print_state == True:    
        print "Test Prediction Rate = " + str(round(test_prediction_rate, 3))
        print "Positive Response % in test data = " + str(round(np.mean(y_test)*100,3))
        print "Positive Response % in test pred = " + str(round(np.mean(test_prediction)*100,3))
        
    #Number of predicted events
    number_of_pred_pos_test = np.mean(test_prediction)*float(len(test_prediction))
    number_of_pred_neg_test = (1 - np.mean(test_prediction))*float(len(test_prediction))
    error_matrix = np.floor(test_prediction*2 - y_test)
    correct_neg_count_test = float(len(error_matrix[error_matrix == 0]))
    correct_pos_count_test = float(len(error_matrix[error_matrix == 1]))
    false_neg_count_test = float(len(error_matrix[error_matrix == -1]))
    false_pos_count_test = float(len(error_matrix[error_matrix == 2]))
    if print_state == True:    
        print "Correct Negative   % = " + str(round(correct_neg_count_test/number_of_pred_neg_test*100,3))
        print "Correct Positive   % = " + str(round(correct_pos_count_test/number_of_pred_pos_test*100,3))
        print "Type I error  (FP) % = " + str(round(false_pos_count_test/number_of_pred_pos_test*100,3))
        print "Type II error (FN) % = " + str(round(false_neg_count_test/number_of_pred_neg_test*100,3))
        print "Recall               = " + str(round(correct_pos_count_test/(correct_pos_count_test+false_neg_count_test)*100,3))        
        
    prediction_recall = round(np.mean(test_prediction)*100,3) 
    correct_pos_rate = round(correct_pos_count_test/number_of_pred_pos_test*100,3)
    
    return Rf, test_prediction, prediction_recall, correct_pos_rate, error_matrix