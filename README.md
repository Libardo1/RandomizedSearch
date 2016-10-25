# RandomizedSearch
Randomized Search that returns precision and recall score



Scikit's randomizedsearch is unable to return both a precision and recall score. 
Functions below take a X_data dataframe, y_data Series, and list of X_data columns to be used as features, 
and returns tuples of precision, custom recall (simple % predicted as Class 1), and randomized hyperparameters.

RandomizedSearch is used over GridSearchCV due to literature showing increased speed. 
Stratified Kfold is used as CV. 
Parallelization is accomplished via Pool.
