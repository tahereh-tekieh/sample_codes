#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 20:08:04 2021
Breast Cancer Database
XGBOOST method to predict the classification of a 
breast tumor based on its characteristics.
@author: ttek2642
"""
# import the libraries
import numpy as np
import pandas as pd

# load and read data
data = pd.read_csv('/Users/ttek2642/Downloads/Machine Learning A-Z (Codes and Datasets) 13/Part 10 - Model Selection _ Boosting/Section 49 - XGBoost/Python/Data.csv')
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

# splitting train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2 )

# implement XGBoost
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(x_train,y_train)

# evaluate the model
from sklearn.metrics import confusion_matrix,accuracy_score
y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print(cm)
print(accuracy_score(y_test,y_pred))

# cross validation
from sklearn.model_selection import cross_val_score
accuracies= cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)
print('Accuracy: {:.2f} %'.format(accuracies.mean()*100))
print('Standard Deviation: {:.2f} %'.format(accuracies.std()*100))