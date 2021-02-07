#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 17:21:59 2021
The data contains the MPG (Mile Per Gallon) variable which is continuous 
data and tells us about the efficiency of fuel consumption of a vehicle in the
 70s and 80s. Data available at :
"http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"

Our aim here is to predict the MPG value for a vehicle, given that we have 
other attributes of that vehicle.

@author: Tahereh Tekieh
"""

import numpy as np 
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

# 1- Data Preprocessing and Exploration

# Define the column names
cols = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
# Read the data file
data = pd.read_csv('auto-mpg.csv',names = cols,
                   na_values = "?",
                   comment = '\t',
                   sep = ' ',
                   skipinitialspace=True)
# Make a copy of the dataframe
df = data.copy()

# Explore the data
data. head()
data.info()
sb.heatmap(data.isnull())
sb.boxplot(data['Horsepower'])

# Impute the missing values by the median
median = data['Horsepower'].median()
data['Horsepower'] = data['Horsepower'].fillna(median)
data.info()

#category distribution
print(data["Cylinders"].value_counts() / len(data))
print(data['Origin'].value_counts())
sb.pairplot(data)

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
X = data.iloc[:,1:].values
y = data['MPG']

ct1 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0,6])], remainder='passthrough')
X = np.array(ct1.fit_transform(X))



# Split the test and train data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=24)


# Feature Scale the independent variable
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train[:, :] = sc_x.fit_transform(x_train[:, :])
x_test[:, :] = sc_x.transform(x_test[:, :])


# 2- Data Modelling

# Try linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_pred)

print(np.sqrt(mse))

# Cross Validate
from sklearn.model_selection import cross_val_score

scores = cross_val_score(regressor, 
                         x_train, 
                         y_train, 
                         scoring="neg_mean_squared_error", 
                         cv = 10)
print( np.sqrt(-scores))


# Try Random Forest
from sklearn.ensemble import RandomForestRegressor
regressor_rf = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor_rf.fit(x_train,y_train)
y_predrf = regressor_rf.predict(x_test)
mse_rf = mean_squared_error(y_test, y_predrf)
print(np.sqrt(mse_rf))

# Cross Validate
scores_rf = cross_val_score(regressor_rf, x_train,y_train,
                            scoring="neg_mean_squared_error", 
                         cv = 10)
print( np.sqrt(-scores_rf))





































