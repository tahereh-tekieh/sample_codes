#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 12:52:51 2021
ANN
@author: ttek2642
"""

import numpy as np
import pandas as pd
import tensorflow as tf


# Part 1 - Data Preprocessing

# load and read the data
data = pd.read_excel('/Users/ttek2642/Downloads/ANN/Dataset/Folds5x2_pp.xlsx')
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

# split the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)

# feature scale
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

sc_y = StandardScaler()
y_train = sc_y.fit_transform (y_train.reshape(len(y_train),1))
##y_test= sc_y.transform (y_test.reshape(len(y_test),1))


# Part 2 - Building ANN

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))    
ann.add(tf.keras.layers.Dense(units = 1))


# Part 3 - Training the ANN

ann.compile(optimizer = 'adam',loss = 'mean_squared_error')
ann.fit(x_train,y_train, batch_size = 32, epochs = 20 )


# Part 4 - Prediction and Evaluation of ANN

y_pred = ann.predict(x_test)
y_pred = sc_y.inverse_transform(y_pred)
print(np.concatenate((y_test.reshape(len(y_test),1),y_pred.reshape(len(y_pred),1)),1))



