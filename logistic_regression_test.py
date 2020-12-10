#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 14:59:27 2020
classification (log_reg) model
@author: tahereh tekieh
"""

import numpy as np
import pandas as pd
from seaborn import heatmap

# Read the data, get info  
data = pd.read_csv('Social_Network_Ads.csv')
heatmap(data.isnull())
data.info()
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

# Split the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)

# Feature scale
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)

# Train the model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train,y_train)

# Predict a result
y_pred = classifier.predict(x_test)
print( np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1) )


# Evaluate the model
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)
acc_score = accuracy_score(y_test,y_pred)

# Visualising the Training set results
import  matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
x_set, y_set = sc_x.inverse_transform(x_train), y_train
X1, X2 = np.meshgrid(np.arange( x_set[:, 0].min() - 10,  x_set[:, 0].max() + 10,  0.25),
                     np.arange( x_set[:, 1].min() - 1000,  x_set[:, 1].max() + 1000,  0.25))
plt.contourf(X1, X2, classifier.predict(sc_x.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# Visualising the Testing set results
import  matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
x_set, y_set = sc_x.inverse_transform(x_test), y_test
X1, X2 = np.meshgrid(np.arange( x_set[:, 0].min() - 10,  x_set[:, 0].max() + 10,  0.25),
                     np.arange( x_set[:, 1].min() - 1000,  x_set[:, 1].max() + 1000,  0.25))
plt.contourf(X1, X2, classifier.predict(sc_x.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Testing set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()














