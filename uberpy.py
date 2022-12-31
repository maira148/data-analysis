# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 03:41:25 2022

@author: MYSTERIOUS
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error
data=pd.read_csv("D:/cs403/uber-raw-data-apr14.csv")
print(data)
head=data.head()
print(data)
tail=data.tail()
print(tail)
shape=data.shape
print(shape)
import seaborn as sns
from matplotlib import pyplot as plt
x=data['Lat']
y=data['Lon']
sns.histplot(x)
plt.show()
sns.histplot(y)
plt.show()
x=np.array(x).reshape(564516,1)
y=np.array(y).reshape(564516,1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,
random_state=100)
print(x_train,x_test,y_train,y_test)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
fit=lr.fit(x_train,y_train)
print(fit)
y_pred=lr.predict(x_test)
print(y_pred)
error=mean_squared_error(y_test,y_pred)
print(error)
absolute=mean_absolute_error(y_test, y_pred)
print(absolute)
