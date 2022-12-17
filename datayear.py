# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 08:47:22 2022

@author: MYSTERIOUS
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
data=pd.read_csv("D:\cs504\Data7602DescendingYearOrder.csv")

print (data)
length=len(data)
print(length)
shape=data.shape
print(shape)
head=data.head()
print(head)
tail=data.tail()
print(tail)
x=data[['ec_count']]
y=data[['geo_count']]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=100)
print (x_train,x_test,y_train,y_test)
lr=LinearRegression()
fit=lr.fit(x_train, y_train)
print(fit)
y_pred=lr.predict(x_test)

print( y_pred)
error= mean_squared_log_error(y_test,y_pred)
print(error)


