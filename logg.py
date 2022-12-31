# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 18:10:48 2022

@author: MYSTERIOUS
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error
data=pd.read_csv("D:\cs504\cp-national-datafile.csv")
data1=data['LowerCIB'].dropna(axis=0)
data2=data['UpperCIB'].dropna(axis=0)
print(data)
import seaborn as sns
from matplotlib import pyplot as plt
sns.barplot(data['Year'],data1)
plt.show()
x=data2
y=data1
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,
  random_state=100)
print(x_train,x_test,y_train,y_test)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
x_trainn=np.array(x_train).reshape(309,1)
print(x_trainn)
y_trainn=np.array(y_train).reshape(309,1)
x_testt=np.array(x_test).reshape(133,1)
y_testt=np.array(y_test).reshape(133,1)
print(y_trainn)
fit=lr.fit(x_trainn,y_trainn)
print(fit)
y_pred=lr.predict(x_testt)
print(y_pred)
pre=np.array(y_pred).reshape(133,1)

error=mean_absolute_error(y_testt,y_pred)
print(error)   

data4=data['SE'].dropna(axis=0)
data3=data['LowerCIB'].dropna(axis=0)
x=data3
y=data4

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,
random_state=35 )
print (x_train,x_test,y_train,y_test)
y_trainn1=np.array(y_train).reshape(309,1)
x_trainn1=np.array(x_train).reshape(309,1)
x_testt1=np.array(x_test).reshape(133,1)
y_testt1=np.array(y_test).reshape(133,1)

lr1=LinearRegression()
fit1=lr1.fit(x_trainn1,y_trainn1)
print(fit1)
y_pred1=lr1.predict(x_testt1)
print(y_pred1)
error1=mean_absolute_error(y_testt1,y_pred1)
print(error1)  

data5=data['SE'].dropna(axis=0)
data6= data['UpperCIB'].dropna(axis=0)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,
random_state=35 )
print (x_train,x_test,y_train,y_test)
y_trainn2=np.array(y_train).reshape(309,1)
x_trainn2=np.array(x_train).reshape(309,1)
x_testt2=np.array(x_test).reshape(133,1)
y_testt2=np.array(y_test).reshape(133,1)
lr2=LinearRegression()
fit2=lr2.fit(x_trainn2,y_trainn2)
print(fit2)
y_pred2=lr2.predict(x_testt2)
print(y_pred2)
error2=mean_absolute_error(y_testt2,y_pred2)
print(error1)





