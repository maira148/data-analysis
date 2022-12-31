# -*- coding: utf-8 -*-
"""

Created on Fri Dec  9 05:41:38 2022

@author: MYSTERIOUS
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score,mean_absolute_error
data=pd.read_csv("C:/Users/MYSTERIOUS/Downloads/heartdata.csv")
print(data)
head=data.head()
print(head)
tail=data.tail()
print(tail)
shape=data.shape
print(shape)
import seaborn as sns
from matplotlib import pyplot as plt
sns.barplot(data['heart.disease'],data['smoking'])
plt.show()
sns.barplot(data['heart.disease'],data['biking'])
plt.show()
x=data['heart.disease']
y=data['biking']
x=np.array(x).reshape(498,1)
y=np.array(y).reshape(498,1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,
 random_state=100)
print(x_train,x_test,y_train,y_test)
lr=LinearRegression()
fit=lr.fit(x_train,y_train)
print(fit)
predict=lr.predict(x_test)
print(predict)
error=mean_absolute_error(y_test, predict)
print(error)
