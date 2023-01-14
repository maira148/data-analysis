# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 21:30:16 2023

@author: MYSTERIOUS
"""

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
data=pd.read_csv("D:\cs504\heart.csv")

print(data)

for col in data.columns:
    print(col)
head=data.head()
print(head)
tail=head.tail()
print(tail)
shape=data.shape
print(shape)
sex=data['sex'].value_counts()
print(sex)
thalach=data['thalach'].value_counts()
print(thalach)
sns.barplot(data['sex'],data['age'])
plt.show()
sns.histplot(data['age'])
plt.show()
sns.barplot(data['sex'],data['chol'])
plt.show()
x=data['age']
y=data['sex']
x=np.array(x).reshape(1025,1)
y=np.array(y).reshape(1025,1)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=100,
 test_size=0.3)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
fit=lr.fit(x_train,y_train)
predict=lr.predict(x_test)
print(predict)
from sklearn.metrics import mean_squared_error
error=mean_squared_error(y_test,predict)
print(error)
a=data.describe()