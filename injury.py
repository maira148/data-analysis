# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 18:08:54 2022

@author: MYSTERIOUS
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
data=pd.read_csv("C:/Users/MYSTERIOUS/Downloads/serious-injury-outcome-indicators-2000-2020-CSV.csv")

print(data)
data2=data.astype({'Severity':'string','Indicator':'string',
'Cause':'string' ,'Population':'string','Period':'string',
'Validation':'string'})

print(data2)
data3=data2.dtypes
shape=data2.shape
print(shape)
for col in data2.columns:
    print(col)
head=data2.head()  
print(head)
from matplotlib import pyplot as plt
import seaborn as sns
sns.histplot(data2['Age'])
plt.show()
sns.histplot(data2['Severity'])
plt.show()
sns.barplot(data2['Age'],data2['Upper_CI'])
plt.show()
x=data2['Upper_CI']
y=data2['Validation']
x=(np.array(x).reshape(2748,1))
y=(np.array(y).reshape(2748,1))
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=44
  ,test_size=0.3)


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
fit=dt.fit(x_train,y_train)
predict=dt.predict(x_test)
print(predict)
from sklearn.metrics import accuracy_score
accuracy_score=accuracy_score(y_test,predict)*100
print(accuracy_score)
