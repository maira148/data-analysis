import pandas as pd 
import numpy as np
data=pd.read_csv("D:\cs403\heartdata.csv")
print(data)
head=data.head()
print(head)
tail=data.tail()
print(tail)
length=len(data)
print(length)
shape=data.shape
print(shape)
from matplotlib import pyplot as plt
import seaborn as sns
sns.scatterplot(x='smoking', y='heart.disease', data=data,hue='smoking')
plt.show()
x=data[['smoking']]
y=data[['heart.disease']]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
print (x_train,x_test,y_train,y_test)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
fit=lr.fit(x_train, y_train)
predict=lr.predict(x_test )
print(predict)
