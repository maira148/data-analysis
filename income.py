import pandas as pd
s=pd.read_csv('income.data.csv')
print (s)
head=s.head()
print (head)
c=s[s['income']>5]
print (c)
from matplotlib import pyplot as plt
import seaborn as sns
sns.scatterplot(x='income', y='happiness', data=s,hue= 'income')
plt.show()
head=s.head()
print (head)
y=s[['happiness']]
x=s[['income']]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
print (x_train,x_test,y_train,y_test)
a=x_train.head()
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)

lr.fit(x_test,y_test)
ypre=lr.predict(x_test)
print (ypre)
print (y_test)
do=ypre[0:5]
print (do)
from sklearn.metrics import mean_squared_error
error=mean_squared_error(y_test,ypre)
print (error)







 

