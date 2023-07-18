#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("diabetes.csv")
df


# In[3]:


df['Glucose'].replace(0,np.nan, inplace = True)
df['BloodPressure'].replace(0,np.nan, inplace = True)
df['SkinThickness'].replace(0,np.nan, inplace = True)
df['Insulin'].replace(0,np.nan, inplace = True)
df['BMI'].replace(0,np.nan, inplace = True)
df['Age'].replace(0,np.nan, inplace = True)
df


# In[4]:


df.isnull().sum()


# In[5]:


mean_gl = df['Glucose'].astype('float').mean(axis=0)
df['Glucose'].replace(np.nan, mean_gl, inplace = True)

mean_st = df['SkinThickness'].astype('float').mean(axis=0)
df['SkinThickness'].replace(np.nan, mean_st, inplace = True)

mean_bp = df['BloodPressure'].astype('float').mean(axis=0)
df['BloodPressure'].replace(np.nan, mean_bp, inplace = True)

mean_insulin = df['Insulin'].astype('float').mean(axis=0)
df['Insulin'].replace(np.nan, mean_insulin, inplace = True)

mean_bmi = df['BMI'].astype('float').mean(axis=0)
df['BMI'].replace(np.nan, mean_bmi, inplace = True)

mean_age = df['Age'].astype('float').mean(axis=0)
df['Age'].replace(np.nan, mean_age, inplace = True)


# In[6]:


df.isnull().sum()


# In[7]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X = df[['Glucose']]
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[8]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[9]:


new_data_point = [[71]]  
prediction = model.predict(new_data_point)
print("Prediction:", prediction)


# In[10]:


new_data_point = [[151]]  
prediction = model.predict(new_data_point)
print("Prediction:", prediction)


# In[11]:


data = df[["Glucose", "Insulin"]]


# In[12]:


msk=np.random.rand(len(df))<0.8
msk
msk=np.random.rand(len(df))<0.8
train=data[msk]
test=data[~msk]


# In[13]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['Glucose']])
train_y = np.asanyarray(train[['Insulin']])
regr.fit (train_x, train_y)
print ('Coefficients: ', regr.coef_) 
print ('Intercept: ',regr.intercept_)


# In[14]:


import matplotlib.pyplot as plt
plt.scatter(train["Glucose"], train["Insulin"],  color='maroon')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-g')
plt.xlabel("Glucose")
plt.ylabel("Insulin")
plt.show()


# In[15]:


from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['Glucose']]) 
test_y = np.asanyarray(test[['Insulin']]) 
test_y_ = regr.predict(test_x) 

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_) )

