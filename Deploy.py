#!/usr/bin/env python
# coding: utf-8

# In[1]:


#library
import pandas as pd
import numpy as np
import joblib 
from sklearn.linear_model import LinearRegression


# In[2]:


#dataset 
df = pd.read_csv("D:/coobaa.csv")


# In[3]:


#Library membuat model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,r2_score
from sklearn.model_selection import train_test_split


# In[5]:


X=df.iloc[:,2:8].values
y=df.iloc[:,1].values


# In[6]:


#Train and Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[7]:


#call model regression
X = np.asanyarray(df[['Luas_hutan']])
Y = np.asanyarray(df[['Intensitas_em']])
model = LinearRegression().fit(X,Y)
model


# In[8]:


#save model
filename = 'model.sav'
joblib.dump(model, filename)


# In[9]:


#load model
loaded_model = joblib.load(filename)


# In[10]:


#prediction model
loaded_model.predict(np.array([10]).reshape(1, 1))

