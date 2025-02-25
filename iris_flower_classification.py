#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# In[13]:


iris = pd.read_csv("iris.csv")
print(iris)


# In[15]:


iris.info()


# In[21]:


iris.isna().sum()


# In[29]:


labelencoder = LabelEncoder()
iris.iloc[:, -1] = labelencoder.fit_transform(iris.iloc[:,-1])


# In[ ]:





# In[23]:


iris.isnull().sum()

