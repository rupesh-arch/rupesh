#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
from sklearn.svm import SVC 
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score,StratifiedKFold


# In[12]:


dataframe = pd.read_csv('diabetes.csv')
dataframe


# In[14]:


array =dataframe.values
X = array[:,0:8]
Y = array[:,8]


# In[16]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify = Y)


# In[20]:


X_train.shape, Y_train.shape, X_test.shape, Y_test.shape


# In[22]:


clf = SVC()
clf.fit(X_train,Y_train)


# In[24]:


Y_predict = clf.predict(X_test)


# In[ ]:





# In[26]:


print(classification_report(Y_test,Y_predict))


# In[32]:


print(classification_report(Y_train, clf.predict(X_train)))


# #### Hyper Parameter Tuning with Randomized Grid Search

# In[37]:


clf = SVC()
param_grid = [{'kernel':['linear','rbf'],'gamma':[0.1,0.5,1],'C':[0.1,1,10]}]
kfold = StratifiedKFold(n_splits=5)
gsv = RandomizedSearchCV(clf,param_grid,cv=kfold,scoring= 'recall')
gsv.fit(X_train,Y_train)

