#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[5]:


data1 = pd.read_csv("NewspaperData.csv")
data1.head()


# In[7]:


data1.info()


# In[9]:


data1.isnull().sum()


# In[17]:


plt.figure(figsize=(6,3))
plt.title("Box plot for Daily sales")
plt.boxplot(data1["daily"], vert = False)
plt.show()


# In[19]:


sns.histplot(data1['daily'], kde = True,stat='density',)
plt.show()


# In[21]:


sns.histplot(data1['sunday'], kde = True,stat='density',)
plt.show()


# In[23]:


sns.histplot(data1['Newspaper'], kde = True,stat='density',)
plt.show()


# #### observations
# . The are no missing values
# . The daily column values appear to be right-skewed
# . The sunday column vqalues also appear to be right- skewed
# . There are two outliera in both daily column and also in sunday column as observed from the 

# In[28]:


x= data1["daily"]
y = data1["sunday"]
plt.scatter(data1["daily"], data1["sunday"])
plt.xlim(0, max(x) + 100)
plt.ylim(0, max(y) + 100)
plt.show()


# In[32]:


data1["daily"].corr(data1["sunday"])


# In[34]:


data1[["daily","sunday"]].corr()


# In[36]:


data1.corr(numeric_only=True)


# #### observations on correlation strength
# .The realtionship between x(daily) and y(sunday) is seen to be linear as seen from scatter plot
# . The correlation is strong and positive with Pearsons"s correlation coefficient of 0.958154

# #### Fit a Linear Regression Model

# In[46]:


#Build regression model

import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data = data1).fit()


# In[48]:


model1.summary()


# In[ ]:





# In[ ]:





# In[ ]:




