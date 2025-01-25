#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 


# In[2]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[3]:


data.info()


# In[4]:


print(type(data))
print(data.shape)
print(data.size)
           


# In[5]:


data1 = data.drop(['Unnamed: 0',"Temp C"], axis =1)
data1


# In[6]:


data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data.info()


# In[7]:


data1[data1.duplicated(keep = False)]


# In[8]:


data1.drop_duplicates(keep='first', inplace = True)
data1


# In[25]:


#checking for duplicated rows in the table 
data1[data1.duplicated()]


# In[37]:


data1.rename({'Solar.R' : 'Solar'}, axis=1, inplace = True)
data1


# impute the missing values

# In[41]:


data.info()


# In[43]:


data1.isnull().sum()


# In[53]:


cols = data1.columns
colors = ['black', 'yellow' ]
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar = True)


# In[55]:


median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ", median_ozone)
print("Mean of Ozone: ", mean_ozone)


# In[59]:


data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[63]:


mean_Solar = data1["Solar"].mean()
print("Mean of Solar: ",mean_Solar)


# In[65]:


data1['Solar'] = data1['Solar'].fillna(mean_Solar)
data1.isnull().sum()


# In[ ]:




