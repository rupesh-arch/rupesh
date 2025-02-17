#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import operator as op
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import plotly.graph_objects as go
import os
import plotly.express as px


# In[15]:


data = pd.read_csv("groceries_dataset.csv")
print(data)


# In[17]:


data.info()


# In[19]:


data.head()


# In[21]:


data.tail()


# In[23]:


data.describe()


# In[27]:


nan_values = data.isna().sum()
nan_values


# In[29]:


data.Date = pd.to_datetime(data.Date)
data.memberID = data['memberID'].astype('str')
data.info()


# In[31]:


Sales_weekly = data.resample('w', on='Date').size()
fig = px.line(data, x=Sales_weekly.index, y=Sales_weekly,
              labels={'y': 'Number of Sales',
                     'x': 'Date'})
fig.update_layout(title_text='Number of Sales Weekly',
                  title_x=0.5, title_font=dict(size=18))
fig.show()


# In[ ]:





# In[ ]:





# In[ ]:




