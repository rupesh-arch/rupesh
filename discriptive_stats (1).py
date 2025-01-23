#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np


# In[2]:


df = pd.read_csv("Universities.csv")
df


# 

# In[4]:


np.mean(df["SAT"])


# In[5]:


np.median(df["SAT"])


# In[6]:


np.std(df["GradRate"])


# In[7]:


np.var(df["GradRate"])


# In[8]:


df.describe()


# In[9]:


#### visualization
#visualize the GradRate using histogram
import matplotlib.pyplot as plt
import seaborn as sns 


# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[11]:


plt.hist(df["GradRate"])


# In[12]:


plt.figure(figsize=(50,90))
plt.title("Graduation Rate")
plt.hist(df["GradRate"])


# In[33]:


#visualiztion using boxplot 
s = [20,15,10,25,30,35,28,35,40,60]
scores = pd.Series(s)
scores


# In[35]:


plt.boxplot(scores, vert=False)


# In[39]:


s = [20,15,10,25,30,35,28,35,40,60,500]
scores = pd.Series(s)
scores


# In[41]:


plt.boxplot(scores, vert=False)


# In[43]:


df = pd.read_csv("universities.csv")
df


# In[55]:


plt.figure(figsize=(6,2))
plt.boxplot(df["SAT"], vert =False)


# In[ ]:




