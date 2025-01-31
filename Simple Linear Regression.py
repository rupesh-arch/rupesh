#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
import statsmodels.formula.api as smf 


# In[6]:


data1 = pd.read_csv("NewspaperData.csv")
data1


# In[10]:


data1.info()


# In[14]:


print(type(data1))
print(data1.shape)
print(data1.size)


# In[16]:


data1.describe()


# In[18]:


data1[data1.duplicated(keep = False)]


# In[20]:


data1.drop_duplicates(keep='first', inplace = True)
data1


# In[22]:


#checking for duplicated rows in the table 
data1[data1.duplicated()]


# In[24]:


data1.isnull().sum()


# In[30]:


fig, axes = plt.subplots(2, 1, figsize=(8,6), gridspec_kw={'height_ratios': [1,3]})

#plot the boxplot in the first (top subplot)
sns.boxplot(data=data1["Newspaper"], ax=axes[0], color='skyblue', width=0.5, orient ='h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("NewspaperLevels")

#plot the histogram with KDE curve int he secondn (bottom) subplot
sns.histplot(data1["Newspaper"], kde=True, ax=axes[1], color='purple', bins=30)
axes[0].set_title("Histogram with KDE")
axes[0].set_xlabel("NewspaperLevels")
axes[0].set_ylabel("Frequency")

#Adjust Layout for better spacing
plt.tight_layout()

#show the plot 
plt.show()


# In[34]:


plt.figure(figsize=(6,2))
plt.boxplot(data1["sunday"], vert=False)


# In[36]:


plt.figure(figsize=(6,2))
plt.boxplot(data1["sunday"], vert=False)


# In[38]:


plt.scatter(data1["daily"], data1["sunday"])


# In[42]:


data1["daily"].corr(data1["sunday"])


# In[44]:


sns.swarmplot(data=data1, x = "Newspaper", y = "sunday",color="orange",palette="Set2", size=6)


# In[46]:


data1.corr(numeric_only=True)


# In[48]:


plt.scatter(data1["daily"], data1["sunday"])


# In[50]:


plt.figure(figsize=(6,3))
plt.boxplot(data1["daily"], vert = False)


# In[ ]:




