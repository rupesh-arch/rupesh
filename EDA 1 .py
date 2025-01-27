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


# In[9]:


#checking for duplicated rows in the table 
data1[data1.duplicated()]


# In[10]:


data1.rename({'Solar.R' : 'Solar'}, axis=1, inplace = True)
data1


# impute the missing values

# In[12]:


data.info()


# In[13]:


data1.isnull().sum()


# In[14]:


cols = data1.columns
colors = ['black', 'yellow' ]
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar = True)


# In[15]:


median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ", median_ozone)
print("Mean of Ozone: ", mean_ozone)


# In[16]:


data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[17]:


mean_Solar = data1["Solar"].mean()
print("Mean of Solar: ",mean_Solar)


# In[18]:


data1['Solar'] = data1['Solar'].fillna(mean_Solar)
data1.isnull().sum()


# In[37]:


print (data1["Weather"].value_counts())
mode_weather = data1["Weather"].mode()[0]
print(mode_weather)


# In[41]:


data1["Weather"] = data1["Weather"].fillna(mode_weather)
data1.isnull().sum


# In[45]:


print (data1["Month"].value_counts())
mode_weather = data1["Month"].mode()[0]
print(mode_weather)


# In[47]:


data1["Month"] = data1["Month"].fillna(mode_weather)
data1.isnull().sum


# DETECTION OF OUTLAYERS IN THE COLUMNS

# METHOD1 : USING HISTOGRAMS AND BOX PLOTS

# In[61]:


#create a figure with two subplots, stacked vertically
fig, axes = plt.subplots(2, 1, figsize=(8,6), gridspec_kw={'height_ratios': [1,3]})

#plot the boxplot in the first (top subplot)
sns.boxplot(data=data1["Ozone"], ax=axes[0], color='skyblue', width=0.5, orient ='h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Ozone Levels")

#plot the histogram with KDE curve int he secondn (bottom) subplot
sns.histplot(data1["Ozone"], kde=True, ax=axes[1], color='purple', bins=30)
axes[0].set_title("Histogram with KDE")
axes[0].set_xlabel("Ozone Levels")
axes[0].set_ylabel("Frequency")

#Adjust Layout for better spacing
plt.tight_layout()

#show the plot 
plt.show()


# In[ ]:


Observations
. The ozone column has extreme values beyond 81 as seen from box plot
. The same is confirmed from the below right-skewed histogram 


# In[63]:


#create a figure with two subplots, stacked vertically
fig, axes = plt.subplots(2, 1, figsize=(8,6), gridspec_kw={'height_ratios': [1,3]})

#plot the boxplot in the first (top subplot)
sns.boxplot(data=data1["Solar"], ax=axes[0], color='skyblue', width=0.5, orient ='h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Solar Levels")

#plot the histogram with KDE curve int he secondn (bottom) subplot
sns.histplot(data1["Solar"], kde=True, ax=axes[1], color='purple', bins=30)
axes[0].set_title("Histogram with KDE")
axes[0].set_xlabel("Solar Levels")
axes[0].set_ylabel("Frequency")

#Adjust Layout for better spacing
plt.tight_layout()

#show the plot 
plt.show()

