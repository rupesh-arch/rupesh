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


# In[19]:


print (data1["Weather"].value_counts())
mode_weather = data1["Weather"].mode()[0]
print(mode_weather)


# In[20]:


data1["Weather"] = data1["Weather"].fillna(mode_weather)
data1.isnull().sum


# In[21]:


print (data1["Month"].value_counts())
mode_weather = data1["Month"].mode()[0]
print(mode_weather)


# In[22]:


data1["Month"] = data1["Month"].fillna(mode_weather)
data1.isnull().sum


# DETECTION OF OUTLAYERS IN THE COLUMNS

# METHOD1 : USING HISTOGRAMS AND BOX PLOTS

# In[25]:


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


# In[26]:


Observations
. The ozone column has extreme values beyond 81 as seen from box plot
. The same is confirmed from the below right-skewed histogram 


# In[31]:


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


# In[33]:


plt.figure(figsize=(6,2))
plt.boxplot(data1["Ozone"], vert=False)


# In[35]:


#extract outlayers from boxplot for ozone column
plt.figure(figsize=(6,2))
boxplot_data = plt.boxplot(data1["Ozone"], vert=False)
[item.get_xdata() for item in boxplot_data['fliers']]


# ### METHOD 2
# . USING MU+/- 3* SIGMA LIMITS (STAN6DARD DEVIATION METHOD)

# In[37]:


data1["Ozone"].describe()


# In[39]:


mu = data1["Ozone"].describe()[1]
sigma = data1["Ozone"].describe()[2]

for x in data1["Ozone"]:
    if ((x < (mu - 3*sigma)) or (x > (mu + 3*sigma))):
        print(x)
        


# #### observations 
# . It is observed that only two outliers are idenfied using std method
# . in box plot method more no of outliers are identified 
# . This is because the assumption of normality is not satisfied in this column

# Quantile Quantile plot for detection of outliers

# In[41]:


import scipy.stats as stats

#create Q-Q plot
plt.figure(figsize=(8,6))
stats.probplot(data1["Ozone"], dist="norm", plot=plt)
plt.title("Q-Q Plot for Outlier Detection", fontsize=14)
plt.xlabel("Theoretical Quantiles", fontsize=12)


# #### Observatrion from Q-Q plot
# . The data does not follow normal distribution as the data points are f=deviating significantly away from the red line 
# . The data shows a right-skewwd distribution and possible outliers 

# In[43]:


sns.violinplot(data=data1["Ozone"], color='lightgreen')
plt.title("Vilolin Plot")


# In[45]:


plt.figure(figsize=(8,6))
stats.probplot(data1["Solar"], dist="norm", plot=plt)
plt.title("Q-Q Plot for Outlier Detection", fontsize=14)
plt.xlabel("Theoretical Quantiles", fontsize=12)


# #### Other visualisations that could help understand the data 

# In[47]:


sns.swarmplot(data=data1, x = "Weather", y = "Ozone",color="orange",palette="Set2", size=6)


# In[49]:


sns.stripplot(data=data1, x ="Weather", y = "Ozone",color="orange",palette="Set1",jitter = True)


# In[51]:


sns.kdeplot(data=data1["Ozone"], fill=True, color="red")
sns.rugplot(data=data1["Ozone"],color="black")


# In[55]:


sns.boxplot(data = data1, x = "Weather", y="Ozone")


# In[57]:


plt.scatter(data1["Wind"], data1["Temp"])


# In[59]:


#compute pearson correlation coefficient
# between wind speed and temperature
data1["Wind"].corr(data1["Temp"])


# #### observations 

# In[61]:


data1_numeric = data1.iloc[:,[0,1,2,6]]
data1_numeric


# In[63]:


data1_numeric.corr()


# ###  observations
# . the highest correalation is 1.00
# . the second highest correaltion is 0.59
# . the third highest correlation is 0.26
# . reasoning high correlation are -0.5,-0.44,-0.52

# In[66]:


sns.pairplot(data1_numeric)


# In[71]:


data2=pd.get_dummies(data1,columns=['Weather'])
data2


# In[73]:


data2=pd.get_dummies(data1,columns=['Month'])
data2


# In[75]:


data1_numeric.values


# In[83]:


#normalization of the data 
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler
array = data1_numeric.values
scaler = MinMaxScaler(feature_range=(0,1))
rescaledX = scaler.fit_transform(array)
#transformed data
set_printoptions(precision=2)
print(rescaledX[0:10,:])

