#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import statsmodels.formula.api as smf 
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np 


# In[2]:


cars = pd.read_csv("Cars.csv")
cars.head()


# In[3]:


cars = pd.DataFrame(cars, columns=["HP", "VOL", "SP", "WT", "MPG"])
cars.head()


# #### Assumptions in multilinear regression 
# 1. Linearity: The relationship between the predictors(X) and the response (Y) is linear
# 2. Independence: Observations are independent of each other
# 3. Homoscedasticity: The residuals (Y-Y_hat) exhibit constant variance at all levels of the predictor
# 4. Normal Distribution of Errors: The residuals of the model are normally distributed
# 5. No multicollinearity: The independent variables should not be too highly correlated with each other, violations of these assumptions may lead to inefficiency in the regression parameters and unreliable predictions 

# #### EDA

# In[13]:


cars.info()


# In[18]:


#check for missing values 
cars.isna().sum()


#  ##### Observations about info(),missing values 
#  . There are no missing values 
#  . There are 81 observations (81 different cars data)
#  . The data types of the columns are also revelant and valid

# In[25]:


#create a figure with two subplots (one above the other)
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

#creating a boxplot 
sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='')

#craeting a histogram in the same x-axis 
sns.histplot(data=cars,x='HP',ax=ax_hist, bins=30, kde=True, stat="density")

#Adjust layout 
plt.tight_layout()
plt.show()


# In[27]:


#create a figure with two subplots (one above the other)
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

#creating a boxplot 
sns.boxplot(data=cars, x='SP', ax=ax_box, orient='h')
ax_box.set(xlabel='')

#craeting a histogram in the same x-axis 
sns.histplot(data=cars,x='SP',ax=ax_hist, bins=30, kde=True, stat="density")

#Adjust layout 
plt.tight_layout()
plt.show()


# In[29]:


#create a figure with two subplots (one above the other)
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

#creating a boxplot 
sns.boxplot(data=cars, x='WT', ax=ax_box, orient='h')
ax_box.set(xlabel='')

#craeting a histogram in the same x-axis 
sns.histplot(data=cars,x='WT',ax=ax_hist, bins=30, kde=True, stat="density")

#Adjust layout 
plt.tight_layout()
plt.show()


# In[31]:


#create a figure with two subplots (one above the other)
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

#creating a boxplot 
sns.boxplot(data=cars, x='MPG', ax=ax_box, orient='h')
ax_box.set(xlabel='')

#craeting a histogram in the same x-axis 
sns.histplot(data=cars,x='MPG',ax=ax_hist, bins=30, kde=True, stat="density")

#Adjust layout 
plt.tight_layout()
plt.show()


# In[33]:


#create a figure with two subplots (one above the other)
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

#creating a boxplot 
sns.boxplot(data=cars, x='VOL', ax=ax_box, orient='h')
ax_box.set(xlabel='')

#craeting a histogram in the same x-axis 
sns.histplot(data=cars,x='VOL',ax=ax_hist, bins=30, kde=True, stat="density")

#Adjust layout 
plt.tight_layout()
plt.show()


# #### Observations from boxplot and histograms 
# . There are some extreme values(outliers) observed in towards the right tail of SP and HP distributions 
# . In VOL and Wt columns, a few outliers are observed in the both tails of their ditributions 
# . The extreme values of cars data may have come from the specially designed nature of cars 
# . As this is muliti-dimensional data, the outliers with respect to spatial dimensions may have to be considered while buildng the regression model

# #### Checking for duplicated rows 

# In[38]:


cars[cars.duplicated()]


# #### Pair plots and correlation coefficients 

# In[41]:


#pair plot 
sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[ ]:





# In[ ]:




