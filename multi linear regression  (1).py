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

# In[6]:


cars.info()


# In[7]:


#check for missing values 
cars.isna().sum()


#  ##### Observations about info(),missing values 
#  . There are no missing values 
#  . There are 81 observations (81 different cars data)
#  . The data types of the columns are also revelant and valid

# In[9]:


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


# In[10]:


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


# In[11]:


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


# In[12]:


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


# In[13]:


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

# In[16]:


cars[cars.duplicated()]


# #### Pair plots and correlation coefficients 

# In[18]:


#pair plot 
sns.set_style(style='darkgrid')
sns.pairplot(cars)


# #### Observations from correlation plots and coefficients
# . Between x and y, all the x variables are showing moderate to high correlation strenghts,highest being between HP and MPG
# . Therefore this dataset qualifies for building a mulitiple linear regression model to predict MPG 
# . Among x columns (x1,x2,x3and x4),some very high correlation strenghts are observed between SP vs HP and VOL vs WT
# . The high correlation among x columns is not desireable as it might lead to multi collinearity problem 

# Preparing a preliminary model considering all X columns 

# In[21]:


#Build model
import statsmodels.formula.api as smf 
model1 = smf.ols('MPG~WT+VOL+SP+HP' ,data=cars).fit()


# In[22]:


model1.summary()


# #### Obsevations from model summary
# . The R-squared and adjusted R-squared values are good and about 75% of variability in Y is explained by Xcolumns 
# . The probability value with respect to F-statistic is close to zero,indicating that all or some of Xcomments are sigificant
# . The p-values for VOL and WT are higher than 5% indicating issue among themselves, which need to be further explored 

# Performance metrics for model1

# In[25]:


#find the performance metrics 
#create a dataframe with actual y and predicted y columns 
df1 = pd.DataFrame()
df1["actual_y1"] = cars["MPG"]
df1.head()


# In[26]:


pred_y1 = model1.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# In[27]:


cars = pd.DataFrame(cars, columns=["HP", "VOL", "SP", "WT", "MPG"])
cars.head()


# In[28]:


from sklearn.metrics import mean_squqared_error
mse = mean_squared_error(df1["actual_y1"], df1["pred_y1"])
print("MSE :", mse )
print("RMSE :",np.sqrt(mse))


# In[44]:


# Compute VIF values
rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp = 1/(1-rsq_hp)

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared  
vif_wt = 1/(1-rsq_wt) 

rsq_vol = smf.ols('VOL~WT+SP+HP',data=cars).fit().rsquared  
vif_vol = 1/(1-rsq_vol) 

rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars).fit().rsquared  
vif_sp = 1/(1-rsq_sp) 

# Storing vif values in a data frame
d1 = {'Variables':['Hp','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# In[46]:


cars1 = cars.drop("WT", axis=1)
cars1.head()


# In[31]:


cars.shape


# #### Leverage (Hat Values):

# In[33]:


#define variables and assign values 
k = 3 
n = 81 
leverage_cutoff = 3*((k + 1)/n)
leverage_cutoff


# In[37]:


from statsmodels.graphics.regressionplots import influence_plot

influence_plot(model1,alpha=.05)

y=[i for i in range(-2,8)]
x=[leverage_cutoff for i in range(10)]
plt.plot(x,y,'r+')
plt.show()


# #### Observations 
# . From the above plot, it is evident that data points 65,70,76,78,79,80 are the influencers 
# . as their H Leverage values are higher and size is higher 

# In[48]:


cars[cars1.index.isin([65,70,76,78,79,80])]


# In[52]:


cars2=cars1.drop(cars1.index[[65,70,76,78,79,80]],axis=0).reset_index(drop=True)
cars2


# #### Build Model3 on cars2 dataset

# In[55]:


#Rebuild the model model
model3= smf.ols('MPG~VOL+SP+HP',data = cars2).fit()


# In[57]:


model3.summary()


# #### Performance Metrics for model3

# In[60]:


df3= pd.DataFrame()
df3["actual_y3"] =cars2["MPG"]
df3.head()


# In[62]:


# Predict on all x data columns 
pred_y3 = model3.predict(cars2.iloc[:,0:3])
df3["pred_y3"] = pred_y3
df3.head()


# In[64]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df3["actual_y3"], df3["pred_y3"])
print("MSE :", mse)
print("RSME :",np.sqrt(mse))


# #### Comparison of models
#                      
# 
# | Metric         | Model 1 | Model 2 | Model 3 |
# |----------------|---------|---------|---------|
# | R-squared      | 0.771   | 0.770   | 0.885   |
# | Adj. R-squared | 0.758   | 0.761   | 0.880   |
# | MSE            | 18.89   | 18.91   | 8.68    |
# | RMSE           | 4.34    | 4.34    | 2.94    |
# 
# 
# - **From the above comparison table it is observed that model3 is the best among all with superior performance metrics**

# In[ ]:





# In[ ]:





# In[ ]:




