#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[2]:


data1 = pd.read_csv("NewspaperData.csv")
data1.head()


# In[3]:


data1.info()


# In[4]:


data1.isnull().sum()


# In[5]:


plt.figure(figsize=(6,3))
plt.title("Box plot for Daily sales")
plt.boxplot(data1["daily"], vert = False)
plt.show()


# In[6]:


sns.histplot(data1['daily'], kde = True,stat='density',)
plt.show()


# In[7]:


sns.histplot(data1['sunday'], kde = True,stat='density',)
plt.show()


# In[8]:


sns.histplot(data1['Newspaper'], kde = True,stat='density',)
plt.show()


# #### observations
# . The are no missing values
# . The daily column values appear to be right-skewed
# . The sunday column vqalues also appear to be right- skewed
# . There are two outliera in both daily column and also in sunday column as observed from the 

# In[10]:


x= data1["daily"]
y = data1["sunday"]
plt.scatter(data1["daily"], data1["sunday"])
plt.xlim(0, max(x) + 100)
plt.ylim(0, max(y) + 100)
plt.show()


# In[11]:


data1["daily"].corr(data1["sunday"])


# In[12]:


data1[["daily","sunday"]].corr()


# In[13]:


data1.corr(numeric_only=True)


# #### observations on correlation strength
# .The realtionship between x(daily) and y(sunday) is seen to be linear as seen from scatter plot
# . The correlation is strong and positive with Pearsons"s correlation coefficient of 0.958154

# #### Fit a Linear Regression Model

# In[16]:


#Build regression model

import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data = data1).fit()


# In[17]:


model1.summary()


# #### Interpretation:
# . R - squared = 1 -> Perfect fit (all variance explained 
# . R - squared = 0 -> Model does not explain any variance 
# . R - squared close to 1 -> Good model fit 
# . R - squared close to 0 -> Poor model fit 

# In[36]:


#Plot the scatter plot and overlay the fitted straight line using matplotlib
x = data1["daily"].values
y = data1["sunday"].values
plt.scatter(x, y, color = "m", marker = "o", s = 30)
b0 = 13.84
b1 =1.33
#plotting the regreesion vector 
y_hat = b0 +b1*x

#plotting the regression line 
plt.plot(x, y_hat, color ="g")

#putting labels 
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# #### observations from model summary 
# 
# . The probability(p-value) for intercept (beta_0) is 0.707 > 0.05
# . Therefore the intercept coefficient may not be that much significant in prediction 
# . However the p-value for "daily"(beta_1) is 0.00 < 0.05
# . Therefore the beta_1 coefficient is highly significant and is contributint to prediction

# In[41]:


#print the fitted the coefficients (Beta_0, beta_1)
model1.params


# In[57]:


#print the model statistics (t and p-values)
print(model1.tvalues)
print(model1.pvalues)


# In[49]:


#predict for 200 and 300 daily circulation
newdata=pd.Series([200,300,1500])


# In[51]:


data_pred=pd.DataFrame(newdata,columns=['daily'])
data_pred


# In[53]:


model1.predict(data_pred)


# In[63]:


#predict on all given training data 
pred = model1.predict(data1["daily"])
pred


# In[65]:


data1["Y_hat"] = pred
data1


# In[75]:


data1["residuals"] = data1["sunday"]-data1["Y_hat"]
data1


# In[ ]:





# In[ ]:





# In[ ]:




