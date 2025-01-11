#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import pandas 
import pandas as pd 


# In[4]:


#create pandas series using list
data = [10, 20, 30, 40]
series = pd.Series(data)
print(series)


# In[8]:


#create series using a custom index 
data = [1, 2, 3, 4]
i = ['A', 'B', 'C', 'D']
series = pd.Series(data, index=i)
print(series)


# In[12]:


data = {'a': 10, 'b': 20, 'c':30}
series = pd.Series(data)
print(series)


# In[14]:


import numpy as np
data = np.array([100, 200, 300])
series = pd.Series(data, index=['a', 'b' ,'c'])
print(series)


# In[28]:


import numpy as np
data = {'Name': ['Alice', 'Bob','Mary'], 'Age': [25,30,34],'Country': ["USA","UK","AUS"]}
df = pd.DataFrame(data)
print(df)


# In[30]:


import numpy as np 
array = np.array([[1,2,3], [4,5,6], [7,8,9]])
print(array)
df = pd.DataFrame(array, columns=['A', 'B', 'C'])
print(df)


# In[40]:


iris_df = pd.read_csv("iris.csv")
print(iris_df)


# In[52]:


iris_df = pd.read_excel("iris.xlsx")
print(iris_df)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




