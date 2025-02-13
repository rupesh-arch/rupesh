#!/usr/bin/env python
# coding: utf-8

# In[25]:


get_ipython().system('pip install mlxtend')


# In[27]:


import pandas as pd
import mlxtend
from mlxtend.frequent_patterns import apriori,association_rules
import matplotlib.pyplot as plt 


# In[29]:


titanic = pd.read_csv("Titanic.csv")
titanic


# In[33]:


titanic.info()


# In[37]:


titanic.describe()


# #### observations 
# . there is no null value 
# . all colums are object and categorical in nature 
# . As the columns are categorical we can adapt one -hat-encoding

# In[43]:


counts = titanic['Class'].value_counts()
plt.bar(counts.index, counts.values)


# #### Observations
# . maximum travellers are the crew
# . next comes 3rd class travellers are highest 
# . next comes 1st class travellers 
# . the last ones are the 2nd class travellers 

# In[46]:


df=pd.get_dummies(titanic,dtype=int)
df.head()


# In[48]:


df.info()


# In[52]:


frequent_itemsets = apriori(df, min_support = 0.05,use_colnames=True,max_len=None)
frequent_itemsets


# In[56]:


frequent_itemsets.info()


# In[62]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

rules


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




