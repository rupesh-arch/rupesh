#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


# In[2]:


df = pd.read_csv('diabetes.csv')
df


# In[3]:


x = df.drop('class', axis=1)

y = df['class']


# In[4]:


scaler = StandardScaler()

x_scaled = scaler.fit_transform(x)

x_scaled


# In[5]:


x_train, x_test, y_train, y_test = train_test_split(x_scaled,y, test_size = 0.8, random_state = 42)


# In[6]:


gbc = GradientBoostingClassifier(random_state=42)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {
    'n_estimators': [50,100,150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(estimator=gbc,
                           param_grid=param_grid,
                           cv=kfold,
                           scoring='recall',
                           n_jobs=-1,
                           verbose=1)


# In[11]:


grid_search.fit(x_train, y_train)
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-validated Recall:", grid_search.best_score_)


# In[19]:


best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_test)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[ ]:




