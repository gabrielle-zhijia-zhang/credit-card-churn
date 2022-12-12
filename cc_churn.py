#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from scipy import stats, special
from sklearn import model_selection, metrics, linear_model, datasets, feature_selection
from seaborn import pairplot, heatmap
import seaborn as sns

import matplotlib.pyplot as plt


# In[3]:


bank = pd.read_csv('BankChurners.csv')


# In[4]:


bank.head()


# In[9]:


### Select the columns I need ###
bank2 = bank.loc[:,'Attrition_Flag':'Credit_Limit']
bank2.shape


# In[10]:


bank2.info()


# In[11]:


### Convert column values to 1 or 0
bank2['Attrition_Flag'] = bank2['Attrition_Flag'].replace({'Attrited Customer': 0, 'Existing Customer': 1})
bank2['Gender'] = bank2['Gender'].replace({'M': 0, 'F': 1})


# In[12]:


bank2['Education_Level'].value_counts()   


# In[14]:


bank2['Income_Category'].value_counts()  


# In[15]:


bank2['Card_Category'].value_counts()  


# In[16]:


### Convert the ordinal variable to numeric
bank2['Card_Category'] = bank2['Card_Category'].map({'Blue': 1, 'Silver': 2, 'Gold': 3, 'Platinum': 4})


# In[17]:


sns.pairplot(data=bank2)


# In[18]:


### Create dummy variables and concat other columns
dummies = pd.get_dummies(bank2[['Education_Level', 'Marital_Status', 'Income_Category']])
dummies


# In[19]:


bank2 = pd.concat([bank2.drop(columns=['Education_Level', 'Marital_Status', 'Income_Category']), dummies],
                   axis=1)


# In[20]:


bank2


# In[ ]:





# In[77]:


### Logistic Regression ###

from sklearn.model_selection import train_test_split
y = bank2["Attrition_Flag"]
X = bank2.drop("Attrition_Flag", axis =1)


# In[78]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[59]:


from sklearn.linear_model import LogisticRegression


# In[79]:


logreg = LogisticRegression(solver="liblinear").fit(X_train,y_train)


# In[80]:


y_pred = logreg.predict(X_test)


# In[81]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[63]:


### Decision Tree ###
from sklearn.tree import DecisionTreeClassifier


# In[82]:


dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)


# In[83]:


dt.score(X_test, y_test)


# In[66]:


### Random Forest ###
from sklearn.ensemble import RandomForestClassifier


# In[84]:


rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)


# In[85]:


rf.score(X_test, y_test)


# In[69]:


### KNN ###
from sklearn.neighbors import KNeighborsClassifier


# In[86]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)


# In[87]:


knn.score(X_test, y_test)


# In[ ]:





# In[ ]:


## Conclusion: Random Forest and Logistic Regression models have the highest accuarcy.

