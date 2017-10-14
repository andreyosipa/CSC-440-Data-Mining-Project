
# coding: utf-8

# In[45]:

import pandas as pd
import numpy as np


# In[46]:

import matplotlib.pyplot as plt


# In[47]:

from sklearn import preprocessing


# In[48]:

import sklearn


# In[49]:

get_ipython().magic(u'matplotlib inline')


# In[50]:

data = pd.read_csv("train.csv")


# In[51]:

data.columns


# In[52]:

np.sort(data['SalePrice'])


# In[53]:

plt.figure(figsize=(25,10))
plt.plot(np.sort(data['SalePrice']), 'bo-')


# In[54]:

plt.figure(figsize=(25,7))
plt.scatter(data['SalePrice'], data['GarageCars'])


# In[55]:

plt.figure(figsize=(25,7))
plt.scatter(data['GarageCars'], data['GarageArea'])


# In[56]:

plt.figure(figsize=(25,7))
plt.scatter(data['PoolArea'], data['SalePrice'])


# Pool size may be categorial variable. Do not affect on price.

# In[13]:

plt.figure(figsize=(25,10))
plt.plot(np.sort(data['SalePrice'][np.random.choice(1460, 1198)]), label='all')
plt.plot(np.sort(data[data['SaleCondition']=='Normal']['SalePrice']), label='normal')
plt.legend()


# Sale price is higher for expensive houses and lower for cheap if it sold with special sale conditions.

# In[14]:

len(data)


# In[30]:

979.0/1460


# In[15]:

len(data[data['SaleCondition']=='Normal'])


# In[38]:

import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams.update({'font.size': 26})


# In[16]:

plt.figure(figsize=(25,25))
plt.scatter(data['SalePrice'], data['MSSubClass'])


# In[17]:

mssubclasses = [20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 160, 180, 190]


# In[18]:

data['MSSubClass'].value_counts()


# In[39]:

plt.figure(figsize=(25,15))
for msclass in mssubclasses:
    plt.plot(np.sort(data[data['MSSubClass'] == msclass]['SalePrice']), label=str(msclass))
plt.legend()


# In[21]:

for msclass in mssubclasses:
    print msclass, np.median(data[data['MSSubClass'] == msclass]['SalePrice'])


# In[22]:

from sklearn.ensemble import RandomForestClassifier


# In[23]:

data.columns.values[:-1]


# In[43]:

matplotlib.rcParams.update({'font.size': 15})


# In[44]:

for column_idx in range(len(data.columns.values[:-1])):
    Xuniques, X = np.unique(data[data.columns.values[column_idx]], return_inverse=True) 
    plt.figure(column_idx)
    plt.figure(figsize=(10,10))
    if len(Xuniques)<100:
        plt.scatter(data['SalePrice'], X, label=str(data.columns.values[column_idx]))
        plt.yticks(range(len(Xuniques)), Xuniques)
    else:
        plt.scatter(data['SalePrice'], data[data.columns.values[column_idx]], label=str(data.columns.values[column_idx]))
    plt.legend()


# In[25]:

np.array(data)[:,1:-1]

