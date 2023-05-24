#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_excel(r"C:\Users\HP\Documents\housing2.xlsm")
df


# In[2]:


import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns


# In[3]:


df.info()


# In[4]:


df.dropna()


# In[5]:


df.dropna(inplace=True)


# In[6]:


df.info()


# In[7]:


from sklearn.model_selection import train_test_split

x = df.drop(['median_house_value'],axis=1)
y = df['median_house_value']


# In[8]:


x


# In[9]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[10]:


train_df = x_train.join(y_train)


# In[11]:


train_df


# In[12]:


train_df.hist()


# In[13]:


train_df.hist(figsize = (15, 8))


# In[14]:


train_df.corr()


# In[15]:


plt.figure(figsize=(15,8))
sns.heatmap(train_df.corr(),annot=True)


# In[16]:


train_df['total_rooms'] = np.log(train_df['total_rooms'] + 1)
train_df['total_bedrooms'] = np.log(train_df['total_bedrooms'] + 1)
train_df['population'] = np.log(train_df['population'] + 1)
train_df['households'] = np.log(train_df['households'] + 1)


# In[17]:


train_df.hist(figsize=(15,8))


# In[18]:


train_df.ocean_proximity.value_counts()


# In[19]:


train_df = train_df.join(pd.get_dummies(train_df.ocean_proximity)).drop(['ocean_proximity'],axis=1)


# In[20]:


train_df


# In[21]:


plt.figure(figsize=(15,8))
sns.heatmap(train_df.corr(),annot=True)


# In[23]:


from sklearn.model_selection import train_test_split

x = df.drop(['ocean_proximity'],axis=1)


# In[24]:


x


# In[25]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[26]:


from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor()
forest.fit(x_train, y_train)


# In[27]:


forest.score(x_test, y_test)


# In[ ]:




