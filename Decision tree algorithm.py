#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install chefboost


# In[26]:


from chefboost import Chefboost as chef
import pandas as pd


# In[42]:


df = pd.read_csv("Mental Health Dataset.csv")


# In[43]:


df.head()


# In[44]:


df.drop(["Timestamp","Country","Occupation","self_employed","family_history","Days_Indoors"],axis=1, inplace=True)
df.drop(df.index[900:], axis=0, inplace=True)
df


# In[45]:


df = df.replace({"Not sure":"No"})
df = df.replace({"Maybe":"Yes"})
df


# In[46]:


config = {"algorithm":"ID3"}


# In[47]:


mod= chef.fit(df, config = config, target_label = 'care_options')


# In[41]:


pred = chef.predict(mod,param=['male','Yes','No','Yes','No','Low','Yes','No','Yes','Yes'])


# In[48]:


pred


# In[49]:


# algorithm- C4.5


# In[50]:


config = {"algorithm":"C4.5"}


# In[51]:


mod= chef.fit(df, config = config, target_label = 'care_options')


# In[54]:


pred = chef.predict(mod,param=['male','Yes','No','Yes','No','Low','Yes','No','Yes','Yes'])
pred


# In[52]:


# algorithm-CART


# In[55]:


config = {"algorithm":"CART"}


# In[56]:


mod= chef.fit(df, config = config, target_label = 'care_options')


# In[57]:


pred = chef.predict(mod,param=['male','Yes','No','Yes','No','Low','Yes','No','Yes','Yes'])
pred


# In[58]:


pred = chef.predict(mod,param=['female','Yes','No','Yes','No','Low','Yes','No','Yes','Yes'])
pred


# In[ ]:




