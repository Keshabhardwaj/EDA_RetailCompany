#!/usr/bin/env python
# coding: utf-8

# In[98]:


## dataset link: https://www.kaggle.com/sdolezel/black-friday?select=train.csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# #Problem Statement
# A retail company “ABC Private Limited” wants to understand the customer purchase behaviour (specifically, purchase amount) against various products of different categories. They have shared purchase summary of various customers for selected high volume products from last month. The data set also contains customer demographics (age, gender, marital status, city_type, stay_in_current_city), product details (product_id and product category) and Total purchase_amount from last month.
# 
# Now, they want to build a model to predict the purchase amount of customer against various products which will help them to create personalized offer for customers against different products.

# In[ ]:





# In[99]:


#importing the dataset
df_train=pd.read_csv('train.csv')
df_train.head()
df_train.shape


# In[100]:


df_train.head(2)


# In[101]:


##  import the test data
df_test=pd.read_csv('test.csv')
df_test.head(2)


# In[102]:


df_test.shape


# In[104]:


# Merge both train and test data 

df=df_train.merge(df_test)


# In[77]:


#Basic
df.info()


# In[78]:


df.describe()


# In[79]:


df.drop(['User_ID'],axis=1,inplace=True)


# In[80]:


df.head()


# In[97]:


# df['Gender']=pd.get_dummies(df['Gender'],drop_first=1)


# In[82]:


##HAndling categorical feature Gender
df['Gender']=df['Gender'].map({'F':0,'M':1})
df.head(10)


# In[83]:


## Handle categorical feature Age
df['Age'].unique()


# In[84]:


#pd.get_dummies(df['Age'],drop_first=True)
df['Age']=df['Age'].map({'0-17':1,'18-25':2,'26-35':3,'36-45':4,'46-50':5,'51-55':6,'55+':7})


# In[85]:


df.head()


# In[86]:


##fixing categorical City_categort
df_city=pd.get_dummies(df['City_Category'],drop_first=True)


# In[87]:


df_city.head()


# In[88]:


df=pd.concat([df,df_city],axis=1)
df.head()


# In[89]:


##drop City Category Feature
df.drop('City_Category',axis=1,inplace=True)


# In[90]:


df.info()


# In[91]:


df['Stay_In_Current_City_Years'].unique()


# In[92]:


df.head()


# In[93]:


## Missing Values
df.isnull().sum()


# In[94]:


## Focus on replacing missing values
df['Product_Category_2'].unique()


# In[95]:


df['Product_Category_2'].value_counts()


# In[96]:


df['Product_Category_2'].mode()[0]


# In[ ]:


## Replace the missing values with mode
df['Product_Category_2']=df['Product_Category_2'].fillna(df['Product_Category_2'].mode()[0])


# In[ ]:


df['Product_Category_2'].isnull().sum()


# In[ ]:


## Product_category 3 replace missing values
df['Product_Category_3'].unique()


# In[ ]:


df['Product_Category_3'].value_counts()


# In[ ]:


## Replace the missing values with mode
df['Product_Category_3']=df['Product_Category_3'].fillna(df['Product_Category_3'].mode()[0])


# In[ ]:


df.shape


# In[62]:


df['Stay_In_Current_City_Years'].unique()


# In[63]:


df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].str.replace('+','')


# In[64]:


df.head()


# In[65]:


df.info()


# In[66]:


##convert object into integers
df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].astype(int)
df.info()


# In[67]:


df['B']=df['B'].astype(int)
df['C']=df['C'].astype(int)


# In[ ]:


df.info()


# In[ ]:


df.drop(['Product_ID'],axis=1,inplace=True)


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


##Visualisation Age vs Purchased
sns.barplot(x='Age',y='Purchase',hue='Gender',data=df)


# In[ ]:


## Visualization of Purchase with occupation
sns.barplot(x='Occupation',y='Purchase',hue='Gender',data=df)


# In[ ]:




