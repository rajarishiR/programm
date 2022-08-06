#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('Fraud.csv')
df


# In[3]:


df.columns


# step - maps a unit of time in the real world. In this case 1 step is 1 hour of time. Total steps 744 (30 days simulation).
# 
# type - CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER.
# 
# amount - amount of the transaction in local currency.
# 
# nameOrig - customer who started the transaction
# 
# oldbalanceOrg - initial balance before the transaction
# 
# newbalanceOrig - new balance after the transaction
# 
# nameDest - customer who is the recipient of the transaction
# 
# oldbalanceDest - initial balance recipient before the transaction. Note that there is not information for customers that start with M (Merchants).
# 
# newbalanceDest - new balance recipient after the transaction. Note that there is not information for customers that start with M (Merchants).
# 
# isFraud - This is the transactions made by the fraudulent agents inside the simulation. In this specific dataset the fraudulent behavior of the agents aims to profit by taking control or customers accounts and try to empty the funds by transferring to another account and then cashing out of the system.
# 
# isFlaggedFraud - The business model aims to control massive transfers from one account to another and flags illegal attempts. An illegal attempt in this dataset is an attempt to transfer more than 200.000 in a single transaction.

# # Fraud detection model.
#      This case requires trainees to develop a model for predicting fraudulent transactions for a 
#      financial company and use insights from the model to develop an actionable plan.

# # Data preprocessing.

# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df.describe()


# In[7]:


df.type.value_counts()


# In[8]:


sns.countplot(df.type)


# There are more outliers in the type column but it can be taken into consideration cause in a financial sector these transcations can happen.

# In[9]:


df.isFraud.value_counts()


# 0 - Not a fraud
# 1 - Its a fraud
# In the isFraud column shows that there are 8213 fraud transcations happend in 30 days.

# In[10]:


df.isFlaggedFraud.value_counts()


# The isFlaggedFraud column is a model which prevents the fraud transcation by keeping a threshold 200.000 in a single transcations so we cannot take this columns frauds into consideration.

# # selecting columns.
# 
#   1. The fraud transcations can happen only if the type of transcations are transfer and cash out in the isFraud columns.

# In[11]:


print('Type of transaction which are fraud:{}'.format(list(df.loc[df.isFraud == 1].type.drop_duplicates().values)))
fraud_transfer = df.loc[(df.isFraud == 1) & (df.type == 'TRANSFER')]
fraud_cashout = df.loc[(df.isFraud == 1) & (df.type == 'CASH_OUT')]
print('number of transfer are fraud :{}'.format(len(fraud_transfer)))
print('number of cashout are fraud :{}'.format(len(fraud_cashout)))


# This clearly shows that the total no of frauds happend in the IsFraud column comes only from Transfer and Cashout cause total fraud happend is 8213 and the total of these two sums up to 8213.

# # Finding the correlation.

# In[12]:


c = df.corr()
fig = plt.figure(figsize=(12,9))

sns.heatmap(c, vmax = 0.8, square=True)
plt.show()


#      selecting variables to be included in the model
#         1. Since transfer and cashout are the types there has been fraud we will be using that as a feature and drop columns                like nameorig , namedest , isFlaggedFraud.
#         2 . we will be encoding the transfer and cashout.

#  # Feature selection and encoding.

# In[13]:


x = df.loc[(df.type == 'TRANSFER') | (df.type == 'CASH_OUT')]
y = x.isFraud
x = x.drop(['isFraud','isFlaggedFraud','nameOrig','nameDest'],axis = 1)
x['type']= x['type'].map({'TRANSFER':1,'CASH_OUT':2})


# # Train test split

# In[14]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=27)


# # ML model.
#     Since it is a classification model.
#     
#     In this case i will be using decision tree model.
#     
#     1. The decision tree model will have n number of nodes which consist of questions based on the data and finally finds it as        fraud or not fraud.
#     2. the model can give the feature importance.

# # Decision tree model.

# In[15]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
q = dt.predict(x_test)


# # Evaluation metrices.

# In[16]:


print("Training data score for DT: {:.2f}".format(dt.score(x_train, y_train)))
print("Test data score for DT: {:.2f}".format(dt.score(x_test, y_test)))


# # Feature importance.

# In[18]:


impo = dt.feature_importances_
list(zip(impo,x.columns))


# # graph for actual vs predicted fraud.

# In[19]:


k = pd.DataFrame({'actual_value':y_test,'predicted_value':q})
plt.figure(figsize=(15,4))
sns.kdeplot(data=k, x='actual_value', label='actual', color = 'red',shade=True)
sns.kdeplot(data=k, x='predicted_value', label='predicted', color='blue', shade=True)
plt.title("Actual fraud Vs Predicted fraud")
plt.legend()
plt.show()


# SO the accuracy score is 1.00 means our model performing well on positive (fraudlant transaction) class.
# CONCLUSION -
# 
# 1 . Company can prevent fraudalnt transaction by focusing more on payment method type - 'Transfer' & 'Cash_out'
# Look Out for Patterns in Fraud and Theft.
# 
# 2 . we can retrain & maintain model after certain intervals so our model perform best under various fraudlant transaction.

# In[ ]:




