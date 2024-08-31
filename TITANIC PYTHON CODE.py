#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[7]:


data = pd.read_csv("Titanic-Dataset.csv")


# In[8]:


data


# In[9]:


print({col: data[col].isna().sum() for col in data.columns})


# In[10]:


data=data.drop(columns='Cabin',axis=1)


# In[11]:


data


# In[12]:


data['Age'] = np.where(data['Age'].isna(), data['Age'].mean(), data['Age'])


# In[13]:


data['Embarked'].fillna(data['Embarked'].mode()[0],inplace=True)


# In[14]:


data.isnull().sum()


# In[15]:


ax = data['Sex'].value_counts().plot(kind='bar', figsize=(6, 5), color=['blue', 'orange'])
ax.set_title("Ratio of Male and Female Passengers")
ax.set_ylabel("Count")
ax.set_xlabel("Sex")
plt.show()


# In[16]:


sns.set()
sns.countplot(x='Survived',data=data)


# In[17]:


sns.countplot(x='Pclass',data=data)
sns.countplot(x='Pclass',hue='Survived', data=data)


# In[18]:


data.replace({'Sex':{'male':0,'female':1},'Embarked':{'S':0,'C':1,'Q':2}},inplace=True)
data


# In[21]:


X=data.drop(columns=['PassengerId','Name','Ticket'],axis=1)
print(X)


# In[22]:


Y=data['Survived']
print(Y)


# In[23]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)

print(f"Original shape: {X.shape}")
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")


# In[24]:


model=LogisticRegression()
model.fit(X_train,Y_train)


# In[25]:


Xpred=model.predict(X_train)
print(Xpred)


# In[26]:


trainacc=accuracy_score(Y_train,Xpred)
print("training:",trainacc)


# In[27]:


Xtst=model.predict(X_test)
print(Xtst)


# In[ ]:


tacc=accurcy_score(Y_test,Xtst)
print("tested accuracy")

