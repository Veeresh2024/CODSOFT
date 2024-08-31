#!/usr/bin/env python
# coding: utf-8

# In[97]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# In[98]:


cc=pd.read_csv("creditcard.csv")


# In[99]:


cc


# In[100]:


cc.describe()


# In[101]:


cc.isnull().sum()


# In[102]:


x=cc.loc[:, cc.columns !='Class'].copy()
y=cc.loc[:,'Class']


# In[103]:


sns.countplot(x=y)
plt.title('Count of Classes')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()


# In[104]:


Scalar = StandardScaler()
x.loc[:, 'Amount'] = Scalar.fit_transform(x[['Amount']])


# In[105]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)


# In[ ]:


from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
x_re, y_re= rus.fit_resample(x_train, y_train)
sns.histplot(y_re, discrete=True)
plt.title('Count of Resampled Classes')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()


# In[ ]:


model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
model.fit(x_re, y_re)


# In[ ]:


y_pred = model.predict(x_test)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[114]:


conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[47]:


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")


# In[111]:


pip install --upgrade scikit-learn imbalanced-learn


# In[112]:


pip install scikit-learn==1.2.0 imbalanced-learn==0.10.0


# In[ ]:


pip show scikit-learn imbalanced-learn


# In[113]:


python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
pip install scikit-learn imbalanced-learn seaborn matplotlib


# In[109]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming y_test and y_pred are already defined
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[ ]:




