#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[4]:


ir=pd.read_csv(r"IRIS.csv")
ir


# In[5]:


ir.info()


# In[6]:


ir.describe()


# In[7]:


ir.isnull().sum()


# In[10]:


cols = ['sepal_length']
for column in cols:
    plt.figure()
    ir[column].hist()
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()


# In[12]:


cols = ['sepal_width']
for column in cols:
    plt.figure()
    ir[column].hist()
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()


# In[13]:


cols = ['petal_length']
for column in cols:
    plt.figure()
    ir[column].hist()
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()


# In[14]:


cols = ['petal_width']
for column in cols:
    plt.figure()
    ir[column].hist()
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()


# In[15]:


feature_combinations = [
    ('sepal_length','sepal_width'),
    ('petal_length','petal_width'),
    ('sepal_length','petal_length'),
    ('sepal_width','petal_width'),  
]


# In[16]:


color_palette = {'Iris-setosa': 'red', 'Iris-versicolor': 'black', 'Iris-virginica': 'teal'}

# Scatter plot for 'sepal_length' vs 'sepal_width'
plt.figure(figsize=(10, 6))
for species in ir['species'].unique():
    subset = ir[ir['species'] == species]
    plt.scatter(subset['sepal_length'], subset['sepal_width'], color=color_palette[species], label=species)
plt.xlabel('sepal_length')
plt.ylabel('sepal_width')
plt.title('Scatter Plot of sepal_length vs sepal_width')
plt.legend()
plt.show()


# In[17]:


plt.figure(figsize=(10, 6))
for species in ir['species'].unique():
    subset = ir[ir['species'] == species]
    plt.scatter(subset['petal_length'], subset['petal_width'], color=color_palette[species], label=species)
plt.xlabel('petal_length')
plt.ylabel('petal_width')
plt.title('Scatter Plot of petal_length vs petal_width')
plt.legend()
plt.show()


# In[18]:


plt.figure(figsize=(10, 6))
for species in ir['species'].unique():
    subset = ir[ir['species'] == species]
    plt.scatter(subset['sepal_length'], subset['petal_length'], color=color_palette[species], label=species)
plt.xlabel('sepal_length')
plt.ylabel('petal_length')
plt.title('Scatter Plot of sepal_length vs petal_length')
plt.legend()
plt.show()


# In[19]:


plt.figure(figsize=(10, 6))
for species in ir['species'].unique():
    subset = ir[ir['species'] == species]
    plt.scatter(subset['sepal_width'], subset['petal_width'], color=color_palette[species], label=species)
plt.xlabel('sepal_width')
plt.ylabel('petal_width')
plt.title('Scatter Plot of sepal_width vs petal_width')
plt.legend()
plt.show()


# In[20]:


features = ir.drop(columns='species')
correlation_matrix = features.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='viridis')
plt.title('Feature Correlation Matrix')
plt.show()


# In[21]:


label_encoder = LabelEncoder()
ir['species'] = label_encoder.fit_transform(ir['species'])


# In[22]:


X_data = ir.drop(columns='species')
y_data = ir['species']


# In[23]:


scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_data), columns=X_data.columns)


# In[24]:


X_train_set, X_test_set, y_train_set, y_test_set = train_test_split(X_scaled, y_data, test_size=0.3, random_state=42)


# In[25]:


classifiers = {
    "Logistic Regression Model": LogisticRegression(max_iter=200),
    "K-Nearest Neighbor Model": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree Model": DecisionTreeClassifier(max_depth=3)
}


# In[26]:


for name, clf in classifiers.items():
    clf.fit(X_train_set, y_train_set)
    predictions = clf.predict(X_test_set)
    acc = accuracy_score(y_test_set, predictions) * 100
    print(f"\nAccuracy with {name}: {acc:.2f}%")


# In[ ]:




