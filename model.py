
# coding: utf-8

# In[42]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import graphviz
import seaborn as sns


# In[5]:


df = pd.read_excel('Traffic data 2.xlsx')


# In[6]:


df.head()


# In[7]:


df.columns[3]


# In[8]:



for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(list(df[col].astype(str).values))

congestionDict = {0 : "none", 1 : "light", 2 : "average", 3 : "heavy"}
dayDict = {0 : "monday", 1 : "tuesday", 2 : "wednesday", 3 : "thursday", 4 : "friday", 5 : "saturday", 6 : "sunday"}
locationDict = {0 : "schoolGateInflow", 1 : "okeOdoInflow", 2 : "tipperGarageInflow"}

# In[9]:


classifier = DecisionTreeClassifier(max_depth = 11, random_state=4, criterion='gini')


# In[10]:


x = df[['Time','Day','Location']]
y = df[['Congestion']] 


# In[11]:


x_test,x_train,y_test,y_train = train_test_split(x,y, test_size=0.25,random_state=42)


# In[12]:


classifier.fit(x_train,y_train)


# In[13]:


print(classifier.score(x_test,y_test))


# In[26]:


plt.bar(df['Time'],df['Congestion'])
plt.show()


# In[76]:


plt.polar(df['Time'].head(),df['Congestion'].head())
plt.show()


# In[41]:


plt.scatter(df['Time'],df['Congestion'])
plt.show()


# In[24]:


plt.bar(df['Day'],df['Congestion'])
plt.show()


# In[72]:


sns.violinplot(df['Time'].head(100), df['Congestion'].head(100))


# In[17]:


data = tree.export_graphviz(classifier, out_file=None, feature_names=df.columns[:3],class_names=df.columns[3],filled=True)
graph = graphviz.Source(data, format="png")


# In[20]:


import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'


# In[21]:


graph.render('graph')

import joblib
joblib.dump(classifier, "model_jlib")