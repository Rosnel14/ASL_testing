#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install pandas')


# In[4]:


import numpy as np
import pandas as pd 

data_train = pd.read_csv("sign_mnist_train.csv")
data_test = pd.read_csv("sign_mnist_test.csv")
#data_train.head()
#data_test.head()


# In[15]:


X_train = data_train.drop("label", axis= 1)
X_train.head()


# In[16]:


Y_train = data_train['label']
Y_train.head()


# In[18]:


x_test = data_test.drop("label", axis = 1)
x_test.head()


# In[19]:


y_test = data_test['label']
y_test.head()


# In[23]:


get_ipython().system('pip install sklearn')


# In[24]:


from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=0,max_depth=50)
clf.fit(X_train, Y_train)


# In[25]:


print(clf.score(x_test,y_test))


# In[27]:


import matplotlib.pyplot as plt
import sklearn.tree as tree

plt.figure(num=None, figsize=(150, 150 ))
tree.plot_tree(clf, filled=True)
plt.show()


# In[31]:


clf.get_depth()


# In[ ]:




