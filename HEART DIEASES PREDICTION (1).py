#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[3]:


df = pd.read_csv(r"C:\Users\hplap\Downloads\heart_disease_data.csv")
df.head()


# In[4]:


df.isnull().sum()


# In[5]:


df.tail()


# In[7]:


df.describe()


# In[10]:


df.info()


# In[11]:


df.isnull().sum()


# In[13]:


df['target'].value_counts()


# In[15]:


df.shape


# In[16]:


x = df.drop('target', axis = 1)
y = df['target']


# In[17]:


print(x)


# In[18]:


y


# In[19]:


x_train ,x_test , y_train ,y_test = train_test_split(x, y, test_size = 0.2 , stratify=y,random_state=2)


# In[22]:


print(x.shape , x_train.shape, y_train.shape)


# In[23]:


#LogisticRegression

model = LogisticRegression()
model.fit(x_train,y_train)


# In[26]:


#Accuracy Score
y_pred = model.predict(x_train)
training_accuracy = accuracy_score(y_train,y_pred)


# In[27]:


print("Accuracy of the training data : ", training_accuracy)


# In[29]:


y_test_accuracy = model.predict(x_test)
test_accuracy = accuracy_score(y_test , y_test_accuracy)

print("Accuracy of the test data  : ", test_accuracy )


# In[35]:


#Build the predictive System
input_shape = (26,1,0,146,218,0,1,105,0,2,1,1,3)

input_data_as_numpy_array = np.asarray(input_shape)
#reshape

input_data_as_reshape = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_as_reshape)
print(prediction)


if (prediction[0]==[0]):
    print("This Person does not have Heart Diease")
    
else:
    print("This Person has Heart Diease")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




