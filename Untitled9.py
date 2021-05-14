#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from urllib.request import urlopen
import os
import sklearn
from sklearn.metrics import accuracy_score
import seaborn as sns


# In[2]:


url="https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
student = pd.read_csv(url)
print("imported succesfully")


# In[3]:


student.head(20)


# In[4]:


student


# In[5]:


student.size


# In[6]:


student.shape


# In[7]:


student.describe()


# In[8]:


student.info()


# In[9]:


#visualisation


# In[10]:


student.hist(figsize=(14,14))
plt.show()


# In[11]:


plt.bar(x=student['Hours'],height=student['Scores'])
plt.show()


# In[12]:


student


# In[13]:


numeric_columns=['Hours' , 'Scores']


# In[14]:


sns.pairplot(student[numeric_columns])


# In[15]:


sns.boxplot(y=student['Scores'])
plt.title("Boxplot")
plt.show()


# In[16]:


student.isnull().sum()


# In[17]:


corre=student.corr()
top_corr_features=corre.index
plt.figure(figsize=(10,10))
g=sns.heatmap(student,annot=True, cmap='gist_rainbow', cbar_kws={"orientation":"vertical"},linewidths=1)


# In[18]:


sns.violinplot(y=student['Hours'])
plt.title("Violinplot")
plt.show()


# In[19]:


#create paiplot and two barplots
plt.figure(figsize=(16,6))
plt.subplot(131)
sns.pointplot(x="Hours", y="Scores", data=student)
plt.legend(['Hours = 1', 'Scores = 0'])


# In[23]:


x,y=student.loc[:,:'Hours'],student.loc[:,'Scores']


# In[24]:


x


# In[25]:


y


# In[26]:


x .shape


# In[27]:


y .shape


# In[28]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[29]:


x=student.drop(['Scores'],axis=1)


# In[30]:


x


# In[31]:


x_train, x_test, y_train, y_test= train_test_split(x,y,random_state=10,test_size=0.3,shuffle=True)


# In[32]:


x_test


# In[33]:


print ("train_set_x shape: " + str(x_train.shape))
print ("train_set_y shape: " + str(y_train.shape))
print ("test_set_x shape: " + str(x_test.shape))
print ("test_set_y shape: " + str(y_test.shape))


# In[34]:


from sklearn.linear_model import LinearRegression


# In[35]:


linreg = LinearRegression()


# In[36]:


linreg.fit(x,y)


# In[37]:


from sklearn.linear_model import LinearRegression
regressor =LinearRegression()
regressor .fit(x_train, y_train)

print("completed!")


# In[38]:


print(x_test)
y_pred = regressor.predict(x_test)


# In[39]:


plt.scatter(x,y)
plt.plot(x, y_pred, colors='red')
plt.show()


# In[40]:


student = pd.DataFrame({'Actual' : y_test, 'Predicted': y_pred})
student


# In[41]:


Hours =9.25
own_pred = regressor.predict([[Hours]])
print("Number of hours = {}".format(Hours))
print("Predicted Score {}".format(own_pred[0]))


# In[ ]:




