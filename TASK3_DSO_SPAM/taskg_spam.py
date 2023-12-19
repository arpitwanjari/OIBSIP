#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# In[4]:


data = pd.read_csv('spam.csv', encoding='latin1')


# In[5]:


sns.countplot(x='v1', data=data)
plt.title('Distribution of ham vs spam')
plt.show()


# In[6]:


tfidf = TfidfVectorizer()
X = tfidf.fit_transform(data['v2'])


# In[7]:


y = data['v1']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[8]:


model = LogisticRegression()
model.fit(X_train, y_train)


# In[9]:


y_pred = model.predict(X_test)


# In[10]:


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# In[11]:


conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[13]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[ ]:




