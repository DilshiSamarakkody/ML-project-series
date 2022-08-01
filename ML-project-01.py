#!/usr/bin/env python
# coding: utf-8

#  ## Addition of two numbers using ML

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv('dataset.csv')


# In[3]:


data


# In[4]:


data.head()


# In[5]:


data.tail()


# In[6]:


data.shape


# #### To identify whether there are any missing values in the data set

# In[7]:


data.info()


# ## Exploratory Data Analysis

# In[8]:


import matplotlib.pyplot as plt


# In[9]:


plt.scatter(data['x'],data['sum'])


# In[10]:


plt.scatter(data['y'],data['sum'])


# #### Store feature matrix in X and target in vector y

# In[11]:


X = data[['x','y']]


# In[12]:


y = data['sum']


# #### split the data set as training set and testing set

# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.33, random_state=42)


# #### Training set

# In[14]:


X_train


# ### Train the model

# In[15]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)


# ### Model prediction performance

# In[16]:


model.score(X_train,y_train)
model.score(X_test,y_test)


# #### Compare the results

# In[17]:


y_pred = model.predict(X_test)
y_pred


# In[18]:


y_test


# In[19]:


df = pd.DataFrame({'Actual':y_test,'prediction':y_pred})
df


# ### Prediction on new samples

# In[20]:


import warnings
warnings.filterwarnings('ignore')


# In[21]:


model.predict([[100,82]])


# #### Save the model using joblib

# In[22]:


import joblib
joblib.dump(model,'model_joblib')


# #### Load the model

# In[23]:


model = joblib.load('model_joblib')
model.predict([[23,45]])


# ### Training the entire data set

# In[24]:


X = data[['x','y']]
y = data['sum']

model = LinearRegression()
model.fit(X,y)


# In[25]:


import joblib
joblib.dump(model,'model_joblib')
model = joblib.load('model_joblib')
model.predict([[23,45]])


# In[ ]:


def show_entry_fields():
    p1=float(e1.get())
    p2=float(e2.get())
    
    model = joblib.load('model_joblib')
    result = model.predict([[p1,p2]])
    
    Label(master,text='sum is = ').grid(row=4)
    Label(master,text='sum is = ').grid(row=4)


    print("sum is ",result)
    

from tkinter import*
import joblib
master = Tk()
master.title("Addition of two numbers using ML")
label = Label(master,text="Addition of two numbers using ML",bg ='black',fg='white').grid(row=0,columnspan=2)

Label(master,text="Enter first number").grid(row=1)
Label(master,text="Enter second number").grid(row=2)

e1=Entry(master)
e2=Entry(master)

e1.grid(row=1,column=1)
e2.grid(row=2,column=1)

Button(master,text='predict',command=show_entry_fields).grid()

mainloop()


# In[ ]:




