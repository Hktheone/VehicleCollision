#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn import *


# In[2]:


# Importing dataset and examining it
dataset = pd.read_csv("/content/drive/MyDrive/VehicleCollisions.csv")
pd.set_option('display.max_columns', None) # to make sure you can see all the columns in output window
print(dataset.head())
print(dataset.shape)
print(dataset.info())
print(dataset.describe())


# In[3]:


# Converting Categorical features into Numerical features
le=LabelEncoder()
def labeling(columName):
  dataset[columName]=le.fit_transform(dataset[columName])


# In[4]:


labeling('Road_Type')
labeling('Speed_limit')
labeling('Light_Conditions')
labeling('Weather_Conditions')
labeling('Road_Surface_Conditions')
labeling('Urban_or_Rural_Area')
labeling('Vehicle_Manoeuvre')
labeling('1st_Point_of_Impact')
labeling('Sex_of_Driver')
labeling('Damage')


# In[5]:


dataset.head


# In[6]:


# Dividing dataset into label and feature sets
X = dataset.drop('Damage', axis = 1) # Features
Y = dataset['Damage'] # Labels
print(type(X))
print(type(Y))
print(X.shape)
print(Y.shape)


# In[7]:


#Using corelation to find the fearures that highly affect the results
cor = dataset.corr()
cor_target = cor['Damage']

#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.01]

#printin
print(relevant_features)


# In[8]:


# Normalizing numerical features so that each feature has mean 0 and variance 1
feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)


# In[9]:


# Dividing dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split( X_scaled, Y, test_size = 0.3, random_state = 100)


# In[10]:


print("Number of observations in each class before oversampling (training data): \n", pd.Series(Y_train).value_counts())


# In[11]:


smote = SMOTE(random_state = 101)
X_train,Y_train = smote.fit_sample(X_train,Y_train)


# In[12]:



print("Number of observations in each class after oversampling (training data): \n", pd.Series(Y_train).value_counts())


# In[13]:


# Building random forest model
rfc = RandomForestClassifier(n_estimators=300, criterion='entropy', max_features='auto')
rfc.fit(X_train,Y_train)
Y_pred = rfc.predict(X_test)
conf_mat = metrics.confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_mat,annot=True)
plt.title("Confusion_matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual class")
plt.show()
print('Confusion matrix: \n', conf_mat)
print('TP: ', conf_mat[1,1])
print('TN: ', conf_mat[0,0])
print('FP: ', conf_mat[0,1])
print('FN: ', conf_mat[1,0])


# In[14]:


# Tuning the random forest parameter 'n_estimators' and implementing cross-validation using Grid Search
rfc = RandomForestClassifier(criterion='entropy', max_features='auto', random_state=1)
grid_param = {'n_estimators': [50, 100, 150, 200, 250, 300]}


# In[15]:


gd_sr = GridSearchCV(estimator=rfc, param_grid=grid_param, scoring='recall', cv=5)


# 
# """
# In the above GridSearchCV(), scoring parameter should be set as follows:
# scoring = 'accuracy' when you want to maximize prediction accuracy
# scoring = 'recall' when you want to minimize false negatives
# scoring = 'precision' when you want to minimize false positives
# scoring = 'f1' when you want to balance false positives and false negatives (place equal emphasis on minimizing both)
# """

# In[16]:


gd_sr.fit(X_train, Y_train)


# In[17]:


best_parameters = gd_sr.best_params_
print(best_parameters)


# In[18]:


best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
print(best_result)


# In[19]:


# Building random forest using the tuned parameter
rfc = RandomForestClassifier(n_estimators=400, criterion='entropy', max_features='auto', random_state=1)
rfc.fit(X_train,Y_train)
featimp = pd.Series(rfc.feature_importances_, index=list(X)).sort_values(ascending=False)
print(featimp)


# In[20]:


Y_pred = rfc.predict(X_test)
conf_mat = metrics.confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_mat,annot=True)
plt.title("Confusion_matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual class")
plt.show()
print('Confusion matrix: \n', conf_mat)
print('TP: ', conf_mat[1,1])
print('TN: ', conf_mat[0,0])
print('FP: ', conf_mat[0,1])
print('FN: ', conf_mat[1,0])


# In[21]:


print('Accuracy score before feature selection: ',metrics.accuracy_score(Y_test,Y_pred)*100)


# In[57]:


# Selecting features with higher sifnificance and redefining feature set
X = dataset[[ 'Speed_limit','Sex_of_Driver','Urban_or_Rural_Area', 'Road_Type','Number_of_Vehicles','Road_Surface_Conditions','Age_of_Driver','1st_Point_of_Impact','Light_Conditions','Weather_Conditions','Vehicle_Manoeuvre']]


# In[58]:


feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)


# In[59]:


# Dividing dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split( X_scaled, Y, test_size = 0.3, random_state = 100)


# In[60]:


smote = SMOTE(random_state = 101)
X_train,Y_train = smote.fit_sample(X_train,Y_train)


# In[61]:


rfc = RandomForestClassifier(n_estimators=400, criterion='entropy', max_features='auto', random_state=1)
rfc.fit(X_train,Y_train)


# In[62]:


Y_pred = rfc.predict(X_test)
conf_mat = metrics.confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_mat,annot=True)
plt.title("Confusion_matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual class")
plt.show()
print('Confusion matrix: \n', conf_mat)
print('TP: ', conf_mat[1,1])
print('TN: ', conf_mat[0,0])
print('FP: ', conf_mat[0,1])
print('FN: ', conf_mat[1,0])


# In[63]:


print('Accuracy score: ',metrics.accuracy_score(Y_test,Y_pred)*100)


# In[ ]:




