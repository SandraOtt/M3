# -*- coding: utf-8 -*-
"""M3_ModulWork.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1A_c3cWnqfwr3tqEIy0qoGdmqbzKy4AZl
"""

from sklearn import linear_model #https://scikit-learn.org/stable/user_guide.html
from sklearn import tree
from sklearn import ensemble

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.inspection import permutation_importance

#For Principal Component Analysis
from sklearn.decomposition import PCA

# common visualization module
from matplotlib import pyplot as plt
import plotly.express as px
import seaborn as sns
sns.set()

# numeric library
import numpy as np

import os
import pandas as pd
from time import time as timer
import tarfile

import tensorflow as tf

from google.colab import drive
drive.mount('/gdrive')

import zipfile


with zipfile.ZipFile('/gdrive/MyDrive/CAS/M3/bank-additional.zip','r') as source:
    source.extractall('/gdrive/MyDrive/CAS/M3')

df = pd.read_csv ('/gdrive/MyDrive/CAS/M3/bank-additional/bank-additional-full.csv', sep= ';', na_values =('NaN', ''), keep_default_na = False)
df.head()

#pip install dtreeviz

#from dtreeviz.trees import dtreeviz # remember to load the package

useful_fields = ['job',
                  'marital', 'education', 'default',
                  'housing', 'loan', 'contact', 'campaign',
                  'pdays', 'previous', 'poutcome',
                  'emp.var.rate', 'cons.price.idx','cons.conf.idx',
                  'euribor3m', 'nr.employed',
                  ]
target_field = {'y'}

#df.dropna(axis=0, subset=useful_fields+[target_field], inplace=True)

cleanup_nums = {  'job':         { 'admin.':0,'blue-collar':1,'entrepreneur':2,'housemaid':3,'management':4,'retired':5,'self-employed':6,'services':7,
                                  'student':8,'technician':9,'unemployed':10,'unknown':11},
                  'marital':     { 'divorced':0,'married':1,'single':2,'unknown':3},
                  'education':   {'basic.4y':0,'basic.6y':1,'basic.9y':2,'high.school':3,'illiterate':4,'professional.course':5,'university.degree':6,'unknown':7},
                  'default':     { 'no':0,'yes':1,'unknown':2},
                  'housing':     { 'no':0,'yes':1,'unknown':2},
                  'loan':        { 'no':0,'yes':1,'unknown':2},
                  'contact':     { 'cellular':1,'telephone':2},
                  'poutcome':    { 'failure':0,'nonexistent':1,'success':2},
                }
cleanup_y = {'y': {'yes':1, 'no':0}}

df_X = df[useful_fields].copy()                              
df_X.replace(cleanup_nums, inplace=True)  # convert continous categorial variables to numerical
df_Y = df[target_field].copy()
df_Y.replace (cleanup_y, inplace= True)   #convert continous categorial Y to numerical

x = df_X.to_numpy().astype(np.float32)

y = df_Y.to_numpy().astype(np.float32)

print(x.shape, y.shape)
df.head()

df.describe ()

"""# Linear Regression"""

#Split in Test and Train Set
x_train, x_test, y_train, y_test =train_test_split(x,y, test_size=0.2)

#macht lineare Regression überhaupt Sinn, wenn y nur 0 oder 1 ist?
reg = linear_model.LinearRegression()
reg.fit(x_train, y_train)

#Calculate meanstandarddeviation for train and test
print('train mse =', np.std(y_train - reg.predict(x_train)))
print('test mse =', np.std(y_test - reg.predict(x_test)))

# R2
print('train R2 =', reg.score(x_train, y_train))
print('test R2 =', reg.score(x_test, y_test))

# 4. plot y vs predicted y for test and train parts
plt.scatter(y_train, reg.predict(x_train))
plt.scatter(y_test, reg.predict(x_test))

"""#Decision Tree

"""

# Da weiss ich nöd so gnau, was ich da mache...
dtcs = []
for depth in (1, 2, 3, 4):
    # do fit
    dtc = tree.DecisionTreeClassifier(max_depth=depth, criterion='gini')  # 'entropy'
    dtcs.append(dtc)
    dtc.fit(x_train, y_train)

    # print the training scores
    print("training score : %.3f (depth=%d)" % (dtc.score(x_train, y_train), depth))
    print("training score : %.3f (depth=%d)" % (dtc.score(x_test, y_test), depth))

"""#Principal Component Analysis
Package required: from sklearn.decomposition import PCA
"""

pca=PCA() # Initialize PCA
pca.fit(df_X) # Call fit methode

#Plot Explained Variance Ratio
plt.plot(pca.explained_variance_ratio_,'-o')
plt.xlabel('Principal component')

#Plot as Sum
plt.plot(np.cumsum(pca.explained_variance_ratio_),'-o')
plt.title('Principal component as sum')

#get Columns

X_columns= df_X.columns
print(X_columns)
print(len(X_columns))

df = pd.DataFrame(pca.components_.transpose(), 
                  columns = [f'V_{i+1}' for i in range(len(X_columns))], 
                  index=X_columns)
df

# The PCA model
pca = PCA(n_components=2) # estimate only 2 PCs
X_new = pca.fit_transform(df_X) # project the original data into the PCA space

fig, axes = plt.subplots()

axes.scatter(X_new[:,0], X_new[:,1], c=df_Y)
axes.set_xlabel('PC1')
axes.set_ylabel('PC2')
axes.set_title('After PCA')
plt.show()

"""#Clustering with K-Means and Silhouette-Score

Packages required: 

*   from sklearn.cluster import KMeans
*   from sklearn.metrics import silhouette_score

"""

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

res= []

for iclust in range(2,10):
    clusterer = KMeans(n_clusters=iclust, random_state=10)
    cluster_labels = clusterer.fit_predict(df_X)
    score=silhouette_score(df_X,cluster_labels)
    res.append(score)
  
print(res)
# 3. Plot the Silhouette scores as a function ok k? What is the number of clusters ?
plt.plot(np.arange(len(res))+2, res, '-o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.title('Clustering')