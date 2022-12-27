# -*- coding: utf-8 -*-

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

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report


#df = pd.read_csv ('/gdrive//MyDriveCAS/M3/bank-additional/bank-additional-full.csv', sep= ';', na_values =('NaN', ''), keep_default_na = False)
df = pd.read_csv ("D:/CAS/bank-additional-full.csv", sep= ';', na_values =('NaN', ''), keep_default_na = False)
df.head()

df = df.drop('duration', axis=1)
df.groupby("y").mean()

#pip install dtreeviz

#from dtreeviz.trees import dtreeviz # remember to load the package

useful_fields = ['age','job',
                  'marital', 'education', 'default',
                  'housing', 'loan', 'contact', 'month','campaign', 
                  'pdays', 'previous', 'poutcome',
                  'emp.var.rate', 'cons.price.idx','cons.conf.idx',
                  'euribor3m', 'nr.employed',
                  ]
target_field = {'y'}

#df.dropna(axis=0, subset=useful_fields+[target_field], inplace=True)
#should we add weekday etc as well as useful field?

cleanup_nums = {  'job':         { 'admin.':0,'blue-collar':1,'entrepreneur':2,'housemaid':3,'management':4,'retired':5,'self-employed':6,'services':7,
                                  'student':8,'technician':9,'unemployed':10,'unknown':11},
                  'marital':     { 'divorced':0,'married':1,'single':2,'unknown':3},
                  'education':   {'basic.4y':0,'basic.6y':1,'basic.9y':2,'high.school':3,'illiterate':4,'professional.course':5,'university.degree':6,'unknown':7},
                  'default':     { 'no':0,'yes':1,'unknown':2},
                  'housing':     { 'no':0,'yes':1,'unknown':2},
                  'loan':        { 'no':0,'yes':1,'unknown':2},
                  'contact':     { 'cellular':1,'telephone':2},
                  'month':       { 'jan':1,'feb':2, 'mar':3,'apr':4, 'may':5,'jun':6, 'jul':7,'aug':8, 'sep':9,'oct':10, 'nov':11,'dec':12},
                  'poutcome':    { 'failure':0,'nonexistent':1,'success':2},
                }
cleanup_y = {'y': {'yes':1, 'no':0}}

def dataset():
 df = pd.read_csv ("D:/CAS/bank-additional-full.csv", sep= ';', na_values =('NaN', ''), keep_default_na = False)
 df_X = df[useful_fields].copy()                              
 df_X.replace(cleanup_nums, inplace=True)  # convert continous categorial variables to numerical
 df_Y = df[target_field].copy()
 df_Y.replace (cleanup_y, inplace= True)   #convert continous categorial Y to numerical

 x = df_X.to_numpy().astype(np.float32)

 y = df_Y.to_numpy().astype(np.float32)
 return (x, y, df)

x, y, df = dataset()
print(x.shape, y.shape)
df.head()

df[useful_fields].head()

df[useful_fields].describe ()

df_X = df[useful_fields].copy()              # this is now the new dataframe with only numerical                
df_X.replace(cleanup_nums, inplace=True)
df_X.describe ()

df_Y = df[target_field].copy()
df_Y.replace (cleanup_y, inplace= True)
df_Y.describe ()

df_new=pd.concat([df_X, df_Y], axis=1)
df_new.describe()

time_previous=df_new["pdays"]
plt.hist(time_previous, bins=1000)
plt.ylim((0,500))
plt.xlim((0,100))
plt.show()

contact_before=df_new["previous"]
plt.hist(contact_before, bins=100)
plt.ylim((0,5000))
plt.show()

success_previous=df_new["poutcome"]
plt.hist(success_previous, bins=20)
plt.ylim((0,5000))
plt.show()

df["y"].value_counts(" ")

df_new["y"].head

df_new["y"].value_counts()

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(aspect="equal"))

split = ["4640",
          "36548"]

data = [float(x.split()[0]) for x in split]
sales = [x.split()[-1] for x in split]


def func(pct, allvals):
    absolute = int(np.round(pct/100.*np.sum(allvals)))
    return "{:.1f}%\n({:d} )".format(pct, absolute)


wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),
                                  textprops=dict(color="w"))

ax.legend(wedges, sales,
          title="New sales",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.setp(autotexts, size=12, weight="bold")

ax.set_title("Success of campaign")

plt.show()

df.groupby("y").mean()

df_new.groupby("y").mean()

df_2=df_new [['age', 'campaign','pdays', 'previous', 'poutcome','y']].copy()
df_2.groupby("y").mean()

corr = df_new.corr()
corr.style.background_gradient(cmap='Spectral') #PiYG



# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
pd.crosstab(df.job,df.y).plot(kind='bar')
plt.title('Purchase Frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_fre_job')

# Commented out IPython magic to ensure Python compatibility.
#  %matplotlib inline
pd.crosstab(df.poutcome,df.y).plot(kind='bar')
plt.title('Purchase Frequency for Success of previous campaign')
plt.xlabel('Previous campaign')
plt.ylim((0,5000))
plt.ylabel('Frequency of Purchase')

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
pd.crosstab(df.month,df.y).plot(kind='bar')
plt.title('Purchase Frequency for Month')
plt.xlabel('Month')
plt.ylabel('Frequency of Purchase')
plt.savefig('pur_fre_month_bar')

# Logistic Regression

x_train, x_test, y_train, y_test =train_test_split(x,y, test_size=0.2) #try stratified split
for multi_class in ('auto', 'ovr'):
    # do fit
    clf = linear_model.LogisticRegression(solver='sag', max_iter=2000,
                             multi_class=multi_class, )
    clf.fit(x_train, y_train.ravel())

    # print the training scores
    print("testing accuracy : %.3f (%s)" % (clf.score(x_test, y_test.ravel()), multi_class))

reg = linear_model.LogisticRegression(solver='lbfgs', max_iter=2000)  
reg.fit(x_train, y_train.ravel())
print("testing accuracy : %.3f (%s)" % (reg.score(x_test, y_test.ravel()), multi_class))

from sklearn.metrics import confusion_matrix
y_pred_train=reg.predict(x_train)
y_pred_test=reg.predict(x_test)
confusion_matrix_train = confusion_matrix(y_train, y_pred_train)
confusion_matrix_test = confusion_matrix(y_test, y_pred_test)
#print(confusion_matrix_train)
print(confusion_matrix_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_test))

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)  #need to play around with SMOTE and random states still
x_res, y_res = sm.fit_resample(x_train, y_train)

reg = linear_model.LogisticRegression(solver='lbfgs', max_iter=2000)  
reg.fit(x_res, y_res.ravel())
print("testing accuracy : %.3f (%s)" % (reg.score(x_test, y_test.ravel()), multi_class))

y_pred_train=reg.predict(x_res)
y_pred_test=reg.predict(x_test)
confusion_matrix_train = confusion_matrix(y_res, y_pred_train)
confusion_matrix_test = confusion_matrix(y_test, y_pred_test)
#print(confusion_matrix_train)
print(confusion_matrix_test)

print(classification_report(y_test, y_pred_test))

#Decision Tree



x_train, x_test, y_train, y_test =train_test_split(x,y, test_size=0.2)

# Da weiss ich nöd so gnau, was ich da mache...
dtcs = []
for depth in (1, 2, 3, 4, 6):
    # do fit
    dtc = tree.DecisionTreeClassifier(max_depth=depth, criterion='gini')  # 'entropy'
    dtcs.append(dtc)
    dtc.fit(x_train, y_train)

    # print the training scores
    #print("training score : %.3f (depth=%d)" % (dtc.score(x_train, y_train), depth))
    print("testing score : %.3f (depth=%d)" % (dtc.score(x_test, y_test), depth))

from sklearn.metrics import confusion_matrix

y_pred_train=dtc.predict(x_train)
y_pred_test=dtc.predict(x_test)
confusion_matrix_train = confusion_matrix(y_train, y_pred_train)
confusion_matrix_test = confusion_matrix(y_test, y_pred_test)
#print(confusion_matrix_train)
print(confusion_matrix_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_test))

#Principal Component Analysis
#Package required: from sklearn.decomposition import PCA


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

#Clustering with K-Means and Silhouette-Score

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
# Neuronal network

# Setup model (keras part of tensorflow)
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(18,)),
  tf.keras.layers.Dense(516, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid'),
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])
model.summary()

# for modeling
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping


#hist = model.fit(x=x_train, y=y_train, epochs=20, batch_size=128,
                # validation_data=(x_test, y_test), shuffle=True)
hist = model.fit(x, y, epochs=50, batch_size=128,
                validation_split=0.2, shuffle=True)

print("Before OverSampling, counts of label '1': {}".format(sum(y == 1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y == 0)))
  
# import SMOTE module from imblearn library
# pip install imblearn (if you don't have imblearn in your system)
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 2)
x_res, y_res = sm.fit_resample(x, y.ravel())
#x_res, y_res = sm.fit_resample(x_train, y_train)
  
print('After OverSampling, the shape of train_X: {}'.format(x_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_res.shape))
  
print("After OverSampling, counts of label '1': {}".format(sum(y_res == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_res == 0)))

hist_smote = model.fit(x_res, y_res, 
                 epochs=50, batch_size=128,
                 validation_split=0.2, shuffle=True)

history_dict = hist.history
# Learning curve(accuracy)
# let's see the training and validation accuracy by epoch

# accuracy
accuracy_values = history_dict['binary_accuracy'] # you can change this
val_accuracy_values = history_dict['val_binary_accuracy'] # you can also change this

#for Smote DAta
history_dict_smote = hist_smote.history
# accuracy
accuracy_smote = history_dict_smote['binary_accuracy'] # you can change this
val_accuracy_smote = history_dict_smote['val_binary_accuracy'] # you can also change this

# range of X (no. of epochs)
epochs = range(1, len(accuracy_values) + 1) 

# plot
plt.rcParams["figure.figsize"] = (9 , 5)
plt.plot(epochs, accuracy_values, 'red', label='Training accuracy')
plt.plot(epochs, val_accuracy_values, 'orange', label='Validation accuracy')
plt.plot(epochs, accuracy_smote, 'green', label='Training accuracy Smote')
plt.plot(epochs, val_accuracy_smote, 'lightgreen', label='Validation accuracy Smote')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

from sklearn.metrics import confusion_matrix
y_pred_test=model.predict(x)

y_pred_test.shape  #https://www.tensorflow.org/guide/keras/train_and_evaluate

x.shape

y_test.shape

#After smote

history_dict = hist_smote.history
# Learning curve(accuracy)
# let's see the training and validation accuracy by epoch

# accuracy
accuracy_values = history_dict['binary_accuracy'] # you can change this
val_accuracy_values = history_dict['val_binary_accuracy'] # you can also change this

# range of X (no. of epochs)
epochs = range(1, len(accuracy_values) + 1) 

# plot
plt.plot(epochs, accuracy_values, 'blue', label='Training accuracy')
plt.plot(epochs, val_accuracy_values, 'orange', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
