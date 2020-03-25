#!/usr/bin/env python
# coding: utf-8

# In[53]:


import numpy as np
from scipy.io import arff
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.ensemble
import sklearn.model_selection
import sklearn.metrics


# In[90]:


#carga los archivos
#los 64 atributos numéricos son datos financieros de las compañías. La clase nominal tiene valores 1 o 0 dependiendo
#si la compañía quebró o no.


# In[100]:


year1=pd.read_csv('1year.arff',skiprows=69)
year2=pd.read_csv('2year.arff',skiprows=69)
year3=pd.read_csv('3year.arff',skiprows=69)
year4=pd.read_csv('4year.arff',skiprows=69)
year5=pd.read_csv('5year.arff',skiprows=69)


# In[165]:


data=pd.concat([year1,year2,year3,year4,year5])
predictors=list(data.keys())
#predictors.remove('Unnamed: 0')
predictors.remove(' Quiebra')

#fuerzo a valor nulo a los valores '?'
for pred in predictors:
    data[pred]=pd.to_numeric(data[pred],errors='coerce')
#quito las compañías que tengan Nan. Lo intenté con los atributos, pero todos tienen al menos un valor Nan
data=data.dropna()


# In[167]:


x_train,x_test,y_train,y_test= sklearn.model_selection.train_test_split(data[predictors], data[' Quiebra'], test_size=0.5, random_state=1)
#Hago dos veces el corte. Primero, queda 50% para train. Luego, corto. el test y lo asigno a validación
x_test, x_val, y_test, y_val = sklearn.model_selection.train_test_split(x_test, y_test, test_size=0.4, random_state=1)


# In[163]:


clf = sklearn.ensemble.RandomForestClassifier(n_estimators=10, max_features='sqrt')


# In[206]:


n_trees = np.arange(1,20,1)
f1_train = []
f1_test = []
f1_val= []
feature_importance = np.zeros((len(n_trees), len(predictors)))

for i, n_tree in enumerate(n_trees):
    clf = sklearn.ensemble.RandomForestClassifier(n_estimators=n_tree, max_features='sqrt')
    clf.fit(x_train, y_train)
    f1_train.append(sklearn.metrics.f1_score(y_train, clf.predict(x_train)))
    f1_test.append(sklearn.metrics.f1_score(y_test, clf.predict(x_test)))
    f1_val.append(sklearn.metrics.f1_score(y_val, clf.predict(x_val)))
    feature_importance[i, :] = clf.feature_importances_


# In[215]:


avg_importance = np.average(feature_importance, axis=0)
a = pd.Series(avg_importance, index=predictors)
print(a[np.argsort(a)][-5:])
a.nlargest().plot(kind='barh')
plt.xlabel('Average Feature Importance')
plt.title(n_trees[np.argmax(f1_test)])
plt.title("M= {:.2f}  F1-score={:.2f}".format(n_trees[np.argmax(f1_test)], max(f1_val)))
plt.tight_layout()

plt.savefig('features.png')


# In[ ]:




