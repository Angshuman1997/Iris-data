# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 15:28:28 2019

@author: Shivank.Agarwal
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

data = pd.read_csv('Iris.csv')
data
real_x= data.iloc[:,[1,2,3,4]].values
real_x
real_y=data.iloc[:,5].values
real_y
training_x,test_x,training_y,test_y=train_test_split(real_x,real_y,test_size=0.30,random_state=0)
model = LogisticRegression()

scaler=StandardScaler()
training_x=scaler.fit_transform(training_x)
test_x=scaler.fit_transform(test_x)

classifer_LR= LogisticRegression(random_state=0)
classifer_LR.fit(training_x,training_y)

y_pred= classifer_LR.predict(test_x)
y_pred 
test_y
classifer_LR.score(training_x,training_y)

 c_m=confusion_matrix(test_y,y_pred)
 c_m
 
 
import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(c_m, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
 
 