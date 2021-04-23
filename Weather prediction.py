#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 11:12:27 2021

@author: bhaskaryuvaraj
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
ab=pd.read_csv('/Users/bhaskaryuvaraj/Downloads/weather (1).csv')
#retaining only temp and humidy column for learning in by 
ab.columns
by=ab.drop(ab.columns[[0,1,2,4,6,7,8,9]],axis=1)
#checking for data types
by.dtypes
by.isnull().sum()  #no null values

#--------------------------------EDA------------------------------------

by.plot(kind='scatter', x='Temperature..C.',y='Humidity')

#from above graph, it is clear that lesser the temperature, higher the humidity and vice versa

#-------------------------------EDA---------------------------------------

#outlier treatement.
plt.boxplot(by['Temperature..C.'])
plt.boxplot(by['Humidity']) #one outlier in humidity, so remove the outlier

#to remove the outlier

def remove_outlier(d,c):
    q1=d[c].quantile(0.25)
    q3=d[c].quantile(0.75)
    iqr=q3-q1
    ub=q3+1.53*iqr
    lb=q1-1.53*iqr
    result=d[(d[c]>lb) & (d[c]<ub)]
    return result

by=remove_outlier(by,'Humidity')

y=by.drop(by.columns[1],axis=1)
x=by.drop(by.columns[0],axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
#linear regression
lr=linear_model.LinearRegression()

model=lr.fit(x_train,y_train)

print(model.score(x_train,y_train))
#0.6770099487430917 accuracy

pred_y=lr.predict(x_test)
print(model.score(x_test,y_test))
#0.6711284103361527 accuracy

#-----------------------------------------------------------------------------------

ab.columns
ab=ab.drop(ab.columns[[0,2,8]],axis=1)
ab.isnull().sum()
ab.dtypes

#outiliers
plt.boxplot(ab['Temperature..C.'])
plt.boxplot(ab['Apparent.Temperature..C.'])
plt.boxplot(ab['Wind.Speed..km.h.'])
plt.boxplot(ab['Wind.Bearing..degrees.'])
plt.boxplot(ab['Pressure..millibars.'])

#no outliers

#dummies
dummy=pd.get_dummies(ab['Summary'])
ab1=pd.concat([ab,dummy],axis=1)
ab=ab1.drop(ab1.columns[[0,2]],axis=1)

##feature engineering
#correlated_features = set()
#correlation_matrix = ab.drop(ab.columns[0], axis=1).corr()
#
#for i in range(len(correlation_matrix.columns)):
#
#    for j in range(i):
#
#        if abs(correlation_matrix.iloc[i, j]) > 0.8:
#
#            colname = correlation_matrix.columns[i]
#
#            correlated_features.add(colname)
##Check correlated features            
#print(correlated_features)

#linear regression
x=ab.drop(ab.columns[0],axis=1)
y=ab['Temperature..C.'].copy()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

lr=linear_model.LinearRegression()

model=lr.fit(x_train,y_train)

print(model.score(x_train,y_train))
#accuracy=0.7376665501561744
pred_y=lr.predict(x_test)
print(model.score(x_test,y_test))
#test accuracy=0.4046358192994128

#conclusion
#1) with simple linear regression the accuracy of the train and test data is almost similar around 60%. 
#the accuracy can be improved with more data
#2) With mulitiple linear regression, there is a problem of overfitting model with accuracy of train 
#being 70% and and test being 40%. This can be solved by more data.

# the conclusion is muliple linear regression will be more accurate compared to simple linear provided with 
#more data leading to more accuracy.



