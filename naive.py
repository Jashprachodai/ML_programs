# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 09:41:23 2024

@author: 91961
"""
import pandas as pd

data = pd.read_csv(r"C:\Users\91961\OneDrive\Desktop\gen ai DR\datasets\iris.csv")

data = data.drop(["color"],axis=1)

data["pw"] = data["pw"].fillna(data["pw"].mean())

x = data.iloc[ : , :-1].values
y = data.iloc[ : ,-1].values

from sklearn.model_selection import train_test_split as tts

xtrain,xtest,ytrain,ytest = tts(x,y,test_size=0.3,random_state=(87))

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb = gnb.fit(xtrain,ytrain)

ypred = gnb.predict(xtest)

from sklearn.metrics import accuracy_score

print(accuracy_score(ytest, ypred)*100)

from sklearn.metrics import classification_report
print(classification_report(ytest, ypred))