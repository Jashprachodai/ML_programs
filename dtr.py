# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:33:41 2024

@author: 91961
"""

import pandas as pd

data = pd.read_csv(r"C:\Users\91961\OneDrive\Desktop\gen ai DR\datasets\CarPrice_Assignment.csv")

data = data.dropna()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cols = ["CarName","carbody","drivewheel","enginetype","fuelsystem"]
for i in cols:
    data[i] = le.fit_transform(data[i].values)

    
x = data.iloc[ : , :-1]
y = data.iloc[ : ,-1]

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,random_state=5)

from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(random_state=(98))

model.fit(xtrain,ytrain)
ypred = model.predict(xtest)

from sklearn.metrics import r2_score
print(r2_score(ytest, ypred)*100)
