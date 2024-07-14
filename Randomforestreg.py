# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 09:54:50 2024

@author: 91961
"""

import pandas as pd

data = pd.read_csv(r"C:\Users\91961\OneDrive\Desktop\gen ai DR\datasets\housing.csv",encoding="ISO-8859-1")


data["total_bedrooms"]  = data["total_bedrooms"].fillna(int(data["total_bedrooms"].mean()))
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

n = ["ocean_proximity"]
for i in n:
    data[i] = le.fit_transform(data[i].values)

x = data.drop(["ocean_proximity"],axis=1)
y = data["median_house_value"]

from sklearn.model_selection import train_test_split as tts

xtrain,xtest,ytrain,ytest = tts(x,y,test_size=0.3,random_state=(345))

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=125)

model.fit(xtrain,ytrain)

ypred = model.predict(xtest)

from sklearn.metrics import r2_score
print(r2_score(ytest, ypred)*100)

from sklearn.metrics import mean_squared_error
print(mean_squared_error(ytest, ypred))