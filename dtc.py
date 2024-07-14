# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 12:13:31 2024

@author: 91961
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:24:41 2024

@author: 91961
"""

'''
0. Data collection data understanding  with pandas
1. Data pre processing
2. Data Visualisation
3. Model Building
4. Model Evaluation
5. Result
'''
import pandas as pd
data = pd.read_csv(r"C:\Users\91961\OneDrive\Desktop\gen ai DR\datasets\iris_raw.csv")
x = data.iloc[ : , :-1].values
y = data.iloc[ : , -1].values

from sklearn.model_selection import train_test_split
dic={}

    
xtest,xtrain,ytest,ytrain = train_test_split(x,y,test_size=0.3,random_state=2111)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=(5))

model.fit(xtrain,ytrain)

ypred = model.predict(xtest)

from sklearn.metrics import accuracy_score as acs
print(acs(ytest,ypred)*100)