import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import pickle
from flask import Flask
import pickle
import json
import numpy as np


data = pd.read_csv('data/rental_1000.csv')
#print(data)

x = data[['rooms','sqft']].values
y = data['price'].values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20)

#train model
model = LinearRegression()
model.fit(x_train,y_train)

#save model
filename = 'model/prediction_model.pkl'
pickle.dump(model, open(filename,'wb'))

#Evaluate model
predict = model.predict(x_test)
rmse = root_mean_squared_error(y_test,predict)
print(rmse)

print ("Model trained successfully")