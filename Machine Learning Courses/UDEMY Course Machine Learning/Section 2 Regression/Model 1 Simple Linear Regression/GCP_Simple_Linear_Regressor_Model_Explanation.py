# Coding Example - Date October 20, 2020.
# Machine Learning Course UDEMY - Gerardo Cano Perea. 
# Simple Linear Regression Model. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix

# Diving the Dataset into [X, y] variables. 
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Dividing Dataset into Training and Testing Values.
[X_train, X_test, y_train, y_test]  = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Creating a Simple Linear Regression Model.
regression = LinearRegression()
regression.fit(X_train, y_train)

# Making a prediction in test data
y_pred = regression.predict(X_test)

# Visualizating the Data Results - Scatter Train
plt.scatter(X_train, y_train, color='royalblue')
plt.plot(X_train, regression.predict(X_train), color='salmon')
plt.title('Predicción de sueldo')
plt.xlabel('No. de años de experiencia')
plt.ylabel('Salario [USD]')
plt.grid(True)
plt.show()

# Visualizating the Data Results - Scatter Test
plt.scatter(X_test, y_test, color='royalblue')
plt.plot(X_train, regression.predict(X_train), color='salmon')
plt.title('Predicción de sueldo')
plt.xlabel('No. de años de experiencia')
plt.ylabel('Salario [USD]')
plt.grid(True)
plt.show()



