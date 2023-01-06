"""
=============================================================================
RANDOM FOREST MODEL FOR REGRESSION. 
Modelo de Regresión por Bosques Aleatorios
author : Gerardo Cano Perea. 
date : December 3, 2020.
=============================================================================
Random Forests are a High Order Model for Joint Learning based on Decision Trees. 
Model Methodology or Steps.
Step 1. Choose a random number K from training dataset. 
Step 2. Construct a Decision Tree for every K value in training dataset.
Step 3. Choose the number of trees to model and repet Step 1 and Step 2. 
Step 4. For predicting a new value, the model uses all Decision Trees. This new 
        value will be the mean of all predictions performed.  
Additional : The mean could be sustituted by median value or a half- mean excluding 
the 5 % of the lowest and highest values to avoid non typical values.
"""

# Importing Relevant Packages. 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Importing Dataset. 
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# Building the Random Forest Model. 
from sklearn.ensemble import RandomForestRegressor
regression = RandomForestRegressor(n_estimators = 100, criterion = 'mse', random_state = 1)
regression.fit(X,y)

# Predicting with Random Forest. 
y_pred = regression.predict([[6.5]])

# Visualizating the model. 
plt.figure(figsize = (12,6))
plt.scatter(X, y, color = 'salmon')
plt.plot(X, regression.predict(X), color ='steelblue')
plt.title('Random Forest Model Prediction')
plt.xlabel('Position')
plt.ylabel('Salary [$]')
plt.grid(True)
plt.show()

# Visualizating the model grid. 
plt.figure(figsize = (12,6))
X_grid = np.linspace(min(X), max(X), 500)
plt.scatter(X, y, color = 'salmon')
plt.plot(X_grid, regression.predict(X_grid), color = 'steelblue')
plt.title('Random Forest Model Prediction')
plt.xlabel('Position')
plt.ylabel('Salary [$]')
plt.grid(True)
plt.show()

# R-Squared Metrics Implementation. 
from sklearn.metrics import r2_score
from sklearn.metrics import max_error
y_real = y
y_predicted = regression.predict(X) 
r2_score(y_real, y_predicted)
max_error(y_real, y_predicted)

