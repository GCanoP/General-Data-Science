"""
=============================================================================
DECISION TREES FOR REGRESSION / CLASSIFICATION AND REGRESION TREES.
Regresión por Arboles de Decisiones. / Clasificación y Arboles de Regresión. 
author : Gerardo Cano Perea  
date : January 2, 2021
=============================================================================
Clasification Trees and Regressión Trees. 
o The model calculate the entrophy of information splitting the data into leaf nodes. 
o Every leaf node is equivalent to mean of all values inside the tree section. 
"""

# Importing Packages. 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the datasets. 
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values
y = y.reshape(10,1)

# Creating the Regression Model / Importing Packages.
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
regression = DecisionTreeRegressor(random_state = 0)
regression.fit(X,y)

# Predict a Specific Value.
y_pred = regression.predict([[5]])

# Viasualization the Model Plot. 
plt.scatter(X, y, color = 'salmon')
plt.plot(X, regression.predict(X), color = 'steelblue')
plt.title('Decision Trees for Regression Model')
plt.xlabel('Position')
plt.ylabel('Salary [$]')
plt.grid(True)
plt.show()

# Plotting the entire prediction. 
X_grid = np.linspace(min(X), max(X),100)
plt.scatter(X, y, color = 'salmon')
plt.plot(X_grid, regression.predict(X_grid), color = 'steelblue')
plt.title('Decision Tree for Regression Model')
plt.xlabel('Position')
plt.ylabel('Salary [$]')
plt.grid(True)
plt.show()

# Plotting the Decision Tree Structure.
y = plot_tree(regression, rounded = False)