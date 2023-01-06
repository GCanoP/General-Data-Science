"""
POLYNOMICAL REGRESSION.
autor: Gerardo Cano Perea.  
date: December 14, 2020.

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset. 
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting the Lineal Regression with the complete Dataset.
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
 
# Fitting the Polynomical Regression with the complete Dataset. 
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
# The input now is a X_poly array. 
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Lineal Model Visualization. 
plt.scatter(X, y, color = 'royalblue')
plt.plot(X, lin_reg.predict(X), color = 'seagreen')
plt.title('Linear Regression Model')
plt.xlabel('Posición del empleado')
plt.ylabel('Sueldo')
plt.grid(True)
plt.show()

# Polynomical Model Visualization
# X_grid = np.arange(min(x), max(X), 0.1) 
# X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'royalblue')
plt.plot(X, lin_reg_2.predict(X_poly), color = 'seagreen')
plt.title('Polynomical Regression Model')
plt.xlabel('Posición del empleado')
plt.ylabel('Sueldo')
plt.grid(True)
plt.show()

# Predicting with Models
print(lin_reg.predict([[6.5]]))
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))


