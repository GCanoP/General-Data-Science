"""
=============================================================================
REGRESSION WITH SUPPORT VECTORIAL MACHINE.
Regresión por Maquinas de Soporte Vectorial.
author: Gerardo Cano Perea
date: December 14, 2020
=============================================================================
Support Vectorial Machine
o Are used for lineal and non lineal regressions/classifications applying a kernel. 
o Increase the dimensionality of the problem, here the problem will be solved.
o The SVM use a training data space T = {X,Y} to find a function F(X) = Y
___
Steps to Build a Support Vector Machine Model. 
1. Define a training dataset space T = {X,Y}
2. Choose a kernel and its parameters. 
3. If is needed, perform a data standardization or regularization.
4. Creat a correlation matrix K.
5. Define and train the model to get the contraction coefficients alpha = {alpha_i}
6. Use these coefficients to create a estimator f(X,alpha,x) = y
___
Kernels for SVM
o Lineal (x,y)
o Non Lineal (phi(x), phi(y)) = K(x,y)
o Gaussian.
IMPORTANT --> Check the equation for the CORRELATION MATRIX.
___
Optimization problem K * alpha = y, where:
[y] is a vector wich contains the values for training dataset.
[K] is the correlation matrix. 
[alpha] are the unknown values. 
A basic linear algebra solution is alpha = K^-1 * y

"""

# Importing Basic Packages. 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset.
dataset = pd.read_csv('Position_Salaries.csv') 
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values
y = y.reshape(10, 1)

# Scaling the Variables. 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Importing Specific Model Packages. 
# [c] --> Penalty parameter; usually is modified to avoid overfitting.
# [kernel = 'rbf'] --> Gaussian kernel; the default option for SVR model.
from sklearn.svm import SVR
regression = SVR(kernel = 'rbf')
regression.fit(X, y)

# Predicting Values.
value = sc_X.transform([[6.5]])
predict_y = regression.predict(value)
predict_original = sc_y.inverse_transform(predict_y)

# Plotting Dataset and Model Predictions. 
plt.scatter(X, y, color = 'salmon')
plt.plot(X, regression.predict(X), color = 'steelblue')
plt.title('SVR Model for Salaries Data')
plt.xlabel('Position')
plt.ylabel('Salary - $')
plt.grid(True)
plt.show()








    