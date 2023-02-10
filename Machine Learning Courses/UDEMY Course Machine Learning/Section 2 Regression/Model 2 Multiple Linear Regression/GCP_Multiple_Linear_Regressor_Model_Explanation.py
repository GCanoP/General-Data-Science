# Coding Example - Date December 10, 2020.
# Machine Learning Course UDEMY - Gerardo Cano Perea. 
# Multiple Linear Regression Model. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Split the Dataset into [X,y] variables.
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Solving for Dummy variables. 
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])

# Transforming Dummy Variables. 
ct_X = ColumnTransformer([("State", OneHotEncoder(),[3])], remainder='passthrough') 
X = ct_X.fit_transform(X)
X = X.astype(float)

# Avoid Multilineal Problems Filtering the Data.
X = X[:,1:]

# Split the dataset into train-test variables.
[X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size = 0.2, random_state = 0) 

# Fitting the Multiple Linear Regression.
regression = LinearRegression()
regression.fit(X_train, y_train)

# Predicting Results in Testing
y_pred = regression.predict(X_test)

# Back Suprissing Method 
X = np.append(arr = np.ones((50,1)).astype(int),  values = X, axis = 1)

""" Manual Backward Elimination
# Calculating the P-Value. The lower the p-value, highest will be the statistical importance.
SL = 0.05 # Significant Level
X_opt = X[:,[0,1,2,3,4,5]].tolist()
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regression_OLS.summary())

X_opt = X[:,[0,1,3,4,5]].tolist()
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regression_OLS.summary())

X_opt = X[:,[0,3,4,5]].tolist()
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regression_OLS.summary())

X_opt = X[:,[0,3,5]].tolist()
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regression_OLS.summary())

X_opt = X[:,[0,3]].tolist()
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regression_OLS.summary())
"""

# Automated Backward Elimination Method. 
def backwardElimination(x,sl):
    numVars = len(x[0])
    for i in range (0, numVars):
        regressor_OLS = sm.OLS(y, x.tolist()).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range (0, numVars-i): # Index [j] is used to find the position of the maxVar variable.
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    print(regressor_OLS.summary())
    return x

SL = 0.05
X_opt = X[:,[0,1,2,3,4,5]]
X_Modeled = backwardElimination(x = X_opt, sl = SL)



