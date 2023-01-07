"""
========================================================================================================================
XGBOOST MODEL.
author : Gerardo Cano Perea.
========================================================================================================================
"""

# Importing Libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import dataset.
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Categorical Data.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder()
labelencoder_X_2 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories = 'auto'), [1])], remainder = 'passthrough')
X = onehotencoder.fit_transform(X)
X = X[:, 1:]
