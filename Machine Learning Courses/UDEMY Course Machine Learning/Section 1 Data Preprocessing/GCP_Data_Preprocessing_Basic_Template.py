# Coding Example - Date October 20, 2020.
# Machine Learning Course UDEMY - Gerardo Cano Perea. 

# Importing Libraries. 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Importing the Dataset. 
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Cleaning the NAN´s values. Pycarte Library. 
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Coding Categorical Datasets.
labelencoder_X = LabelEncoder()
labelencoder_y = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
y = labelencoder_y.fit_transform(y)

# Creating Dummy Binary Categorical Values. 
ct_X = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder='passthrough')
X = ct_X.fit_transform(X)
X = X.astype(float)

# Dividing Training and Testing Sets
[X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.2, random_state=0)

# Standarisation gives a Gauss Distribution - Normalisations gives a [0-1] Distribution
# Scaling Variables. It is valid to use MinMaxScaler from sklear Library
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
# X_train[:,3:5] = sc_X.fit_transform(X_train[:,3:5])
# Also is valid omit the Dummy variables in scaling Transform
X_test = sc_X.transform(X_test)
# X_test[:,3:5] = sc.transform(X_test[:,3:5])
# The same principle is applied to X_test variables.
# Transform Y_variables is not needed cause the algorithm is a CLASSIFICATION. 
# If algorithm were a PREDICT, thus a scaling is needed for a good convergence. 

# Applying a Decision Tree Clasiffication
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
tree.plot_tree(clf)
y_proof = clf.predict(X_test)
