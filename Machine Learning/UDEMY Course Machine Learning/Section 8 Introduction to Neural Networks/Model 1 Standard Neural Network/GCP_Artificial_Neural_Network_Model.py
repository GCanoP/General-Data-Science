"""
==========================================================================================
ARTIFICIAL NEURAL NETWORK MODEL INTRODUCTION.
Introduccion a los Modelos de Redes Neuronales Artificiales.
author : Gerardo Cano Perea.
date : February 20, 2021.
==========================================================================================
INPUTS.
o The variables must be independent.
o A standardization / normalization is needed to avoid overfitting or underfitting.
OUTPUTS.
o The datatype can be : Continuous, Binary, Categorical.
ACTIVATION FUNCTION.
o Step Function / Threshold Function. Range [0, 1].
o Sigmoid Function. Range phi(X) = 1/(1+e^(-x)).
o ReLU Function (Rectifier Linear Unit).
o Tanh Function. Range phi(x) = (1-e^(-2x))/(1+e^(-2x))
A NEURAL NETWORK WITH ONLY ONE LAYER IS KNOWN AS A PERCEPTRON.
"""

# Relevant Packages.
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# Importing Dataset.
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Coding Categorical Variables.
from sklearn.preprocessing import LabelEncoder
label_encoder_X1 = LabelEncoder()
label_encoder_X2 = LabelEncoder()
X[:, 1] = label_encoder_X1.fit_transform(X[:, 1])
X[:, 2] = label_encoder_X2.fit_transform(X[:, 2])

# Coding Dummy Categorical Values / Check the Collinear Problem.
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
onehotencoder = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories = 'auto'), [1])], remainder = 'passthrough')
X = onehotencoder.fit_transform(X)
X = X[:, 1:]

# Split into Train/Test Sets
from sklearn.model_selection import train_test_split
[X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Applying a Standard Scaler.
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Building the Artificial Neural Network.
from keras.models import Sequential
from keras.layers import Dense
# Building the Structure of the Network.
# First Layer has 11 units, the second and third layers have 8 units.
model = Sequential()
model.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_shape = (11, )))
model.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
model.summary()
# Compiling and Fitting the Network.
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Predictions and Evaluate the Model.
y_pred = model.predict(X_test)
# Establish a Threshold for Data.
y_pred = (y_pred > 0.5)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm = confusion_matrix(y_test, y_pred.round())
acs = accuracy_score(y_test, y_pred, normalize=False)


