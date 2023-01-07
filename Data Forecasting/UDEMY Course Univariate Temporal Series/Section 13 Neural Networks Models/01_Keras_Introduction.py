"""
=========================================================================================================================
PYTHON KERAS API TENSORFLOW INTRODUCTION.
author : Gerardo Cano Perea.     date : May 25, 2021
=========================================================================================================================
"""

# Importing Main Packages.
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# Simulate a Linear Regression Model.
[m, b] = [2, 3]
x = np.linspace(0, 50, 100)
np.random.seed(101)
noise = np.random.normal(loc = 0.0, scale = 4.0, size = len(x))
y = 2 * x + b + noise
plt.plot(x, y, 'o', color = 'teal')

# ========================================
# Building a Simple Keras Neural Network.
# ========================================

from keras.models import Sequential
from keras.layers import Dense

# Build the Sequential Model.
model = Sequential()
model.add(Dense(units = 4, activation = 'relu', input_shape = (1,)))
model.add(Dense(units = 4, activation = 'relu'))
model.add(Dense(units = 1, activation = 'linear'))
model.summary()

# Compile and Fit the Model.
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x, y, batch_size = 1, epochs = 200, validation_split = 0.2, verbose = 1)

# Plot Loss.
loss = model.history.history['loss']
epochs = range(len(loss))
plt.plot(epochs, loss)

# Predict New Values.
x_for_prediction = np.linspace(0, 50, 100)
y_predicted = model.predict(x_for_prediction)

# Plot Predictions vs Real Values.
plt.plot(x, y, 'o', color = 'teal')
plt.plot(x_for_prediction, y_predicted, color = 'red')
plt.show()

# Evaluate the Error in Model.
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mae = mean_absolute_error(y, y_predicted)
mse = mean_squared_error(y, y_predicted)
r2 = r2_score(y, y_predicted)


