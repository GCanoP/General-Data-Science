"""
=========================================================================================================================
RECURRENT NEURAL NETWORK BASIC MODEL EXPLANATION.
author : Gerardo Cano Perea.  date : May 25, 2021
=========================================================================================================================
Classical Syntax for a Neural Network. [Input] -> [Output]
Syntax for Recurrent Neural Network [Input_(t-1)] -> [Output] --> [Input_(t)] -> [Output] --> [Input_(t+1)] -> [Output]
Relevant Concept : MEMORY CELLS
____________________________________ LONG SHORT TERM MEMORY NETWORK ____________________________________________________
C1. Forget Gate Layer.
C2. Preserve Gate Layer. [Input Gate Layer, New Cell State Layer]
C3. Update Cell State Layer.
C4. Update Cell Memory Values.
"""
# ________________________ PREPROCESSING DATA _______________________________
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# Import Data.
df = pd.read_csv('Alcohol_Sales.csv', index_col = 'DATE', parse_dates = True)
df.index.freq = 'MS'
df.columns = ['Sales']
df.plot(figsize = (16, 7))

# Check Seasonality.
from statsmodels.tsa.seasonal import seasonal_decompose

results = seasonal_decompose(df['Sales'])
results.plot()

# Split into Train / Test sets.
train = df.iloc[:len(df) - 12]
test = df.iloc[len(df) - 12:]

# Scaling the Dataset.
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
sc_train = scaler.fit_transform(train)
sc_test = scaler.transform(test)

# ________________ Time Series Generator ______________________
from keras.preprocessing.sequence import TimeseriesGenerator

# Defining the generator.
n_input = 12  # Length of the Data Generator Array.
n_features = 1  # Number of Features per Array.
generator = TimeseriesGenerator(sc_train, sc_train, length = n_input, batch_size = 1)

# _________________ BUILD THE MODEL ____________________________
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Construct the Sequential Model.
model = Sequential()
model.add(LSTM(units = 100, activation = 'relu', input_shape = (n_input, n_features)))
model.add(Dense(units = 1))
model.summary()

# Compiling the Model.
model.compile(optimizer = 'adam', loss = 'mse')

# Fitting the Model.
model.fit_generator(generator, epochs = 50, verbose = 1)

# Check Errors.
model.history.history.keys()
loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)), loss_per_epoch)

# ____________________ EVALUATE TEST DATA______________________________
first_eval_batch = sc_train[-12:]
first_eval_batch = first_eval_batch.reshape((1, n_input, n_features))

print(model.predict(first_eval_batch))
print(sc_test[0])

# Create a Loop for Prediction.
test_prediction = []
first_eval_batch = sc_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test)):
    # Get the Prediction.
    current_pred = model.predict(current_batch)[0]
    # Save the Prediction.
    test_prediction.append(current_pred)
    # Update the batch.
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis = 1)

# Graph Inverse Data.
true_predictions = scaler.inverse_transform(test_prediction)
test['Predictions'] = true_predictions

test.plot(figsize = (18, 9))

model.save('LSTM_AS.h5')