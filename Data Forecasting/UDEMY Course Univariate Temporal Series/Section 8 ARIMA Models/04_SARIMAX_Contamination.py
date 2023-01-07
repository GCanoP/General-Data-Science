"""
SARIMAX MODEL FOR CONTAMINATION DATA. 
author : Gerardo Cano Perea. 
date : December 31,2020.
"""
# Importing Packages. 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

# Importing the dataset. 
df = pd.read_csv('co2_mm_mlo.csv')

# Adding the [date] variable and setting as the index. 
df['date'] = pd.to_datetime(dict(year = df['year'], month = df['month'], day = 1))
df.set_index('date', inplace = True)
df.asfreq('MS')

# Plotting the dataset values for interpolated column. 
title = 'Monthly Mean Carbon Dioxide (C02) over Mauna Loa in Hawaii'
ylabel = 'Parts Per Million'
xlabel = ''
ax = df['interpolated'].plot(figsize = (12,6), title = title)
ax.autoscale(axis = 'x', tight = True)
ax.set(xlabel = xlabel, ylabel = ylabel)

# Decomposing the data into 3 main series factors. 
result = seasonal_decompose(df['interpolated'], model = 'additive')
result.plot() 

# Split the dataset into train/test. 
df_train = df.iloc[:717]
df_test = df.iloc[717:]

# SARIMA (0,1,1)(1,0,1,12) - There is no a X-exogenous variable. 
# The model order in infered at the best combination for the series.
model_sarima_011_10112 = SARIMAX(df_train['interpolated'], order = (0,1,1), seasonal_order = (1,0,1,12))
results_sarima_011_10112 = model_sarima_011_10112.fit()
results_sarima_011_10112.summary()

# Obtain predicted values.
start_test = len(df_train) 
end_test = (len(df_train) + len(df_test)) - 1
predictions_sarima_011_10112_test = results_sarima_011_10112.predict(start = start_test, end = end_test, typ = 'levels').rename('Predictions')
predictions_sarima_011_10112_test = pd.DataFrame(predictions_sarima_011_10112_test)

# Plotting the predicted values agains the known values. 
df_test['interpolated'].plot(legend = True, figsize = (12,6))
predictions_sarima_011_10112_test['Predictions'].plot(legend = True)
plt.title('Real Values vs Predicted SARIMA values')
plt.grid(True)
plt.show()

# Re-training the model with all values.
model_sarima_011_10112_full = SARIMAX(df['interpolated'], order = (0,1,1), seasonal_order = (1,0,1,12))
results_sarima_011_10112_full = model_sarima_011_10112_full.fit()
results_sarima_011_10112_full.summary()

# Making a Forecast. 
start_full = len(df)
end_full = len(df) + 36 
fcast_sarima_011_10112_full = results_sarima_011_10112_full.predict(start = start_full, end = end_full, typ = 'levels').rename('Predictions')
fcast_sarima_011_10112_full = pd.DataFrame(fcast_sarima_011_10112_full)

# Plotting results. 
df['interpolated'].plot(legend = True)
fcast_sarima_011_10112_full['Predictions'].plot(legend = True)
plt.title('Monthly Mean CO2 Levels (ppm) over Mauna Loa in Hawaii')
plt.grid(True)
plt.show()