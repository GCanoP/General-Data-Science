# -*- coding: utf-8 -*-
"""
DOUBLE AND TRIPLE SMOOTHING METHODS
author : Gerardo Cano Perea
date : December 17, 2000

[level] - [trend] - [model] - [predict]
"""

# Importing Packages. 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing datsets. 
df = pd.read_csv('airline_passengers.csv', index_col = 'Month', parse_dates = True)
df.dropna(inplace = True)
df.index.freq = 'MS'

"""
SIMPLE EXPONENTIAL SMOOTHING
HOLT - WINTERS SIMPLE METHOD
"""
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

span = 12
alpha = 2/(span+1)

# In df EWMA alpha = alpha is equivalent to span = span = [12] 
# For not know reason, when optimised = False is passed to function .fit(), the function
# statsmodels SimpleExpSmoothing deplace the fitted values one row below
# We can fix this problem applying the method .shift(-1)
# span / alpha is included in .fit(Smoothing_level = alpha)

df['EWMA_12'] = df['Thousands of Passengers'].ewm(alpha = alpha, adjust = False).mean()
df['SE_12'] = SimpleExpSmoothing(df['Thousands of Passengers']).fit(smoothing_level = alpha, optimized = False).fittedvalues.shift(-1)
print(df.head())

"""
DOUBLE EXPONENTIAL SMOOTHING
HOL - WINTER DOUBLE METHOD
Smoothness Parameter : [Beta] - Tend in dataset. 
[Addittive] is for lineal trends.
[Multiplicative] is for exponential trends.
"""

# Calculating the Double Exponential Smoothing for 12 months span
from statsmodels.tsa.holtwinters import ExponentialSmoothing
df['DES_Add_12'] = ExponentialSmoothing(df['Thousands of Passengers'], trend = 'add').fit().fittedvalues.shift(-1)
df['DES_Mul_12'] = ExponentialSmoothing(df['Thousands of Passengers'], trend = 'mul').fit().fittedvalues.shift(-1)
print(df.head())

# Plotting values in comparison. 
df[['Thousands of Passengers','EWMA_12','DES_Add_12']].iloc[:24].plot(figsize = (15,10)).autoscale(axis = 'x', tight = True)
df[['Thousands of Passengers','EWMA_12','DES_Mul_12']].iloc[:24].plot(figsize = (15,10)).autoscale(axis = 'x', tight = True)

"""
TRIPLE EXPONENTIAL SMOOTHING
HOL - WINTER TRIPLE METHOD
Smooth Parameter [Beta] - Suport for Tendency
Smooth Parameter [L] - Support for Stationality
"""
# Calculating the Triple Exponential Smoothing for 12 months.
df['TES_Add_12'] = ExponentialSmoothing(df['Thousands of Passengers'], trend = 'add', seasonal = 'add', seasonal_periods = 12).fit().fittedvalues
df['TES_Mul_12'] = ExponentialSmoothing(df['Thousands of Passengers'], trend = 'mul', seasonal = 'mul', seasonal_periods = 12).fit().fittedvalues

# Plotting results. 
df[['Thousands of Passengers','TES_Add_12','TES_Mul_12']].plot(figsize = (15,10))

"""
STEPS TO MAKING PREDICTION FOR NEW VALUES
"""
# Step 1 Save the model in a new variable
# Model 1 : Triple Exponential Smoothing with Holt - Winters additive method. 
model_add = ExponentialSmoothing(df['Thousands of Passengers'], trend = 'add', seasonal = 'add', seasonal_periods = 12).fit()
# Model 2 : Triple Exponential Smoothing with Holt - Winters multiplicative method.  
model_mul = ExponentialSmoothing(df['Thousands of Passengers'], trend = 'mul', seasonal = 'mul', seasonal_periods = 12).fit()

# Step 2 Making a forecast.
# Temporal windows is defined by 36 months/3 years [base timesteps] in future
forecast_add = model_add.forecast(72)
forecast_mul = model_mul.forecast(72)

# Plotting the Forecast.
df['Thousands of Passengers'].plot(figsize = (15,10))
forecast_add.plot(label = 'TES Forecast Additive Model')
forecast_mul.plot(label = 'TES Forecast Multiplicative Model')
plt.legend()
plt.grid(True)
plt.show()





