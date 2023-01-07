"""
AUTO ARIMA MODEL FOR PASSENGERS DATASET. 
author : Gerardo Cano Perea. 
date : December 28, 2020.
"""

#Importing Packages. 
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt
from statsmodels.tsa.ar_model import AR
import statsmodels.tsa.stattools as sts
from pmdarima.arima import ADFTest


# Importing dataset. 
data = pd.read_csv('airline_passengers.csv')
data.Month = pd.to_datetime(data.Month)
data.set_index('Month', inplace = True)
data = data.asfreq('MS')
data = data.fillna(method = 'ffill')

# Applying Dickey - Fuller test. 
# Null Hypothesis : The timeserie is not stationary if p-value > 0.05
# The p-value is 0.9918. The time serie is not stationary. 
sts.adfuller(data['Thousands of Passengers'], autolag ='AIC')

# Applying the Dickey - Fuller test from [pmdarima]
# False : The timeserie is not stationary. 
alpha = 0.05
adf_test = ADFTest(alpha = alpha)
adf_test.should_diff(data['Thousands of Passengers'])

# Split into train/test datasets
size = int(len(data) * 0.8)
[df_train, df_test] = [data.iloc[:size], data.iloc[size:]]

# Warning treatments
import warnings
warnings.filterwarnings('ignore')

df_train['Thousands of Passengers'].plot()
df_test['Thousands of Passengers'].plot() 
plt.title('Trian/Test Data for Airline Passengers CSV')
plt.grid(True)
plt.show()

# Transforming Data 1. Log Transformation / 2. Differential Transformation.
df_train_ts_log = np.log(df_train)
df_train_ts_log_diff = pd.DataFrame(df_train_ts_log['Thousands of Passengers'] - df_train_ts_log['Thousands of Passengers'].shift())
df_train_ts_log_diff_2 = pd.DataFrame(df_train_ts_log_diff['Thousands of Passengers'] - df_train_ts_log_diff['Thousands of Passengers'].shift())

# Applying a Dickey Fuller Tests 
# First differentiation is stationary with a p-value 0.08 but only for alpha = 0.01
# Second differentiation is completely stationary with a p-value = 5.9229e-11 
sts.adfuller(df_train_ts_log_diff['Thousands of Passengers'][1:])
sts.adfuller(df_train_ts_log_diff_2['Thousands of Passengers'][2:])

# Construction of model with second differentiation.
model_5 = AR(df_train_ts_log_diff_2['Thousands of Passengers'][2:])
AR5_fit = model_5.fit(maxlag = 10, method = 'mle')
AR5_fit.summary()

# Making Predictions
start = 1
end = 140
predictions_5_ts_log_diff_2 = AR5_fit.predict(start = start, end = end, dynamic = False).rename('AR_5_Predictions')

# Plotting Results
# Plotting the results. 
df_train_ts_log_diff_2.plot()
predictions_5_ts_log_diff_2.plot()
plt.grid(True)
plt.show()

predictions_5_ts_log_diff = predictions_5_ts_log_diff_2.cumsum()
predictions_5_ts_log = predictions_5_ts_log_diff.cumsum()
predictions_5 = pd.DataFrame(np.exp(predictions_5_ts_log_diff_2))
