"""
AUTO-REGRESSIVE MODEL EXAMPLE : AIRLINE PASSENGERS
@author: Gerardo Cano Perea
"""
# Importing Packages. 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AR
import statsmodels.tsa.stattools as sts

# Importing dataset. 
data = pd.read_csv('airline_passengers.csv')
data.Month = pd.to_datetime(data.Month)
data.set_index('Month', inplace = True)
data = data.asfreq('MS')
data = data.fillna(method = 'ffill')

# Split data into train/test
size = int(len(data) * 0.8)
[df_train, df_test] = [data.iloc[:size], data.iloc[size:]]

# Log Transformation. 
df_train_ts_log = np.log(df_train)

# Exponential Weight Moving Average. 
ewma_avg = pd.DataFrame(df_train_ts_log['Thousands of Passengers'].ewm(halflife = 12).mean())

# Removing Trend to Timeseries
df_train_ts_log_diff = df_train_ts_log - ewma_avg

# Applying a Dickey - Fuller test. 
# The p-value = 0.018 suggest that the time serie is stationary 
sts.adfuller(df_train_ts_log_diff['Thousands of Passengers'], autolag = 'AIC')

# Creatting the AR model. 
model_6 = AR(df_train_ts_log_diff['Thousands of Passengers'])
AR6_fit = model_6.fit(maxlag = 6, method = 'mle')
AR6_fit.summary()

# Making Predictions
start = 1
end = len(df_train)
predictions_6 = AR6_fit.predict(start = start, end = end, dynamic = False).rename('AR_6_Predictions')

# Plotting the results. 
df_train_ts_log_diff.plot()
predictions_6.plot()
plt.grid(True)
plt.show()

# Reconvert
# Cumsum for cancel the differentiation. 
# Real value  = Base value + Prediction
# df_train = df_train_ts_log + prediction_6 or ewma + predictions_6

predictions_6 = pd.DataFrame(predictions_6)
predictions_6_log = ewma_avg['Thousands of Passengers'] + predictions_6['AR_6_Predictions']
predictions_6_log = pd.DataFrame(predictions_6_log)

predictions_6_exp = np.exp(predictions_6_log)
predictions_6_exp = predictions_6_exp[1:]
predictions_6_exp['Predictions'] = predictions_6_exp[1:]

predictions_6_exp['Predictions'][1:].plot(legend = 'Predicted')
df_train['Thousands of Passengers'].plot()
plt.legend()
plt.grid(True)
plt.show()
