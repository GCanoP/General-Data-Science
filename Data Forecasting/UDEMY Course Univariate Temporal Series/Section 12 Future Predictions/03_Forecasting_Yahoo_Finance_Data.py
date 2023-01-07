"""
==============================================================================================================
FORECASTING THE FUTURE WITH YAHOO FINANCE DATA.
author ; Gerardo Cano Perea.
date : May 23, 2021
==============================================================================================================
"""
# Import Packages.
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima import auto_arima
from arch import arch_model
import yfinance
import warnings
warnings.filterwarnings('ignore')
sns.set()

# Loading the Data.
raw_data = yfinance.download(tickers = '^GSPC ^FTSE ^N225 ^GDAXI', start = '1994-01-07', end = '2019-09-01', interval = '1d', group_by = 'ticker', auto_adjust = True, treads = True)
# Replace the dataset.
df_comp = raw_data.copy()
df_comp = pd.DataFrame(df_comp)
df_comp['spx'] = df_comp['^GSPC'].Close[:]
df_comp['dax'] = df_comp['^GDAXI'].Close[:]
df_comp['ftse'] = df_comp['^FTSE'].Close[:]
df_comp['nikkei'] = df_comp['^N225'].Close[:]
# Delete first row.
df_comp = df_comp.iloc[1:]
# Delete non used data.
del df_comp['^N225']
del df_comp['^GSPC']
del df_comp['^GDAXI']
del df_comp['^FTSE']
df_comp = df_comp.asfreq('b')
df_comp = df_comp.fillna(method = 'ffill')

# Creating Returns.
df_comp['ret_spx'] = df_comp.spx.pct_change(1).mul(100)
df_comp['ret_ftse'] = df_comp.ftse.pct_change(1).mul(100)
df_comp['ret_dax'] = df_comp.dax.pct_change(1).mul(100)
df_comp['ret_nikkei'] = df_comp.nikkei.pct_change(1).mul(100)
# Creating Normalized Returns.
df_comp['norm_ret_spx'] = df_comp.ret_spx.div(df_comp.ret_spx[1])*100
df_comp['norm_ret_ftse'] = df_comp.ret_spx.div(df_comp.ret_ftse[1])*100
df_comp['norm_ret_dax'] = df_comp.ret_spx.div(df_comp.ret_dax[1])*100
df_comp['norm_ret_nikkei'] = df_comp.ret_spx.div(df_comp.ret_nikkei[1])*100

# Splitting the Data.
size = int(len(df_comp)*0.8)
[df, df_test] = [df_comp.iloc[:size], df_comp.iloc[size:]]

# ===============================
# Fitting the Model ARIMA(0,0,1).
# ===============================

model_ar = ARIMA(df.ftse, order = (1, 0, 0))
results_ar = model_ar.fit()

# Create start and end dates.
start_date = '2014-07-15'
end_date = '2015-01-01'

# Predicting Data.
df_pred = results_ar.predict(start = start_date, end = end_date)

# Plot Predictions.
df_pred[start_date:end_date].plot(figsize = (18, 9), color = 'red')
plt.title('Predictions', size = 24)
plt.show()

# Check for a new date.
end_date = '2019-10-23'
df_pred = results_ar.predict(start = start_date, end = end_date)
df_pred[start_date:end_date].plot(figsize = (18, 9), color = 'red')
plt.title('Predictions', size = 24)
plt.show()

# Compare Predicted vs Actual.
df_pred[start_date:end_date].plot(figsize = (18, 9), color = 'red')
df_test.ftse[start_date:end_date].plot(color = 'blue')
plt.title('Prediction vs Real Value')
plt.legend()
plt.show()

# ============================
# Fitting with an ARIMA(0,0,1)
# ============================

end_date = '2015-01-01'
model_ret_ma = ARIMA(df.ret_ftse[1:], order = (0, 0, 1))
results_ret_ma = model_ret_ma.fit()

# Predicting Data.
df_pred_ma = results_ret_ma.predict(start = start_date, end = end_date)

df_pred_ma[start_date:end_date].plot(figsize = (18, 9), color = 'red')
df_test.ret_ftse[start_date:end_date].plot(color = 'blue')
plt.title('Prediction vs Actual (Returns)', size = 24)
plt.show()

# ==========================
# Fitting with an ARMAX Model
# ==========================

model_ret_armax = ARIMA(df.ret_ftse[1:], exog = df[['ret_spx', 'ret_dax', 'ret_nikkei']][1:], order = (1, 0, 1))
results_ret_armax = model_ret_armax.fit()

df_pred_armax = results_ret_armax.predict(start = start_date, end = end_date, exog = df_test[['ret_spx', 'ret_dax', 'ret_nikkei']][start_date:end_date])

df_pred_armax[start_date:end_date].plot(figsize = (18, 9), color = 'red')
df_test.ret_ftse[start_date:end_date].plot(color = 'blue')
plt.title('Predictions vs Actual (Returns) ARMAX Model')
plt.show()

# ==========================
# Fitting with SARMA Model.
# ==========================

end_date = '2015-01-01'
model_ret_sarma = SARIMAX(df.ret_ftse[1:], order = (3, 0, 4), seasonal_order = (3, 0, 2, 5))
results_ret_sarma = model_ret_sarma.fit()

df_pred_sarma = results_ret_sarma.predict(start = start_date, end = end_date)

df_pred_sarma[start_date:end_date].plot(figsize = (18, 9), color = 'red')
df_test.ret_ftse[start_date:end_date].plot(color = 'blue')
plt.title('Prediction vs Real (Returns) for a SARMA Model', size = 20)
plt.show()

# ======================================
# CHOOSE THE BEST MODEL WITH AUTOARIMA.
# ======================================


