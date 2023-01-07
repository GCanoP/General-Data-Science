"""
AUTOREGRESIVE METHODS FOR FORECASTING
author: Gerardo Cano Perea  
date : December  18, 2020
AUTOREGRESSIVE MODELS ARE NOT FITTED FOR NON-SATIONARY DATA
"""

# Importing Packages 
# Warning : statsmodels.tsa.arima_model.ARMA and statsmodels.tsa.arima_model.ARIMA have
# been deprecated in favor of statsmodels.tsa.arima.model.ARIMA (note the between arima and 
# model) and statsmodels.tsa.SARIMAX. These will be removed after the 0.12 release.
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt
from statsmodels.tsa.arima_model import ARMA
from scipy.stats.distributions import chi2
import statsmodels.tsa.stattools as sts 
import seaborn as sns

# Importing Data and Pre-processing  
raw_csv_data = pd.read_csv('Index2018.csv')
df_comp = raw_csv_data.copy()
df_comp.date = pd.to_datetime(df_comp.date, dayfirst = True)
df_comp.set_index('date', inplace = True)
df_comp = df_comp.asfreq('b')
df_comp = df_comp.fillna(method = 'ffill')

# Now the data evaluation will be for ftse index
df_comp['market_value'] = df_comp.ftse 
del df_comp['spx']
del df_comp['dax']
del df_comp['ftse']
del df_comp['nikkei']

# Split the dataset in test/train
size = int(len(df_comp)*0.8)
[df, df_test] = [df_comp.iloc[:size], df_comp.iloc[size:]]

# Dickey - Fuller Test [Null] serie is not stationary [Alternative] serie is stationary
sts.adfuller(df.market_value) # Serie is not stationary with a p-value : 0.3301089

"""
USING RETURNS - CHANGE IN PRICES
"""

# pct_change(1) : percentage of change for 1 previous day 
df['returns'] = df.market_value.pct_change(1).mul(100)
df = df.iloc[1:]
# Applying Dickey - Fuller in returns values.
sts.adfuller(df.returns) # Now the serie is stationary with a p-value 7.79e-24


# Plotting ACF and PACF
sgt.plot_acf(df.returns, zero = False, lags = 40)
plt.title('ACF for Return Prices', size = 14)
plt.grid(True)
plt.show()
sgt.plot_pacf(df.returns, zero = False, lags = 40)
plt.title('PACF for Returns Prices', size = 14)
plt.grid(True)
plt.show()

"""
AUTO-REGRESIVE MODELS N-DIMENSIONAL
"""
# 1st Order Model
model_ar = ARMA(df.returns, order = (1,0))
results_ar = model_ar.fit()
results_ar.summary()

model_ar_2 = ARMA(df.returns, order = (2,0))
results_ar_2 = model_ar_2.fit()
results_ar_2.summary()

model_ar_3 = ARMA(df.returns, order = (3,0))
results_ar_3 = model_ar_3.fit()
results_ar_3.summary()

model_ar_4 = ARMA(df.returns, order = (4,0))
results_ar_4 = model_ar_4.fit()
results_ar_4.summary()

"""
LOG_LIKELIHOOD RATIO TEST
"""

def LLR_test (mod_1, mod_2, DF = 1):
    L1 = mod_1.llf
    L2 = mod_2.llf
    LR = (2*(L2 - L1))
    p = chi2.sf(LR, DF).round(3)
    return p

LLR_test(mod_1 = results_ar_2, mod_2 = results_ar_3) # 0.001 < 0.05 Thus model are different.
LLR_test(mod_1 = results_ar_3, mod_2 = results_ar_4) # 0.001 < 0.05 Thus model are different.

model_ar_5 = ARMA(df.returns, order = (5,0))
results_ar_5 = model_ar_5.fit()
print(results_ar_5.summary())
print("\n LLR test p-value = " + str(LLR_test(results_ar_4, results_ar_5)))

model_ar_6 = ARMA(df.returns, order = (6,0))
results_ar_6 = model_ar_6.fit()
print(results_ar_6.summary())
print("\n LLR test p-value = " + str(LLR_test(results_ar_5, results_ar_6)))

model_ar_7 = ARMA(df.returns, order = (7,0))
results_ar_7 = model_ar_7.fit()
print(results_ar_7.summary())
print("\n LLR test p-value = " + str(LLR_test(results_ar_6, results_ar_7)))

# CONCLUSION. Before 7 timesteps, the prediction is not more accurate. The ideal lag is 6 timesteps. 

print("\n LLR test p-value = " + str(LLR_test(results_ar, results_ar_6, DF = 5)))

# Applying Dickey - Fuller test to check stationarity properties. 
df['res_price'] = results_ar_6.resid
sts.adfuller(df.res_price) # The serie is stationary

# Applying Auto-Correlation Factor to Residuals. 
sgt.plot_acf(df.res_price, zero = False, lags = 40)
plt.title('ACF for Residual Prices', size = 14)
plt.grid(True)
plt.show()

df.res_price[1:].plot(figsize = (15,10))
plt.title('Residuals of Price', size = 14)
plt.grid(True)
plt.show()





