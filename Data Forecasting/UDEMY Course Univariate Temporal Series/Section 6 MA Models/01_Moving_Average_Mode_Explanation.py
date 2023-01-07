"""
MOVING AVERAGE MODEL FOR TIME SERIES
INDEX_2018 EXAMPLE
author : Gerardo Cano Perea  
date : December 20, 2020
AR = x_t = c + phi_1 * X_t-1 + e_t     Use of the previous value x_t-1
MA = x_t = c + phi_1 * e_t-1 + e_t     Use of the previous error e_t-1
MA(Firsth_Order) = AR(Infinit Order)
MA(Infinit Order) = AR(Firsth Order)
Predictive Models
"""

# Importing packages. 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts 
from statsmodels.tsa.arima_model import ARMA
from scipy.stats.distributions import chi2
from math import sqrt
import seaborn as sns 

# Loading the dataset. 
raw_csv_data = pd.read_csv('Index2018.csv')
df_comp = raw_csv_data.copy()
df_comp.date = pd.to_datetime(df_comp.date, dayfirst = True)
df_comp.set_index('date', inplace = True)
df_comp = df_comp.asfreq('b')
df_comp = df_comp.fillna(method = 'ffill')

# Pre-processing the data  
df_comp['market_value'] = df_comp.ftse
del df_comp['spx']
del df_comp['ftse']
del df_comp['dax']
del df_comp['nikkei']
size = int(len(df_comp)*0.8)
[df, df_test] = [df_comp.iloc[:size], df_comp.iloc[size:]]

# Applying the LLR
# Model.llf compute the log-likelihood function.
def LLR_test(mod_1, mod_2, DF = 1):
    L1 = mod_1.llf
    L2 = mod_2.llf
    LR = (2*(L2 - L1))
    p = chi2.sf(LR, DF).round(3)
    return p

import warnings
warnings.filterwarnings('ignore')

# Creating Returns variable. 
df['returns'] = df.market_value.pct_change(1)*100 # Used to get a stationary serie. 

# Negative ACF / PACF = The prices of today increases/decreases in the opposite direction to yesterday's prices
# Possitive ACF / PACF = The prices of today increases/decreases in the same direction as yesterday's prices.  
# Auto - Correlation Factor. 
sgt.plot_acf(df.returns[1:], zero = False, lags = 40)
plt.title('ACF for Returns', size = 16)
plt.grid(True)
plt.show()

# Partial Auto - Correlation Factor. 
sgt.plot_pacf(df.returns[1:], zero = False, lags = 40, method = 'ols')
plt.title('PACF for Returns', size = 16)
plt.grid(True)
plt.show()

"""
MOVING AVERGA MODEL - FIRST ORDER
Building the First Order MA
"""

# MA - Returns model.
# AR(1)MA(0) is defined by order = (1,0)
# AR(0)MA(1) is defined by order = (0,1)
model_ret_ma_1 = ARMA(df.returns[1:], order = (0,1))
results_ret_ma_1 = model_ret_ma_1.fit()
results_ret_ma_1.summary()

# Higher-Lag MA Models for Returns
# LLR function is also applied in this model.
model_ret_ma_2 = ARMA(df.returns[1:], order = (0,2))
results_ret_ma_2 = model_ret_ma_2.fit()
print(results_ret_ma_2.summary())
print('LLR test p-value = ', str(LLR_test(results_ret_ma_1, results_ret_ma_2)))

model_ret_ma_3 = ARMA(df.returns[1:], order = (0,3))
results_ret_ma_3 = model_ret_ma_3.fit()
print(results_ret_ma_3.summary())
print('LLR test p-value = ', str(LLR_test(results_ret_ma_2, results_ret_ma_3)))

model_ret_ma_4 = ARMA(df.returns[1:], order = (0,4))
results_ret_ma_4 = model_ret_ma_4.fit()
print(results_ret_ma_4.summary())
print('LLR test p-value = ', str(LLR_test(results_ret_ma_3, results_ret_ma_4)))

model_ret_ma_5 = ARMA(df.returns[1:], order = (0,5))
results_ret_ma_5 = model_ret_ma_5.fit()
print(results_ret_ma_5.summary())
print('LLR test p-value = ', str(LLR_test(results_ret_ma_4, results_ret_ma_5)))

model_ret_ma_6 = ARMA(df.returns[1:], order = (0,6))
results_ret_ma_6 = model_ret_ma_6.fit()
print(results_ret_ma_6.summary())
print('LLR test p-value = ', str(LLR_test(results_ret_ma_5, results_ret_ma_6)))

model_ret_ma_7 = ARMA(df.returns[1:], order = (0,7))
results_ret_ma_7 = model_ret_ma_7.fit()
print(results_ret_ma_7.summary())
print('LLR test p-value = ', str(LLR_test(results_ret_ma_6, results_ret_ma_7)))

model_ret_ma_8 = ARMA(df.returns[1:], order = (0,8))
results_ret_ma_8 = model_ret_ma_8.fit()
print(results_ret_ma_8.summary())
print('LLR test p-value = ', str(LLR_test(results_ret_ma_7, results_ret_ma_8)))

# CONCLUSIONS : The model MA(8) is slightly better than MA(6)
# Comparing MA(6) and MA(8)
LLR_test(results_ret_ma_6, results_ret_ma_8, DF = 2)

"""
ANALYSING RESIDUALS
"""

df['res_ret_ma_8'] = results_ret_ma_8.resid[1:]

# Computing the mean and covariance of the residuals
print('The mean of the residuals is', str(round(df.res_ret_ma_8.mean(),3)))
print('The variance of the residuals is', str(round(df.res_ret_ma_8.var(),3))) 

# Computing the Standar Deviation.
round(sqrt(df.res_ret_ma_8.var()), 3)

# Plotting the Residuals of Returns
df.res_ret_ma_8[1:].plot(figsize = (15,10))
plt.title('Residuals of Returns', size =16)
plt.grid(True)
plt.show()

# Applying the Dickey - Fuller test  
# Null Hypothesis : The time serie is not stationary.
sts.adfuller(df.res_ret_ma_8[2:]) # p-value : 0.0 the null hypothesis is rejected
sts.adfuller(df.market_value) # p-value : 0.3301 the null hypothesis is accepted

# Plotting the ACF for Residuals. 
sgt.plot_acf(df.res_ret_ma_8[2:], zero = False, lags = 40)
plt.title('ACF for Residual MA(8)', size = 16)
plt.grid(True)
plt.show()

"""
METHOD FOR A MARKET VALUE NORMALIZATION
"""
benchmark = df.market_value.iloc[0]
df['norm'] = df.market_value.div(benchmark).mul(100)

sts.adfuller(df.norm) # the serie is not stationary.

"""
METHOD FOR A RETURN NORMALIZATION
"""

bench_ret = df.returns.iloc[1]
df['norm_ret'] = df.returns.div(bench_ret).mul(100)

sts.adfuller(df.norm_ret[1:]) # The serie is stationary.

sgt.plot_acf(df.norm_ret[1:], zero = False, lags = 40)
plt.title('ACF for Normalized Returns', size = 16)
plt.grid(True)
plt.show()

# Apparently the normalization does not have a significative impact in coefficient estimations. 
# The principal advantages is to compare two differen temporal series. 
model_norm_ret_ma_8 = ARMA(df.norm_ret[1:], order = (0,8))
results_norm_ret_8 = model_norm_ret_ma_8.fit()
results_norm_ret_8.summary()

# Applying Dickey - Fuller test to residual normalization
df['res_norm_ret_ma_8'] = results_norm_ret_8.resid[1:]
sts.adfuller(df.res_norm_ret_ma_8[2:]) # The serie is not stationary

# Plotting Values 
df['res_norm_ret_ma_8'][1:].plot(figsize = (15,10))
plt.title('Residuals for Normalized Returns', size = 16)
plt.grid(True)
plt.show()

# Plotting Normalized Returns Residual
sgt.plot_acf(df.res_norm_ret_ma_8[2:], zero = False, lags = 40)
plt.title('ACF for Normzalized Residual Returns', size =18)
plt.grid(True)
plt.show()

"""
MOVING AVERAGE MODEL FOR STATIONARY TIME SEIES
The models usually do a good prediction in non stationary time series.
Remain: The Autoregressive models are usually not confident to predict non stationarity time series
"""

# Auto Correlation Factor for Market Values.  
sgt.plot_acf(df.market_value, zero = False, lags= 40)
plt.title('ACF for Market Values', size = 18)
plt.grid(True)
plt.show()

# Partial Auto Correlation Factor for Market Values.
sgt.plot_pacf(df.market_value, zero = False, lags = 40)
plt.title('PACF for Market Value', size = 18)
plt.grid(True)
plt.show()

model_ma_1 = ARMA(df.market_value, order = (0,1))
results_ma_1 = model_ma_1.fit()
results_ma_1.summary()


























    