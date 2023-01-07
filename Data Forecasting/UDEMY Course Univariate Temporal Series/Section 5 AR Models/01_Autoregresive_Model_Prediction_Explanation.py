"""
AUTOREGRESIVE METHODS FOR FORECASTING
author: Gerardo Cano Perea  
date : December  18, 2020

Step 1 : Choose the model.   Step 2 : Split data in train/test.
Step 3 : Fit the model with train data.   Step 4 : Evaluate in test data.
Step 5 : Fit the model again.   Step 6 : Make a forecast.  

Selective Criterions for Avoid Overfitting.
AIC : Akaike Information Criteria
BIC : Bayesian Information Criteria
Residual most be compared with White Noise.
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

# Split the dataset
size = int(len(df_comp)*0.8)
[df, df_test] = [df_comp.iloc[:size], df_comp.iloc[size:]]

# Checking the AUTO CORRELATION FACTOR.
# Default alpha = 0.05
sgt.plot_acf(df.market_value, zero = False, lags = 40)
plt.title('ACF for FTSE Prices', size = 14)
plt.grid(True)
plt.show()

# Cheking the PARCIAL AUTO CORRELATION FACTOR
# Default alpha = 0.05
sgt.plot_pacf(df.market_value, zero = False, lags = 40)
plt.title('PACF for FTSE Prices', size = 14)
plt.grid(True)
plt.show()

"""
THE AUTOREGRESIVE METHOD 1ST ORDER.
# Null Hypothesis : [z] is significative equal to zero
# Alternative Hypothesis : [z] is significative less than zero
"""

# ar.L1.market_value for Lag = 1
# [z] Parameter of statistic relevance
# [Confident Range] if zero is in this range the null hypothesis is not rejected thus
# the coefficient is significatively equal to zero. 
# The p-value must not be higher than alpha = 0.05
model_ar = ARMA(df.market_value, order = (1,0))
results_ar = model_ar.fit()
results_ar.summary()

"""
THE HIGHER-LAG AR MODELS
The AR for higher orders
"""
model_ar_2 = ARMA(df.market_value, order = (2,0))
results_ar_2 = model_ar_2.fit()
results_ar_2.summary()

model_ar_3 = ARMA(df.market_value, order = (3,0))
results_ar_3 = model_ar_3.fit()
results_ar_3.summary()

model_ar_4 = ARMA(df.market_value, order = (4,0))
results_ar_4 = model_ar_4.fit()
results_ar_4.summary()

"""
LOG LIKELIHOOD RATIO TEST
Check the similitud between two predictive models.
"""

# Null Hypothesis : Two models are significatively the same 
# Alternative Hypothesis : Two models are significatively different
# Degrees of Freedoom between two models. 
def LLR_test (mod_1, mod_2, DF = 1):
    L1 = mod_1.llf
    L2 = mod_2.llf
    LR = (2*(L2 - L1))
    p = chi2.sf(LR, DF).round(3)
    return p

#Comparing Higher - Lag AR Models. 
# Between results_ar_2 and results_ar_3 there are 1 DF
# Meanwhile, between results_ar_2 and results_ar_4 there are 2 DF 

LLR_test(mod_1 = results_ar_2, mod_2 = results_ar_3) # 0.001 < 0.05 Thus model are different.
LLR_test(mod_1 = results_ar_3, mod_2 = results_ar_4) # 0.001 < 0.05 Thus model are different.

model_ar_5 = ARMA(df.market_value, order = (5,0))
results_ar_5 = model_ar_5.fit()
print(results_ar_5.summary())
print("\n LLR test p-value = " + str(LLR_test(results_ar_4, results_ar_5)))

model_ar_6 = ARMA(df.market_value, order = (6,0))
results_ar_6 = model_ar_6.fit()
print(results_ar_6.summary())
print("\n LLR test p-value = " + str(LLR_test(results_ar_5, results_ar_6)))

model_ar_7 = ARMA(df.market_value, order = (7,0))
results_ar_7 = model_ar_7.fit()
print(results_ar_7.summary())
print("\n LLR test p-value = " + str(LLR_test(results_ar_6, results_ar_7)))

# The p-valie is 0.571 it is not significative.
# The compared p-value is not significative
model_ar_8 = ARMA(df.market_value, order = (8,0))
results_ar_8 = model_ar_8.fit()
print(results_ar_8.summary())
print("\n LLR test p-value = " + str(LLR_test(results_ar_7, results_ar_8)))

# CONCLUSION. Before 8 timesteps, the prediction is not more accurate. The ideal lag is 7 timesteps. 

# Comparing the simplest model with the 7yh order ARM
print("\n LLR test p-value = " + str(LLR_test(results_ar, results_ar_7, DF = 6)))

"""
ANALYSING THE RESIDUALS
Null: The serie is not stationary. To acept : p-value > alpha = 0.05
Alternative : The serie is stationary 
"""
# Applying Dickey - Fuller test to check stationarity properties. 
df['res_price'] = results_ar_7.resid
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











