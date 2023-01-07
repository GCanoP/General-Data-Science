"""
=============================================================================
AUTO REGRESSIVE INTEGRATED MOVING AVERAGE ARIMA
Modelo autorregresivo integrado de medias moviles.
author : Gerardo Cano Perea  
date : December 23, 2020 
=============================================================================
----
Integrated Part : Explain the number of non seasonal differences that we must examinate to
establish stationarity
----
ARIMA(p,d,q)  p->AR q->MA d->Integration Order
ARIMA(p,0,q) -> ARMA(p,q)
ARIMA(0,0,q) -> MA(q)
ARIMA(p,0,0) -> AR(p)
----
Integration Order (1) -> Differences between prices. 
Integration Order (2) -> Differences between the differences of prices.
----
Basic ARIMA Mathematical Formulation. 
ARIMA(1,1,1) = delta(x)_t = c + phi_1 * delta(x)_t-1 + Phi_1 * e_t-1 + e_t
----
ARIMA does not have Auto-Correlation Function neither Partial Auto-Correlation Function.
Steps to Complete the Model.
Step 1. Modeling the basic ARIMA(1,1,1).
Step 2. Check the residuals.
Step 3. Propose a more complex model. 
Step 4. Correct the number of components. 
Note : For every integration, one observation is lost. Be careful with large-order models.  
"""

# Importing Relevant Packages and Libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats.distributions import chi2
from math import sqrt
import seaborn as sns


# Importing and Pre-processing Dataset. 
raw_csv_data = pd.read_csv('Index2018.csv')
df_comp = raw_csv_data.copy()
df_comp.date = pd.to_datetime(df_comp.date, dayfirst = True)
df_comp.set_index('date', inplace = True)
df_comp = df_comp.asfreq('b')
df_comp = df_comp.fillna(method = 'ffill')

# Creating Market Value
df_comp['market_value'] = df_comp.ftse

# Warnings
import warnings
warnings.filterwarnings('ignore')

# Simplifying the dataset. 
# del df_comp['spx']
# del df_comp['ftse']
# del df_comp['dax']
# del df_comp['nikkei']
size = int(len(df_comp)* 0.8)
[df, df_test] = [df_comp.iloc[:size], df_comp.iloc[size:]]

# Defining the LLR_test 
def LLR_test(mod_1, mod_2, DF = 1):
    L1 = mod_1.llf
    L2 = mod_2.llf
    LR = (2*(L2 - L1))
    p = chi2.sf(LR, DF).round(4)
    return p 

# Creating Returns
df['returns'] = df.market_value.pct_change(1)*100

"""
=============================================================================
MODELING A SIMPLE ARIMA(1,1,1).
=============================================================================
"""

# ARIMA(1,1,1) Model
model_ar_1_i_1_am_1 = ARIMA(df.market_value, order = (1,1,1))
results_ar_1_i_1_am_1 = model_ar_1_i_1_am_1.fit()
results_ar_1_i_1_am_1.summary()

# ARIMA(1,1,1) Model Residuals
df['res_ar_1_i_1_ma_1'] = results_ar_1_i_1_am_1.resid

# Auto-Correlation Factor for Residual. 
# Be careful with the NAN values at the beggining of the dataset. 
sgt.plot_acf(df.res_ar_1_i_1_ma_1[1:], zero = False, lags = 40)
plt.title('ACF fo Residuals ARIMA(1,1,1) for market_value', size = 18)
plt.grid(True)
plt.show()

"""
HIGHER-LAG ARIMA MODELS.
The combinations (p,d,q) were filtered previously.
Basic Model ARIMA(p,1,q) with only one integration. 
"""

# ARIMA(1,1,2)
model_ar_1_i_1_am_2 = ARIMA(df.market_value, order = (1,1,2))
results_ar_1_i_1_am_2 = model_ar_1_i_1_am_2.fit()
# ARIMA(1,1,3)
model_ar_1_i_1_am_3 = ARIMA(df.market_value, order = (1,1,3))
results_ar_1_i_1_am_3 = model_ar_1_i_1_am_3.fit()
# ARIMA (2,1,1)
model_ar_2_i_1_am_1 = ARIMA(df.market_value, order = (2,1,1))
results_ar_2_i_1_am_1 = model_ar_2_i_1_am_1.fit()
# ARIMA (3,1,1)
model_ar_3_i_1_am_1 = ARIMA(df.market_value, order = (3,1,1))
results_ar_3_i_1_am_1 = model_ar_3_i_1_am_1.fit()
# ARIMA (3,1,2)
model_ar_3_i_1_am_2 = ARIMA(df.market_value, order = (3,1,2))
results_ar_3_i_1_am_2 = model_ar_3_i_1_am_2.fit(start_ar_lags = 5)

# Printing results 
# ARIMA(1,1,3) shows the highest LLR and the lowest AIC.
print('\n\nARIMA MODEL COMPARISON\n')
print('ARIMA(1,1,1)   LLR :',round(results_ar_1_i_1_am_1.llf, 6),'AIC :',round(results_ar_1_i_1_am_1.aic, 6))
print('ARIMA(1,1,2)   LLR :',round(results_ar_1_i_1_am_2.llf, 6),'AIC :',round(results_ar_1_i_1_am_2.aic, 6))
print('ARIMA(1,1,3)   LLR :',round(results_ar_1_i_1_am_3.llf, 6),'AIC :',round(results_ar_1_i_1_am_3.aic, 6))
print('ARIMA(2,1,1)   LLR :',round(results_ar_2_i_1_am_1.llf, 6),'AIC :',round(results_ar_2_i_1_am_1.aic, 6))
print('ARIMA(3,1,1)   LLR :',round(results_ar_3_i_1_am_1.llf, 6),'AIC :',round(results_ar_3_i_1_am_1.aic, 6))
print('ARIMA(3,1,2)   LLR :',round(results_ar_3_i_1_am_2.llf, 6),'AIC :',round(results_ar_3_i_1_am_2.aic, 6))

# Computing the LLR between nested models. 
# The model ARIMA(1,1,3) is significatively better.
LLR_test(results_ar_1_i_1_am_2, results_ar_1_i_1_am_3)
LLR_test(results_ar_1_i_1_am_1, results_ar_1_i_1_am_3)

# Analyzing the residuals for ARIMA(1,1,3)
df['res_ar_1_i_1_ma_3'] = results_ar_1_i_1_am_3.resid
sgt.plot_acf(df.res_ar_1_i_1_ma_3[1:], zero = False, lags = 40)
plt.title('ACF for Residuals Model ARIMA(1,1,3)', size = 18)
plt.grid(True)
plt.show()

# We must compared a high order levels to define the best model. 
model_ar_5_i_1_am_1 = ARIMA(df.market_value, order = (5,1,1))
results_ar_5_i_1_am_1 = model_ar_5_i_1_am_1.fit()
results_ar_5_i_1_am_1.summary()

# Comparing the Loglikelihood Values
# ARIMA (5,1,1) is the best model. 
print('ARIMA(1,1,3)   LLR :',round(results_ar_1_i_1_am_3.llf, 6),'AIC :',round(results_ar_1_i_1_am_3.aic, 6))
print('ARIMA(5,1,1)   LLR :',round(results_ar_5_i_1_am_1.llf, 6),'AIC :',round(results_ar_5_i_1_am_1.aic, 6))

# Analyzing the residuals for ARIMA(5,1,1)
df['res_ar_5_i_1_ma_1'] = results_ar_5_i_1_am_1.resid
sgt.plot_acf(df.res_ar_5_i_1_ma_1[1:], zero = False, lags = 40)
plt.title('ACF for Residuals Model ARIMA(5,1,1)', size = 18)
plt.grid(True)
plt.show()

"""
MODELS WITH A HIGH - ORDER LEVEL OF INTEGRATION.
The model ARIMAX in this case only needs a first integration cause the delta_price
integration is a sationary serie. The model do not need a high order integration.  
"""

# Creating a new variable that integrates manually the market_value series.  
df['delta_prices'] = df.market_value.diff(1) # The serie is stationary. 

# The model ARIMA(1,1,1) is equivalent to ARMA(1,1) if we pre-integrate the values to
# get a stationary serie instead a non-stationary serie in the case of market-value variable. 

# Model ARIMA(1,1,1) -> ARMA(1,1) with pre-integrated values. 
model_delta_ar_1_i_1_ma_1 = ARIMA(df.delta_prices[1:], order = (1,0,1))
results_delta_ar_1_i_1_ma_1 = model_delta_ar_1_i_1_ma_1.fit()
results_delta_ar_1_i_1_ma_1.summary()

# Modelo ARIMA(1,1,1) with no pre-integrated values.
model_ar_1_i_1_am_1 = ARIMA(df.market_value, order = (1,1,1))
results_ar_1_i_1_am_1 = model_ar_1_i_1_am_1.fit()
results_ar_1_i_1_am_1.summary()

sts.adfuller(df.delta_prices[1:]) # delta_prices is a stationary serie with a p-value = 0.0 

"""
ARIMA model are more expensive that ARMA models for Stationary Time Series. 
For this reason, if the serie is stationary, an ARMA model is better as predictor.
"""

# Evaluating fitting train for ARIMA (5,1,1)
start = 1
end = len(df.market_value)
predictions_ar_5_i_1_am_1_train = results_ar_5_i_1_am_1.predict(start = start, end = end,typ = 'levels').rename('ARIMA(5,1,1) Train Predictions')
# Evaluating fitting test for ARIMA (5,1,1)
start_1 = len(df.market_value)
end_1 = (len(df.market_value) + len(df_test))-1
predictions_ar_5_i_1_am_1_test = results_ar_5_i_1_am_1.predict(start = start_1, end = end_1, dynamic = True, typ = 'levels').rename('ARIMA(5,1,1) Test Predictions')
# Forecasting
forecast_ar_5_i_1_am_1_test = results_ar_5_i_1_am_1.forecast(steps = (end_1 - start_1))
# Plotting results 
# df['market_value'].plot(legend =True)
df_test['market_value'].plot(legend = True)
# predictions_ar_5_i_1_am_1_train.plot(legend = True)
predictions_ar_5_i_1_am_1_test.plot(legend = True)
plt.grid(True)
plt.show()

results_ar_5_i_1_am_1.plot_predict(start = start_1, end = end_1)
forecast = forecast_ar_5_i_1_am_1_test[0]
