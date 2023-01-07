"""
AUTO ARIMA MODEL FOR PASSENGERS DATASET. 
author : Gerardo Cano Perea. 
date : December 27, 2020.
"""
#Importing Packages. 
import itertools
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as sts
import statsmodels.graphics.tsaplots as sgt
from statsmodels.tsa.arima_model import ARIMA
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

"""
DEFINING AN AUTO-ARIMA MODEL / STATISTICS BASIC MODEL SELECTION
The model is saved in a decision matriz with the LLF and AIC discriminators. 
The model filtering is based on AIC / Akaike Information Criteria. 
The model could be implemented with a BIC / Bayesian Information Criteria. 
"""

# Defining the [p,d,q] ranges and combinations. 
p = d = q = range(1,5)
pdq = list(itertools.product(p,d,q))

# Generating the method. 

desicion = []
for param in pdq:
    try:
        mod = ARIMA(df_train, order = param)
        results = mod.fit()
        parameters = str('ARIMA:' + str(param) + 'LLF:' + str(results.llf) + 'AIC:' + str(results.aic))
        desicion.append(parameters)
    except:
        continue

# LLF must be maximized / AIC must be minimized 

# ARIMA(1,1,1) Model
model_ar_1_i_1_am_1 = ARIMA(df_train['Thousands of Passengers'], order = (1,1,1))
results_ar_1_i_1_am_1 = model_ar_1_i_1_am_1.fit()
results_ar_1_i_1_am_1.summary()

# ARIMA (1,1,1) Residual Verification. 
df_train['res_ar_1_i_1_ma_1'] = results_ar_1_i_1_am_1.resid

# ARIMA (1,1,1) Residual ACF Verification. 
sgt.plot_acf(df_train.res_ar_1_i_1_ma_1[1:], zero = False, lags = 40)
plt.title('ACF fo Residuals ARIMA(1,1,1) for Thousands of Passengers Variable', size = 18)
plt.grid(True)
plt.show()

# ARIMA(1,1,2)
model_ar_1_i_1_am_2 = ARIMA(df_train['Thousands of Passengers'], order = (1,1,2))
results_ar_1_i_1_am_2 = model_ar_1_i_1_am_2.fit()
# ARIMA(1,1,3)
model_ar_1_i_1_am_3 = ARIMA(df_train['Thousands of Passengers'], order = (1,1,3))
results_ar_1_i_1_am_3 = model_ar_1_i_1_am_3.fit()
# ARIMA (2,1,1)
model_ar_2_i_1_am_1 = ARIMA(df_train['Thousands of Passengers'], order = (2,1,1))
results_ar_2_i_1_am_1 = model_ar_2_i_1_am_1.fit()
# ARIMA (3,1,1)
model_ar_3_i_1_am_1 = ARIMA(df_train['Thousands of Passengers'], order = (3,1,1))
results_ar_3_i_1_am_1 = model_ar_3_i_1_am_1.fit()
# ARIMA (3,1,2)
model_ar_3_i_1_am_3 = ARIMA(df_train['Thousands of Passengers'], order = (3,1,3))
results_ar_3_i_1_am_3 = model_ar_3_i_1_am_3.fit()

print('\n\nARIMA MODEL COMPARISON\n')
print('ARIMA(1,1,1)   LLR :',round(results_ar_1_i_1_am_1.llf, 6),'AIC :',round(results_ar_1_i_1_am_1.aic, 6))
print('ARIMA(1,1,2)   LLR :',round(results_ar_1_i_1_am_2.llf, 6),'AIC :',round(results_ar_1_i_1_am_2.aic, 6))
print('ARIMA(1,1,3)   LLR :',round(results_ar_1_i_1_am_3.llf, 6),'AIC :',round(results_ar_1_i_1_am_3.aic, 6))
print('ARIMA(2,1,1)   LLR :',round(results_ar_2_i_1_am_1.llf, 6),'AIC :',round(results_ar_2_i_1_am_1.aic, 6))
print('ARIMA(3,1,1)   LLR :',round(results_ar_3_i_1_am_1.llf, 6),'AIC :',round(results_ar_3_i_1_am_1.aic, 6))
print('ARIMA(3,1,2)   LLR :',round(results_ar_3_i_1_am_2.llf, 6),'AIC :',round(results_ar_3_i_1_am_2.aic, 6))

# ARIMA (3,1,3) Residual Verification. 
df_train['res_ar_3_i_1_ma_3'] = results_ar_3_i_1_am_3.resid

# ARIMA (3,1,2) Residual ACF Verification. 
sgt.plot_acf(df_train.res_ar_3_i_1_ma_3[1:], zero = False, lags = 40)
plt.title('ACF fo Residuals ARIMA(3,1,3) for Thousands of Passengers Variable', size = 18)
plt.grid(True)
plt.show()

# Predictions for train data. 
start_train = 1
end_train = len(df_train)
predictions_ar_3_i_1_ma_3_train = results_ar_3_i_1_am_3.predict(start = start_train, end = end_train, typ = 'levels')
# Predictions for test data. 
# start_test = len(df_train)
# end_test = (len(df_train) + len(df_test)) - 1
start_test = datetime(1958, 8, 1)
end_test = datetime(1960, 11, 1)
predictions_ar_3_i_1_ma_3_test = results_ar_3_i_1_am_3.predict(start = start_test, end = end_test, dynamic = False, typ = 'levels')

# Plotting Predicitions Train/Test. 
df_train['Thousands of Passengers'].plot()
predictions_ar_3_i_1_ma_3_train.plot()
df_test['Thousands of Passengers'].plot()
predictions_ar_3_i_1_ma_3_test.plot()
plt.grid(True)
plt.show()

# CONCLUSION: The Best Model is ARIMA (3,1,3)
