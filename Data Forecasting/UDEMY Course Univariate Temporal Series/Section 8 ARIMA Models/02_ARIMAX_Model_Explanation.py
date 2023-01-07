"""
ARMAX & ARIMAX MODEL EXPLANATION
author: Gerardo Cano Perea 
date : December 28, 2020
___
The models X include the exogenous variables for a more reliable prediction of 
the endougenous variable in the time series. 
The variable Y is the exogenous variable / vector of parameters. 
___
ARMAX Mathematical Expression. 
x_t = c + Beta * Y + phi_1 * x_t-1 + Phi_1 * e_t-1 + e_t
ARIMAX Mathematical Expression. 
delta(x)_t = c + Beta * Y + phi_1 * delta(x)_t-1 + Phi_1 * e_t-1 + e_t
___
QUESTION : What will happen if the exogenous variable in a ARIMAX model is the 
residuals components from another ARIMAX model with a lower order. The model could
get a better performance, or a worse? 
___
SIMPLE_IMPUTER = Applied to complet np.nan values. 
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

# Creating the Market Value Variable
df_comp['market_value'] = df_comp.ftse

# Warnings Treatment
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

"""
IMPLEMENTING AN ARIMAX EXOGENOUS MODEL. 
In this case the endogenous variable is FTSE, meanwhile the exogenous is SPX.
"""
# ARIMAX (1,1,1) X = spx
# The p-value for [spx] as exogenous variable is 0.652, this could suggest that the
# application of ARIMAX with this variable is not relevant at all. 
model_ar_1_i_1_ma_1_X_spx = ARIMA(df.market_value, exog = df.spx, order = (1,1,1))
results_ar_1_i_1_ma_1_X_spx = model_ar_1_i_1_ma_1_X_spx.fit()
results_ar_1_i_1_ma_1_X_spx.summary()

# Analyzing the Residuals. 
df['res_ar_1_i_1_ma_1_X_spx'] = results_ar_1_i_1_ma_1_X_spx.resid

# Plotting ACF for Residuals ARIMAX(1,1,1,spx)
sgt.plot_acf(df.res_ar_1_i_1_ma_1_X_spx[1:])
plt.title('ACF for Residuals ARIMAX (1,1,1,spx)')
plt.grid(True)
plt.show()

# Evaluating the fitting model. 
start_train = 1
end_train = len(df)-1
predictions_ar_1_i_1_ma_spx_train = results_ar_1_i_1_ma_1_X_spx.predict(start = start_train, end = end_train, exog = df.spx, typ = 'levels').rename('pred_ar_1_i_1_ma_1_X_spx')
predictions_ar_1_i_1_ma_spx_train = pd.DataFrame(predictions_ar_1_i_1_ma_spx_train)

# Evaluattin the model with test data. 
start_test = len(df) 
end_test = (len(df) + len(df_test)) - 1
predictions_ar_1_i_1_ma_spx_test = results_ar_1_i_1_ma_1_X_spx.predict(start = start_test, end = end_test, exog = df_test.spx, typ = 'levels').rename('pred_ar_1_i_1_ma_1_x_spx')
predictions_ar_1_i_1_ma_spx_test = pd.DataFrame(predictions_ar_1_i_1_ma_spx_test)

# Potting Predictions. 
df['market_value'].plot()
df_test['market_value'].plot()
predictions_ar_1_i_1_ma_spx_train['pred_ar_1_i_1_ma_1_X_spx'].plot()
predictions_ar_1_i_1_ma_spx_test['pred_ar_1_i_1_ma_1_x_spx'].plot()
plt.grid(True)
plt.show()
"""
EVALUATION AN ARIMAX MODEL WITH A MATRIX AS EXOGENOUS VARIABLE
The exogenous matrix will be the complete stock prices. 
"""
# Creating a matrix for exogenous parameter 
df_exog = df[df.columns.difference(['ftse', 'market_value','res_ar_1_i_1_ma_1_X_spx'])] 

# Creating the new ARIMAX(1,1,1,all)
model_ar_1_i_1_ma_1_X_all = ARIMA(df.market_value, exog = df_exog, order = (1,1,1))
results_ar_1_i_1_ma_1_X_all = model_ar_1_i_1_ma_1_X_all.fit()
results_ar_1_i_1_ma_1_X_all.summary()

"""
NEW MODEL WITH EXOGENOUS RESIDUALS.
"""
model_ar_1_i_1_ma_1_X_resid = ARIMA(df.market_value[1:], exog = df.res_ar_1_i_1_ma_1_X_spx[1:], order = (1,1,1))
results_ar_1_i_1_ma_1_X_resid = model_ar_1_i_1_ma_1_X_resid.fit()
results_ar_1_i_1_ma_1_X_resid.summary()

df['res_ar_1_i_1_ma_1_X_resid'] = results_ar_1_i_1_ma_1_X_resid.resid
sgt.plot_acf(df.res_ar_1_i_1_ma_1_X_resid[2:], zero = False, lags = 40)
plt.title('ACF factor for Residuals ARIMAX(1,1,1,resid)')
plt.grid(True)
plt.show() 

# Plotting Predictions. 
df['market_value'].plot()
