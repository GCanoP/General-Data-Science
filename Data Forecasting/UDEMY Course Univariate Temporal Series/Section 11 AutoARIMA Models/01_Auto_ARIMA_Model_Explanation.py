"""
AUTO - ARIMA Model Explantion Complete
author : Gerardo Cano Perea 
date : January 1, 2021
"""
# Importing Packages. 
import scipy
import sklearn
import yfinance
import numpy as np
import pandas as pd
import seaborn as sns
from arch import arch_model
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima_model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# Loading and Pre-processing the Datasets. 
raw_data = yfinance.download(tickers = '^GSPC ^FTSE ^N225 ^GDAXI', start = '1994-01-07', end = '2020-03-20',
                             interval = '1d', group_by = 'ticker', auto_adjust = True, treads = True)
df_comp = raw_data.copy()
df_comp['spx'] = df_comp['^GSPC'].Close[:]
df_comp['dax'] = df_comp['^GDAXI'].Close[:]
df_comp['ftse'] = df_comp['^FTSE'].Close[:]
df_comp['nikkei'] = df_comp['^N225'].Close[:]
df_comp = df_comp.loc[:, ['spx','dax','nikkei','ftse']]
df_comp = df_comp.asfreq('b')
df_comp = df_comp.fillna(method = 'ffill')

# Creating Returns. 
df_comp['ret_spx'] = df_comp.spx.pct_change(1) * 100
df_comp['ret_ftse'] = df_comp.ftse.pct_change(1) * 100
df_comp['ret_dax'] = df_comp.dax.pct_change(1) * 100
df_comp['ret_nikkei'] = df_comp.nikkei.pct_change(1) * 100

# Split the Dataset into Train/Test. 
size = int(len(df_comp) * 0.8)
[df_train, df_test] = [df_comp.iloc[:size], df_comp.iloc[size:]]

# Fitting the Model. 
# The model selected is an SARIMAX (3,0,5) / There is no seasonal order ARIMAX (3,0,5)
# There is no X exogenous data ARIMA (3,0,5) / There is no integrative ARMA (3,5)
model_auto = auto_arima(df_train.ret_ftse[1:])
model_auto.summary()

# Re-fitting the Model. 
model_auto_2 = auto_arima(df_comp.ret_ftse[1:],
                          exogenous = df_comp[['ret_spx','ret_dax','ret_nikkei']][1:],
                          m = 5, max_order = None, max_p = 7, max_q = 7, max_d = 2,
                          max_P = 4, max_Q = 4, max_D = 2, maxiter = 50, alpha = 0.05,
                          n_jobs = -1, trend = 'ct', information_criterion = 'oob',
                          out_of_sample_size = int(len(df_comp) * 0.2))
# [exogenous] -->  outside factors (e.g another time series).
# [m] -->  seasonal cycle length.
# [max_order]  -->  maximum amount of variables to be used in the regression (p+q)
# [max_p]  -->  maximum AR component.
# [max_d]  -->  maximum Integrations.
# [max_q]  -->  maximum MA component.
# [max_iter]  -->  maximum iterations for the coefficient convergence in the model.
# [alpha]  -->  level of significance, the default is 5%. 
# [n_jobs]  -->  Number of models to fit in a step (-1 indicates 'as many as possible')
# [trend]  -->  'ct' as an usual value. 
# [information_criteria]  -->  'aic','aicc','bic','hqic','oob' (Out of Bags)
# [out_of_samples]  -->  To validate the model samples 20%
# [trend] can be a boolean for example tren = 1001 --> 1 term cte and 1 term cubic. 
