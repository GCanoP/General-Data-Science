"""
SEASONAL ARIMAX EXOGENOUS MODEL EXPLANATION
author: Gerardo Cano Perea.
date: December 30, 2020
___
Nomeclature : SARIMAX (p,d,q)(P,D,Q,s)
(p,d,q) : Non Seasonal Parameters
(P,D,Q,s) : Seasonal Parameters ; (s) : Stational Parameter
The parameter (s) is defined as the frequency for a trend repetition.
___
Example for SARIMA(1,0,2)(2,0,1,5)
Added values from a past time : x_t-5 / x_t-10 (2 SAR: 5 Periods)
Added errors from a past time : e_t-5 (1 MAR : 5 Periods)
"""

# Importing Relevant Packages and Libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
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

"""
CONFIGURING A SIMPLE SARIMAX MODEL.
"""
# Model ARIMAX(1,0,1)(2,0,1,5)
model_sarimax_101_2015 = SARIMAX(df.market_value, exog = df.spx, order = (1,0,1), seasonal_order = (2,0,1,5))
results_sarimax_101_2015 = model_sarimax_101_2015.fit()
results_sarimax_101_2015.summary()
