"""
=============================================================================
GARCH MODEL EXPLANATION. 
Generalized Autoregressive Conditional Heteroscedascity Model. 
Modelos Autorregresivos con Heterocedasticidad Condicional Generalizada.
Author : Gerardo Cano Perea 
Date : December 27, 2020.
=============================================================================
Volatility ~ Variance Equations
Var(x_t|x_t-1) = Omega + alpha_1 * (e_t-1)**2 + beta_1*(sigma_t-1)**2
GARCH(1,1) --> ARCH(1) + GARCH(1) --> (e_t)**2 + (sigma_t)**2 --> Error + Variance.
More Complex Model. ARMA - GARCH / ARIMA - GARCH / SARIMA - GARCH
"""
# Import Packages. 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as sts  
import statsmodels.graphics.tsaplots as sgt  
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats.distributions import chi2
from arch import arch_model
from math import sqrt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Importing the datset. 
raw_csv_data = pd.read_csv('Index2018.csv')
df_comp = raw_csv_data.copy() 
df_comp.date = pd.to_datetime(df_comp.date , dayfirst = True)
df_comp.set_index('date', inplace = True)
df_comp = df_comp.asfreq('b')
df_comp = df_comp.fillna(method = 'ffill')
df_comp['market_value'] = df_comp.ftse
del df_comp['spx']
del df_comp['dax']
del df_comp['nikkei']
del df_comp['ftse']

# Split the dataset into train/test. 
size = int(len(df_comp)*0.8)
[df, df_test] = [df_comp.iloc[:size], df_comp.iloc[size:]]

# Creating Returns variable. 
df['returns'] = df.market_value.pct_change(1).mul(100)

"""
The Simple GARCH Model. 
Usually a Simple Garch (1,1) is the best model to measure the volatility of stock
prices returns. Apply a High Lag GARCH model is meaningless. 
"""

# GARCH(1,1)
model_garch_1_1 = arch_model(df.returns[1:], mean = 'Constant', vol = 'GARCH', p = 1, q = 1)
results_garch_1_1 = model_garch_1_1.fit(update_freq = 5)
results_garch_1_1.summary()

# GARCH(1,2)
model_garch_1_2 = arch_model(df.returns[1:], mean = 'Constant', vol = 'GARCH', p = 1, q = 2)
results_garch_1_2 = model_garch_1_2.fit(update_freq = 5)
results_garch_1_2.summary()

# GARCH(2,1)
model_garch_2_1 = arch_model(df.returns[1:], mean = 'Constant', vol = 'GARCH', p = 2, q = 1)
results_garch_2_1 = model_garch_2_1.fit(update_freq = 5)
results_garch_2_1.summary()
