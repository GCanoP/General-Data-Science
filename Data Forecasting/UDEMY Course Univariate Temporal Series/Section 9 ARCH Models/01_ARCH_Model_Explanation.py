"""
ARCH MODEL EXPLANATION. 
Autoregressive Conditional Heteroscedascity Model. 
Modelos Autorregresivos con Heterocedasticidad Condicional.
These models are defined by two equations for mean(mu) and variance(sigma**2)
author: Gerardo Cano Perea. 
date : December 31, 2020. 
___
Volatility. 
Its presented in a timeserie when variance take random high values. A low volatility
is usually related to a less variance over time. 
Less Variance --> Less Volatility --> Less Risk --> More Security
__
Common Practice. 
It is a common practice transform residuals (e_t) into (e_t)**2.
There is no problems with a sign (positive/negative) handle. 
The penalties for differences between values and predictions are more significatives.
___
Basic Nomenclature for ARCH Models
Heteroscedascity means "different dispersion"
The know metrics for dispersion are std deviation (sigma) and variance (sigma**2)
Condicional Probabilites (P(A|B))
Heteroscedascity + Condicional suggest that ((sigma_t)**2|x_1, x_2, x_3 ... x_n)
___ 
Basic ARCH(1) Model. 
Conditional Variance ---> Var(x_t|x_t-1) = alpha_0 + alpha_1 *(e_t-1)**2
Three Basic ARCH equations. 
x_t = mu_t + e_t   ...(1)
mu_t = c_0 + phi_1 * Mu_t-1   ...(2)
(sigma_t)**2 = alpha_0 + alpha_1 *(e_t-1)**2   ...(3)
We can substitute equation (1) into equation (2)
x_t = c_0 + phi_1 * Mu_t-1 + e_t   ...(4)
"""

# Importing Packages.
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats.distributions import chi2
from math import sqrt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Importing and pre-processing the dataset. 
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

# Split into train/test datasets.
size = len(df_comp)
[df, df_test] = [df_comp.iloc[:size], df_comp.iloc[size:]]

# Creating simple return ans squared returns variables.
df['returns'] = df.market_value.pct_change(1).mul(100)
df['sq_returns'] = df.returns.mul(df.returns)

# Returns plot. 
df.returns.plot(figsize = (12,6))
plt.title('Returns', size = 18)
plt.grid(True)
plt.show() 

# Sq Returns plot. 
df.sq_returns.plot(figsize = (12,6))
plt.title('Sq Returns', size = 18)
plt.grid(True)
plt.show() 

# In ARCH model the PACF are avalible. 
# Computing the PACF for Returns.
sgt.plot_pacf(df.returns[1:], lags = 40, alpha = 0.05, zero = False, method = 'ols')
plt.title('PACF for Returns', size = 18)
plt.grid(True)
plt.show()

# Computing the PACF for Sq Returns.
sgt.plot_pacf(df.sq_returns[1:], lags = 40, alpha = 0.05, zero = False, method = 'ols')
plt.title('PACF for Squared Returns', size = 18)
plt.grid(True)
plt.show()

"""
Modelling a Simple ARCH Model.
The python package 'arch' must be installed previously.
"""
from arch import arch_model

# Simple ARCH. 
model_arch_1 = arch_model(df.returns[1:])
results_arch_1 = model_arch_1.fit(update_freq = 5)
results_arch_1.summary()

# Updating the correct Simple ARCH(1) - p = ARCH's order
model_arch_1 = arch_model(df.returns[1:], mean = 'Constant', vol = 'ARCH', p = 1)
results_arch_1 = model_arch_1.fit(update_freq = 1)
results_arch_1.summary()

"""
HIGHER LAG ARCH MODELS
"""
# ARCH(2)
model_arch_2 = arch_model(df.returns[1:], mean = 'Constant', vol = 'ARCH', p = 2)
results_arch_2 = model_arch_2.fit(update_freq = 5)
results_arch_2.summary()

# ARCH(3)
model_arch_3 = arch_model(df.returns[1:], mean = 'Constant', vol = 'ARCH', p = 3)
results_arch_3 = model_arch_3.fit(update_freq = 5)
results_arch_3.summary()

# ARCH(13)
model_arch_13 = arch_model(df.returns[1:], mean = 'Constant', vol = 'ARCH', p = 13)
results_arch_13 = model_arch_13.fit(update_freq = 5)
results_arch_13.summary()

# ARCH(14)
model_arch_14 = arch_model(df.returns[1:], mean = 'Constant', vol = 'ARCH', p = 14)
results_arch_14 = model_arch_14.fit(update_freq = 5)
results_arch_14.summary()

# The best model is an ARCH(13) p=13. After this order, AIC is not the lowest value in
# comparison with the previous values. 