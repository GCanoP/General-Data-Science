"""
AutoRegressive Moving Average - ARIMA MODEL
autor : Gerardo Cano Perea 
date : December 21, 2020. 
ARMA(1) : C + phi_1 * x_t-1 + Phi_1 * e_t-1 + e_t  / First Order ARMA(p,q)
AR(1)MA(1) is a simple combination of AR Model and MA Model.

The AR and MA are good predictors but, the residuals does not have the same behaviour 
as White Noise Gaussian Distribution. For this, the ARMA model is introduced. 
"""

# Importing Packages. 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts 
from statsmodels.tsa.arima_model import ARMA
from scipy.stats.distributions import chi2
import seaborn as sns

# Importing and Preprocessing Dataset. 
raw_csv_data = pd.read_csv('Index2018.csv')
df_comp = raw_csv_data.copy()
df_comp.date = pd.to_datetime(df_comp.date, dayfirst = True)
df_comp.set_index('date', inplace = True)
df_comp = df_comp.asfreq('b')
df_comp = df_comp.fillna(method = 'ffill')

# Transforming Dataset. 
df_comp['market_value'] = df_comp.ftse
del df_comp['spx']
del df_comp['ftse']
del df_comp['dax']
del df_comp['nikkei']
size = int(len(df_comp)* 0.8)
[df, df_test] = [df_comp.iloc[:size], df_comp.iloc[size:]]

# Creating a Log - likelihood Ratio Test
def LLR_test(mod_1, mod_2, DF = 1):
    L1 = mod_1.llf
    L2 = mod_2.llf
    LR = (2*(L2 - L1))
    p = chi2.sf(LR, DF).round(3)
    return p

# Creating Returns variable. 
df['returns'] = df.market_value.pct_change(1)*100

"""
MODELING WITH ARMA(P,Q) = ARMA(1,1)
"""

# If ar.L1.returns 0.7649 > 0.75 there are increases/decreases persistents in terms of previous values.
# For ma.L1.returns -0.8141 the negative signs suggest that past values are avoided for calibration.
model_ret_ar_1_ma_1 = ARMA(df.returns[1:], order = (1,1))
results_ret_ar_1_ma_1 = model_ret_ar_1_ma_1.fit()
results_ret_ar_1_ma_1.summary()

# Making an LLR Test for individuals models. 
model_ret_ar_1 = ARMA(df.returns[1:], order = (1,0))
model_ret_ma_1 = ARMA(df.returns[1:], order = (0,1))
results_ret_ar_1 = model_ret_ar_1.fit()
results_ret_ma_1 = model_ret_ma_1.fit()

# ARMA is significatively different respect the both individual models.
# ARMA 2 coefficients - AR/MA 1 coefficient. Thus the DF = 1 for comparisons. 
# In function LLR_test mod_2 must be the more complex model. In the other hand LLR_test = 1
# CONCLUSION : ARMA is a better predictor than a simple AR/MA Model.
print('ARMA vs AR', LLR_test(results_ret_ar_1, results_ret_ar_1_ma_1))
print('ARMA vs MA', LLR_test(results_ret_ma_1, results_ret_ar_1_ma_1))

"""
HIGHER-LAG ARMA MODELS
"""
# Computing ACF Factor for Returns
sgt.plot_acf(df.returns[1:], zero = False, lags = 40)
plt.title('ACF for Returns', size = 18)
plt.grid(True)
plt.show()

# Computing PACF Factor for Returns.
sgt.plot_pacf(df.returns[1:], zero = False, lags = 40)
plt.title('PACF for Returns', size = 18)
plt.grid(True)
plt.show()

# Include a high order ARMA model can be counterproductive cause coefficients can be non significatives.
# ARMA order = (3,3) DOF = 6
model_ret_ar_3_ma_3 = ARMA(df.returns[1:], order = (3,3))
results_ret_ar_3_ma_3 = model_ret_ar_3_ma_3.fit()
results_ret_ar_3_ma_3.summary()

# Computing the LLR Test
LLR_test(results_ret_ar_1_ma_1, results_ret_ar_3_ma_3, DF = 4)

# ARMA order = (3,2) DOF = 5
# All coefficients are significatives.
model_ret_ar_3_ma_2 = ARMA(df.returns[1:], order = (3,2))
results_ret_ar_3_ma_2 = model_ret_ar_3_ma_2.fit()
results_ret_ar_3_ma_2.summary()

# ARMA order = (2,3) DOF = 5
model_ret_ar_2_ma_3 = ARMA(df.returns[1:], order = (2,3))
results_ret_ar_2_ma_3 = model_ret_ar_2_ma_3.fit()
results_ret_ar_2_ma_3.summary()

# ARMA order = (3,1) DOF = 4
# All coefficients are significative
model_ret_ar_3_ma_1 = ARMA(df.returns[1:], order = (3,1))
results_ret_ar_3_ma_1 = model_ret_ar_3_ma_1.fit()
results_ret_ar_3_ma_1.summary()

# Computing the LLR_test ARMA (3,1) and ARMA (3,2).
# Value 0.01 CONCLUSION : Complexing the model usually gives better predictions.
LLR_test(results_ret_ar_3_ma_1, results_ret_ar_3_ma_2, DF = 1)

# ARMA order = (2,2) DOF = 4
model_ret_ar_2_ma_2 = ARMA(df.returns[1:], order = (2,2))
results_ret_ar_2_ma_2 = model_ret_ar_2_ma_2.fit()
results_ret_ar_2_ma_2.summary()

# ARMA order = (1,3) DOF = 4
model_ret_ar_1_ma_3 = ARMA(df.returns[1:], order = (1,3))
results_ret_ar_1_ma_3 = model_ret_ar_1_ma_3.fit()
results_ret_ar_1_ma_3.summary()

""" 
IMPORTANT NESTED CONDITIONS
MODELS ARMA(p_1, q_1) / more complex and ARMA(p_2, q_2) / less complex 
Conditions for a nested models. 
1. p_1 + q_1 > p_2 + q_2
2. p_1 >= p_2
3. q_1 >= q_2
Example ARMA(3,2) and ARMA(1,3) are not nested models thus LLR_test is not applied. 
In this case we have to compare them manually
"""

# Comparing manual ARMA(3,2) and ARMA(1,3)
# We select the model that produce a higher Log Likelihood and have the lowest AIC/BAIC Parameter. 
print('ARMA(3,2) \nLL : ', results_ret_ar_3_ma_2.llf,'\nAIC : ', results_ret_ar_3_ma_2.aic )
print('ARMA(1,3) \nLL : ', results_ret_ar_1_ma_3.llf,'\nAIC : ', results_ret_ar_1_ma_3.aic )
# The best models is the ARMA (3,2)

"""
ANALYZING THE MODEL RESIDUALS FOR ARMA (3,2)
"""

# Defining the residuals.
df['res_ret_ar_3_ma_2'] = results_ret_ar_3_ma_2.resid[1:]

# Plotting the Residuals
df.res_ret_ar_3_ma_2.plot(figsize = (15,10))
plt.title('Residuals for Returns', size = 18)
plt.grid(True)
plt.show()

# Plotting the ACF
sgt.plot_acf(df.res_ret_ar_3_ma_2[2:], zero = False, lags = 40)
plt.title('ACF for Residuals Returns')
plt.grid(True)
plt.show()

"""
RE-EVALUATING THE SELECTED MODEL
"""
# ARMA order = (5,5)
model_ret_ar_5_ma_5 = ARMA(df.returns[1:], order = (5,5))
results_ret_ar_5_ma_5 = model_ret_ar_5_ma_5.fit()
results_ret_ar_5_ma_5.summary()

# ARMA order = (5,1)
model_ret_ar_5_ma_1 = ARMA(df.returns[1:], order = (5,1))
results_ret_ar_5_ma_1 = model_ret_ar_5_ma_1.fit()
results_ret_ar_5_ma_1.summary()

# ARMA order = (1,5)
model_ret_ar_1_ma_5 = ARMA(df.returns[1:], order = (1,5))
results_ret_ar_1_ma_5 = model_ret_ar_1_ma_5.fit()
results_ret_ar_1_ma_5.summary()

# Comparing the new ARMA models.
print('ARMA(5,1) \nLL : ', results_ret_ar_5_ma_1.llf,'\nAIC : ', results_ret_ar_5_ma_1.aic )
print('ARMA(1,5) \nLL : ', results_ret_ar_1_ma_5.llf,'\nAIC : ', results_ret_ar_1_ma_5.aic )
# The best model is ARMA(5,1)

# If we compare ARMA(5,1) with ARMA (3,2). THE BEST MODEL IS ARMA(5,1)

"""
ANALYZING THE RESIDUAL MODEL FOR ARMA (5,1)
"""
df['res_ret_ar_5_ma_1'] = results_ret_ar_5_ma_1.resid

# Plotting the ACF Factor for ARMA(5,1)
sgt.plot_acf(df.res_ret_ar_5_ma_1[1:], zero = False, lags = 40)
plt.title('ACF for Returns Resdiuals', size = 18)
plt.grid(True)
plt.show()

"""
ARMA MODELS FOR ORIGINAL PRICES VARIABLE
"""
# Computing ACF for values
sgt.plot_acf(df.market_value, unbiased = True, zero = False, lags = 40)
plt.title('ACF for Market Values FTSE', size = 18)
plt.grid(True)
plt.show()

# Computing PACF for values
sgt.plot_pacf(df.market_value, lags = 40, alpha = 0.05, zero = False, method = 'ols')
plt.title('PACF for Market Values FTSE', size = 18)
plt.grid(True)
plt.show()

# Build a simple ARMA (1,1) 
model_ar_1_ma_1 = ARMA(df.market_value, order = (1,1))
results_ar_1_ma_1 = model_ar_1_ma_1.fit()
results_ar_1_ma_1.summary()

# Examining Residuals of model
df['res_ar_1_ma_1'] = results_ar_1_ma_1.resid
sgt.plot_acf(df.res_ar_1_ma_1, zero = False, lags = 40)
plt.title('ACF for Market Value Model Residuals')
plt.grid(True)
plt.show()

# Modeling ARMA (6,6)
# ValueError: The computed initial AR coefficients are not stationary.You should induce 
# stationarity, choose a different model order, or you can pass your own start_params.
model_ar_6_ma_6 = ARMA(df.market_value, order = (6,6))
results_ar_6_ma_6 = model_ar_6_ma_6.fit(start_ar_lags = 11)
results_ar_6_ma_6.summary()

# Modeling ARMA(6,2)
# The next step is compute the ARMA(6, n) and ARMA(n, 6) models. 
model_ar_6_ma_2 = ARMA(df.market_value, order = (6,2))
results_ar_6_ma_2 = model_ar_6_ma_2.fit()
results_ar_6_ma_2.summary()

df['res_ar_6_ma_2'] = results_ar_6_ma_2.resid
sgt.plot_acf(df.res_ar_6_ma_2, zero = False, lags = 40)
plt.title('ACF for Residuals of Prices', size = 18)
plt.grid(True)
plt.show()

"""
ARMA for Returns vs ARMA for Prices
"""

# CONCLUSION: AR/MA and ARMA models do not make good prediction for not stationary
print('ARMA(6,2) \nLL : ', results_ar_6_ma_2.llf,'\nAIC : ', results_ar_6_ma_2.aic )
print('ARMA(5,1) \nLL : ', results_ret_ar_5_ma_1.llf,'\nAIC : ', results_ret_ar_5_ma_1.aic )
# The ARMA(6,2) for a non stationary serie have a poor LL and AIC values.
# ARMA(6,2) 
# LL :  -27589.75001826396 
# AIC :  55199.50003652792
# ARMA(5,1) 
# LL :  -7889.31128294591 
# AIC :  15794.62256589182