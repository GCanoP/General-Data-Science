"""
AUTO REGRESSIVE INTEGRATED MOVING AVERAGE ARIMA
Modelo autorregresivo integrado de medias moviles.
IMPLEMENTATION AUTO-ARIMA PROCESS FOR OPTIMAL SET OF COEFFCIENTS
author : Gerardo Cano Perea  
date : December 23, 2020 
"""

# Importing Packages  
import itertools
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as sts
import statsmodels.graphics.tsaplots as sgt
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import ADFTest

# Importing the dataset. 
sales_data = pd.read_csv('Champagne_Sales.csv') 
sales_data.Month = pd.to_datetime(sales_data.Month)
sales_data.set_index('Month', inplace = True )
sales_data = sales_data.asfreq('MS')
sales_data = sales_data.fillna(method = 'ffill')

# Applying a Deickey - Fuller test with statsmodel.
# Null Hypothesis : The serie is not stationary if p-value > 0.05
# The p-valeu is 0.3639, thus the null hypothesis is accepted.  
sts.adfuller(sales_data.Champagne_Sales) # The serie is not stationary 

# Applying a Dickey - Fuller test with pmdarima. 
# alpha = 0.05 significance basic level
adf_test = ADFTest(alpha = 0.05)
adf_test.should_diff(sales_data) # False : The serie is not stationary. 

# Split into train an test data.
size = int(len(sales_data) * 0.8)
[df_train, df_test] = [sales_data.iloc[:size], sales_data.iloc[size:]]
plt.plot(df_train)
plt.plot(df_test)

# Warning treatment. 
import warnings
warnings.filterwarnings('ignore')

"""
DEFINING A AUTO-ARIMA (BASIC VERSION) MANUAL CODE TO SELECT MODEL ORDER. 
In the interative process, it is possible a order assigment that is not fittable, 
in this case, the specified order is ignored and the process continue. 
The models are saved in a decision matrix string with LLF and AIC discriminators. 
The models selection is based in Akaike Information Criteria and Loglikelihood Criteria.
"""
# Defining the [p,d,q] values for combinations. 
p = d = q = range(1,6)
# Generating all different combinations. 
pdq = list(itertools.product(p,d,q))
 
desicion = []
for param in pdq:
    try:
        mod = ARIMA(df_train, order = param)
        results = mod.fit()
        parameters = str('ARIMA:' + str(param) +'  LL:' + str(results.llf) + '  AIC:' + str(results.aic))
        desicion.append(parameters)
    except:
        continue
# LLF must be maximized, in the other hand, AIC must be minimized. 
# The best model in range is an ARIMA order = (2,1,5)
# ARIMA (2,1,5) Model.
# Note : The ma.L3 coefficient is not significative, but the model is the best 
# among the total combination of models analized. 
model_ar_2_i_1_am_5 = ARIMA(df_train, order = (2,1,5))
results_ar_2_i_1_am_5 = model_ar_2_i_1_am_5.fit()
results_ar_2_i_1_am_5.summary()

# Analyzing the model residuals
df_train['res_ar_2_i_1_ma_5'] = results_ar_2_i_1_am_5.resid
sgt.plot_acf(df_train.res_ar_2_i_1_ma_5[1:], zero = False, lags = 40)
plt.title('ACF for ARIMA(2,1,5) in Dataset Champagne Sales', size = 18)
plt.grid(True)
plt.show() 

# Generating predictions / len approach start/end. 
# start_dt = datetime(1971, 1, 1) /format datetime
# end_dt = datetime(1975,9,1) / format datetime 

# Predictions for train data
start = len(df_train)
end = len(df_train)+len(df_test) - 1
predictions_train = results_ar_2_i_1_am_5.predict(start = start, end = end, dynamic = False, typ = 'levels')

df_train['Champagne_Sales'].plot()
predictions_train.plot()
plt.grid(True)
plt.show()

