"""
WORKING WITH TEMPORAL SERIES
author : Gerardo Cano Perea
date : December 13, 2020.  
"""

# Importing Packages and Libraries. 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns 
sns.set()

# Loading and Transforming the Datset. 
raw_csv_data = pd.read_csv('Index2018.csv')
df_comp = raw_csv_data.copy()
df_comp.date = pd.to_datetime(df_comp.date, dayfirst = True)
df_comp.set_index('date', inplace = True)
df_comp = df_comp.asfreq('b')
df_comp = df_comp.fillna(method='ffill')

# Creating a new variable and deleting scratch
# Use method df_comp.describe() to check the dataset.
df_comp['market_value'] = df_comp.spx
del df_comp['spx']
del df_comp['dax']
del df_comp['ftse']
del df_comp['nikkei']

# Splitting the datset [80/20]. 
size = int(len(df_comp)*0.8)
[df, df_test] = [df_comp.iloc[:size],df_comp.iloc[size:]]

"""
CONCEPT : WHITE NOISE IN TEMPORAL SERIES
date : December 13. 2020
"""

# Setting a white noise vector. 
wn = np.random.normal(loc = df.market_value.mean(), scale = df.market_value.std(), size = len(df))
df['wn'] = wn

# Plotting then White - Noise
df.wn.plot(figsize = (20,10))
plt.title('White Noise Time Serie', size = 18)
plt.grid(True)
plt.show()

# Plotting the S&P 500 Market Values.
df.market_value.plot(figsize = (20,10))
plt.title('S&P 500 Market Value', size = 18)
plt.grid(True)
plt.show()

"""
CONCEPT : RANDOM WALKS
date : December 14, 2020
"""

# Defining and Fitting the Random Walk
rw = pd.read_csv('RandWalk.csv')
rw.date = pd.to_datetime(rw.date, dayfirst = True)
rw.set_index('date', inplace = True)
rw.asfreq('b')
df['rw'] = rw.price

# Plotting a Random walk
# The method to generate a random walk is described on UDEMY resources.
df.rw.plot(figsize = (20,10))
plt.title('Random Walk', size = 18)
plt.grid(True)
plt.show()

# Plotting 3 objects. 
df.wn.plot(figsize = (20,10))
df.market_value.plot()
df.rw.plot()
plt.title('White Noise - S&P 500 Values - Random Walk')
plt.legend()
plt.grid(True)
plt.show()

"""
DICKEY-FULLER PROOF TO DEFINE STATIONARITY
Hypothesis Null = The serie is not stationary
Hypothesis ALternative = The serie is stationary
"""

# Method for critical levels for significance and comparisond.
# If the statistical value is less than the Significance Levels [1%-5%-10%], the null hypothesis is rejected.
# If the P-Value is low (near zero), thus the null hypothesis is rejected.
 
# [First value] = Statistical Value   [Second value] = P-Value   
# [Third value] = No of Backsteps to Determine the Statistical Value
# [Fourth value] = ? [Fifth - Seventh] = Significance Levels [1%-5%-10%]
# [Eigth value] = ?

sts.adfuller(df.market_value) # The serie is not stationary
sts.adfuller(df.wn) # The serie is stationary
sts.adfuller(df.rw) # The serie is not stationary

"""
SEASONALITY IN TEMPORAL SERIES
Effecto of [Trend - Seasonal - Residuals]
Classical Decompotion [1. Additive  2. Multiplicative]
"""

# Additive Decomposition
s_dec_additive = seasonal_decompose(df.market_value, model='additive')
s_dec_additive.plot()
plt.show()

#Multiplicative Decomposition
s_dec_multiplicative = seasonal_decompose(df.market_value, model='multiplicative')
s_dec_multiplicative.plot()
plt.show()
 
"""
AUTO CORRELATION FACTORS.
Show how a previous period impact in the value of the current period.
The calculation take in count the effect of intermediate periods.
date  : December 15, 2020
"""

# [lags] No of periods before the current period. 
# [zero = False] The current period is not needed for determine the ACF
# Values > 1 Positive Correlation - Values < 1 Negative Correlation.
# Blue region is for significative values. 
# Inside this region the values are not significatives.
sgt.plot_acf(df.market_value, lags = 40, zero = False)
plt.title('Auto-Correlation Factor for S&P 500 Serie', size = 14)
plt.grid(True)
plt.show()

sgt.plot_acf(df.wn, lags = 40, zero = False)
plt.title('Auto-Correlation Factor for White Noise', size = 14)
plt.grid(True)
plt.show()

sgt.plot_acf(df.rw, lags = 40, zero = False)
plt.title('Auto-Correlation Factor for Random Walk', size = 14)
plt.grid(True)
plt.show()

"""
PARTIAL AUTO CORRELATION FACTORS.
Show how a previous period impact in the value of the current period.
The calculation do not take in count the effect of intermediante periods.
date  : December 15, 2020
"""

# Opt method [ols] is for ordinary least squares.
# Blue region is for significative values. 
# Inside this region the values are not significatives.
sgt.plot_pacf(df.market_value, lags = 40, zero = False, method='ols')
plt.title('Partial Auto-Correlation Factor for S&P 500 Serie', size = 14)
plt.grid(True)
plt.show()

sgt.plot_pacf(df.wn, lags = 40, zero = False, method='ols')
plt.title('Partial Auto-Correlation Factor for White Noise', size = 14)
plt.grid(True)
plt.show()

sgt.plot_pacf(df.rw, lags = 40, zero = False, method='ols')
plt.title('Partial Auto-Correlation Factor for Random Walk', size = 14)
plt.grid(True)
plt.show()










