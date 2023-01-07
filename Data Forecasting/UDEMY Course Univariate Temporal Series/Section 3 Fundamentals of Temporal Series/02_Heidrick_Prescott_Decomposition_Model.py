"""
HEIDRICK PRESCOTT DECOMPOSITION TEMPORAL SERIES
autor : Gerardo Cano Perea 
date : December 15, 2020
"""

# Importing Packages and Libraries 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

# Importing the dataset. 
df = pd.read_csv('macrodata.csv', index_col = 0, parse_dates = True)

# Ploting
ax = df['realgdp'].plot()
ax.autoscale(axis = 'x', tight = True)
ax.set(ylabel = 'Real GDP')

# Additive Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
s_dec_additive = seasonal_decompose(df['realgdp'], model='additive')
s_dec_additive.plot()
plt.show()

# Hendrick - Prescott Decomposition Filter
# Lambda Values. Monthly [129600]  Quarterly [1600]  Annual[6.25]
from statsmodels.tsa.filters.hp_filter import hpfilter
[gdp_cycle, gdp_trend] = hpfilter(df['realgdp'], lamb=1600)
df['trend'] = gdp_trend
df['cyclic'] = gdp_cycle

# Plotting Results from H-P Filter
df[['trend','cyclic','realgdp']].plot(figsize = (15,10))
plt.show()
df[['trend','realgdp']]['2005-01-01':].plot(figsize = (15,10))
plt.show()