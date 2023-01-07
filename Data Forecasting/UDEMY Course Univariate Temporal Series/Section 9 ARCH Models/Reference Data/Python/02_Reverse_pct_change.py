"""
Reverse Method for [ptc_change()]
author : Gerardo Cano Perea.
date : December 31, 2020.
"""
# Importing Packages.
import numpy as np 
import pandas as pd

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

# Applying a return pct_change transformation. 
df_comp['returns'] = df_comp.market_value.pct_change(1).mul(100)

# Applying a reverse from return pct_change transformation. 
df_comp['reverse'] = pd.DataFrame((1 + (df_comp['returns']/100)).cumprod() * df_comp.iloc[0,0])

