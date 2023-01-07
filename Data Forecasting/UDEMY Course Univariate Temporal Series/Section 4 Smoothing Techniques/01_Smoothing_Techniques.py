# -*- coding: utf-8 -*-
"""
SMOOTHING TECHNIQUES
SIMPLE MOVING AVERAGE METHOD 
author : Gerardo Cano Perea
date : December 16, 2020
"""

# Importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset
airline = pd.read_csv('airline_passengers.csv', index_col = (0), parse_dates = True)
airline.dropna(inplace = True)
airline.head()

"""
SIMPLE MOVING AVERAGE: Promedio movil simple
SMA it is a arithmetic mean for n previous periods.
"""
# Method [rolling] select a set of periods in terms of a specific window.
# SMA calculates the mean value of n previous periods. If there is no n previous periods the value is NaN 
airline['3-month-SMA'] = airline['Thousands of Passengers'].rolling(window = 3).mean()
airline['6-month-SMA'] = airline['Thousands of Passengers'].rolling(window = 6).mean()
airline['12-month-SMA'] = airline['Thousands of Passengers'].rolling(window = 12).mean()

# Plotting Data
airline.plot(figsize = (15,10))
plt.title('SMA Method [Real - 3 months - 6 months - 12 months]')
plt.grid(True)
plt.show()
"""
EXPONENTIAL WEIGTH MOVING AVERAGE METHOD : EWMA
Previous periods far from the current period are less significative. 
Previous periods near to the current period are more significative.
[Span]
[Center of mass]
[Half life]
[Alpha] 
"""

# EWMA Method Implementation. 
airline['EWMA_12'] = airline['Thousands of Passengers'].ewm(span = 12, adjust = False).mean()
airline[['EWMA_12','Thousands of Passengers']].plot(figsize = (15,10))
plt.title('EWMA Method for Span : 12')
plt.grid(True)
plt.show()

# Comparing SMA and EWMA Methods for 12 Previous Periods
airline[['Thousands of Passengers', '12-month-SMA', 'EWMA_12']].plot(figsize = (15,10))
plt.title('Thousands of Passengers - SMA_12 - EWMA_12')
plt.grid(True)
plt.show()














