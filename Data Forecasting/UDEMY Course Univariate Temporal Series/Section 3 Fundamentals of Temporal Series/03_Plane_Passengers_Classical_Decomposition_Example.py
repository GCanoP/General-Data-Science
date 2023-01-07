"""
CLASICAL DECOMPOSITION IN TEMPORAL SERIES
author : Gerardo Cano Perea
date : December 15, 2020
"""

# Importing Pachages and Libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Defining the Dataset
airline = pd.read_csv('airline_passengers.csv', index_col = 'Month', parse_dates = True)
airline.plot()
plt.grid(True)
plt.show()

# Decomposition of Dataset.
# Residuals show in which part the data serie is in order/disorder
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(airline['Thousands of Passengers'], model = 'additive')
result.plot()
plt.show()
result = seasonal_decompose(airline['Thousands of Passengers'], model = 'multiplicative')
result.plot()
plt.show()

