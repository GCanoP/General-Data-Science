"""
=============================================================================
FACEBOOK PROPHET LIBRARY FOR TIME SERIES FORECASTING.
Libreria Facebook Prophet para Pronostico de Series Temporales.
author : Gerardo Cano Perea
date : January 2, 2020.
=============================================================================
"""
# Importing Packages.
import numpy as np
import pandas as pd
from fbprophet import Prophet

# Importing the Dataset. 
df = pd.read_csv('Car_Sales.csv')
df.columns = ['ds','y']
df['ds'] = pd.to_datetime(df['ds'])

# Defining and Fitting the Prophet Model. 
model = Prophet()
model.fit(df)


