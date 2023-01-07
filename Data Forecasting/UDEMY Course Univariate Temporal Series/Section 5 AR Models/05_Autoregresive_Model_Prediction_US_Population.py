"""
AUTO-REGRESSIVE MODELS FOR TIME SERIES
US-POPULATION EXAMPLE
author : Gerardo Cano Perea  
date : December 19, 2020
"""

# Importing Packages  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loading specific forecast models. 
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.ar_model import ARResults

# Loading the U.S Population dataset. 
df = pd.read_csv('uspopulation.csv', index_col=0, parse_dates = True)
df.index.freq = 'MS'
df.head()

# Plotting the datset.
df['PopEst'].plot(figsize = (20,10))
plt.title('U.S Population', size = 18)
plt.xlabel('Date', size = 18)
plt.ylabel('Population [Thousands]', size = 18)
plt.grid(True)
plt.show()

# Split the datset in train/test
train = df.iloc[:84]
test = df.iloc[84:]

"""
AUTO_REGRSSIVE MODEL [1]
First Order ARM
"""

# Ignore harmless warning.
import warnings
warnings.filterwarnings('ignore')

# method = mle - maximum likelihood, maxlag = 1 - maximul lag model order. 
model_1 = AR(train['PopEst'])
AR1_fit = model_1.fit(maxlag = 1, method = 'mle')
print('Lags', AR1_fit.k_ar)
print('Coefficients:\n', AR1_fit.params)

# Making predictions
start = len(train)
end = len(train) + len(test) - 1
predictions_1 = AR1_fit.predict(start = start, end = end, dynamic = False).rename('AR_1_Predictions')

# Comparing values with the original dataset test.
for i in range (len(predictions_1)):
    print('predicted = ', predictions_1[i], 'expected = ', test['PopEst'][i])

# Plotting the test/predicted values. 
# A Simple 1-lag model is not enough for a reliable prediction. 
train['PopEst'].plot(legend = True)
test['PopEst'].plot(legend = True)
predictions_1.plot(legend = True, figsize = (15,10))
plt.grid(True)
plt.show()

"""
AUTO_REGRSSIVE MODEL [2]
Second Order ARM
"""

# Recall the model was already created above. 
model_2 = AR(train['PopEst'])
AR2_fit = model_2.fit(maxlag = 2, method = 'mle')
print('Lags', AR2_fit.k_ar)
print('Coefficients:\n', AR2_fit.params)

# Making new predicitions. 
predictions_2 = AR2_fit.predict(start = start, end = end, dynamic = False).rename('AR_2_Predictions')

# Plotting Results. 
train['PopEst'].plot(legend = True)
test['PopEst'].plot(legend = True)
predictions_2.plot(legend = True, figsize = (15,10))
plt.grid(True)
plt.show()

"""
GENERAL AR FIT MODEL AUTOMATIC LAG
Automatic Lag Definition
"""
# ic returns the correct lag based on a metric [bic = bayesian information criteria]
model = AR(train['PopEst'])
AR_fit = model.fit(ic = 'bic')
print('lag', AR_fit.k_ar)
print('Coefficients \n', AR_fit.params)

# making predictions with new model.
predictions_8 = AR_fit.predict(start = start, end = end, dynamic = False).rename('AR_Predictions')

# Plotting results
train['PopEst'].plot(legend = True)
test['PopEst'].plot(legend = True)
predictions_8.plot(legend = True, figsize = (15,10))
plt.grid(True)
plt.show()

"""
EVALUATING ERRORS BETWEEN MODELS
"""

from sklearn.metrics import mean_squared_error

labels = ['AR(1)','AR(2)','AR(8)']
preds = [predictions_1, predictions_2, predictions_8]

# Printing the MSE parameter
for i in range(3):
    error = mean_squared_error(test['PopEst'], preds[i])
    print(labels[i],'Error', error)

# Printing the Akaike Information Criteria [for example]
modls = [AR1_fit, AR2_fit, AR_fit]
for i in range (3):
    print(labels[i], 'AIC : ', modls[i].aic)
    
    
# Making a new model with lag=8
# NOW THE TRAINING DATA IS THE FULL DATASET
model_8 = AR(df['PopEst'])
AR8_fit = model_8.fit(maxlag = 8, method = 'mle')    

# Making a Forecast
fcast = AR8_fit.predict(start = len(df), end = len(df)+12, dynamic = False).rename('Forecast')    

# plot the results. 
df['PopEst'].plot(legend = True)
fcast.plot(legend = True, figsize = (15,10))
plt.title('Predictions 8-lag AR Model for 12 months in future', size = 16)
plt.grid(True)
plt.show()    
    
    
    
    
    
    