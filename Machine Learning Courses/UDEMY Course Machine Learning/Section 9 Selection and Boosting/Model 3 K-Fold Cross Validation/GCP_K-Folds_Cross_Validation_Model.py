"""
=======================================================================================
K-FOLD CROSS VALIDATION MODEL.
author : Gerardo Cano Perea.
date : March 30, 2021
=======================================================================================
Main Idea. The training dataset is divided into k-folds, for example 10.
The model will use the data as follows.
First iteration  -> Use [1-9] folds for training and [10] for validation.
Second iteration -> Use [1-8, 10] folds for training and [9] for validation.
Third iteration  -> Use [1-7, 9-10] folds for training and [8] for validation.
Every datapoint is used as training and validation data, so the accuracy is increased.
______________________________________________________________________________________
Compensate the Bias-Variance.
Low Bias.      -> The model's predictions are closer to real data.
High Bias.     -> The model's predictions are not closer to real data.
Low Variance.  -> The model is executed many times and predictions not vary.
High Variance. -> The model is executed many times and predictions vary.
"""

# Importing
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# Dataset.
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Split into train/test
from sklearn.model_selection import train_test_split
[X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Scale Variables.
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# SVC Machine Model.
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predictions.
y_pred = classifier.predict(X_test)

# Precision Metrics.
# Confusion Matrix.
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# k-Fold Cross Validation.
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()
