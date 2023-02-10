"""
====================================================================================
GRID SEARCH MODEL HYPERPARAMETER OPTIMIZATION
author : Gerardo Cano Perea
date : March 30, 2021.
====================================================================================
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

# Grid Search Hyperparameter Tuning.
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


