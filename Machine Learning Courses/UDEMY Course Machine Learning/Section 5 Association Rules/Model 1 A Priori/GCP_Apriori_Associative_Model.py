"""
================================================================================
APRIORI ASSOCIATIVE MODELS IN MACHINE LEARNING.
Modelos Asociativos Apriori de Aprendizaje Automatizado.
author : Gerardo Cano Perea
date : February 13, 2021
================================================================================
Support Magnitude.
sup(M) = abs(users who watched M) / abs(users)
Confidence Magnitude.
conf(M1 -> M2 ) = abs(users who watched M1 and M2) / abs(users who watched M1)
Lift Magnitude.
lift(M1 -> M2) = conf(M1 -> M2) / sup(M1)
"""
# Import Relevant Packages.
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# Importing Dataset and Preprocessing.
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
probe = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, dataset.shape[0]):
    transactions.append([str(dataset.values[i, j]) for j in range(0, dataset.shape[1])])

# Training the Model.
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Results.
results = list(rules)
print(results[0])
