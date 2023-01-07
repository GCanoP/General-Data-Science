"""
====================================================================================
HIERARCHICAL CLUSTERING MODEL.
Modelo de Cluster Jerarquico.
author : Gerardo Cano Perea.
date : February 12, 2021
====================================================================================
Steps for Algorithm.
Step 1 : Every point is an independent cluster.
Step 2 : Choose two nearest point and convert in a new cluster. N-1 Clusters
Step 3 : Choose two nearest clusters and turn in a new cluster. N-2 Clusters
The Algorithm is Supported in a Dendrogram Plot.
"""
# Importing Relevant Packages.
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# Importing the Dataset.
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Use the Dendrogram to select the optimal number of clusters.
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

# Creating an Agglomerate Hierarchical Clustering
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# Graphic Cluster Results
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 01')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 02')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 03')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'purple', label = 'Cluster 04')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'yellow', label = 'Cluster 05')
plt.title('Hierarchical Customer Cluster')
plt.legend()
plt.xlabel('Annual Income [1000$]')
plt.ylabel('Customer Score')
plt.show()
