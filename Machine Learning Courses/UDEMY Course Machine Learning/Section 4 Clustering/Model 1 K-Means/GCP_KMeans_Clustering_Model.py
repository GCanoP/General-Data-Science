"""
==============================================================================
K-MEANS CLUSTERING MODEL
Modelo de Clustering K-MEANS
author : Gerardo Cano Perea
date : January 11, 2021
==============================================================================
Step 1 : Choose the K number of clusters.
Step 2 : Allocate randomly the number of K points (barycenters).
Step 3 : Assign every data point to the nearest barycenter.
Step 4 : Calculate the new barycenters.
An advanced algorithm to set correctly the barycenters is the K-MEANS ++ Model
______________________________________________________________________________
To select an appropriated number of clusters we use a specific metric.
The Within Cluster Sum of Squares (WSCC) must be minimized
"""

# Importing Relevant Packages
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# Importing the Dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Check the optimal cluster number.
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WSCC Factor')
plt.grid(True)
plt.show()

# Applying the Correct K-Means model.
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

# Graphic Results
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'salmon', label = 'Cluster 01')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'teal', label = 'Cluster 02')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'gold', label = 'Cluster 03')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'steelblue', label = 'Cluster 04')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'seagreen', label = 'Cluster 05')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = "purple", label = 'Barycenters')
plt.title('Customer Cluster')
plt.xlabel('Salary / Annual Income [1000$]')
plt.ylabel('Store Score')
plt.legend()
plt.show()

