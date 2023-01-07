"""
==========================================================================================================
FACTOR ANALYSIS MODEL.
author : Gerardo Cano Perea.
date: March 11, 2021.
==========================================================================================================
"""
# Relevant Packages.
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Loading Data.
data = load_iris()
X = StandardScaler().fit_transform(data['data'])
feature_names = data['feature_names']

# Plotting the Covariance.
ax = plt.axes()
im = ax.imshow(np.corrcoef(X.T), cmap = 'RdBu_r', vmin = -1, vmax = 1)
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(list(feature_names), rotation = 90)
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(list(feature_names))
plt.colorbar(im).ax.set_ylabel('$r$', rotation = 0)
ax.set_title('Iris Feature Correlation Matrix')
plt.tight_layout()

