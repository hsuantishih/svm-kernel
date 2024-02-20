# 資料集
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles

# X, y = make_blobs(n_samples=400, centers=2, random_state=0, cluster_std=1.2)
X, y = make_circles(400, factor=.1, noise=.1)
colors = np.array(["red", "green"])


plt.scatter(X[:, 0], X[:, 1], c=colors[y], s=50, cmap='autumn')

plt.show()