# ============================================
# CULTUS K-MEANS ASSIGNMENT – COMPLETE SOLUTION
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# ----------------------------------------------------
# 1. Generate Synthetic Data (4 clusters)
# ----------------------------------------------------

X, y = make_blobs(
    n_samples=1000,
    centers=4,
    cluster_std=1.0,
    random_state=42
)

plt.scatter(X[:,0], X[:,1])
plt.title("Synthetic Data (4 True Clusters)")
plt.show()

# ----------------------------------------------------
# 2. K-Means from Scratch (NumPy implementation)
# ----------------------------------------------------

class KMeansScratch:
    def __init__(self, K=4, max_iters=100, tol=1e-4):
        self.K = K
        self.max_iters = max_iters
        self.tol = tol

    def initialize_centroids(self, X):
        np.random.seed(42)
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.K]]
        return centroids

    def compute_distance(self, X, centroids):
        # Euclidean distance
        return np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

    def fit(self, X):
        # Step 1: Initialize centroids
        self.centroids = self.initialize_centroids(X)

        for _ in range(self.max_iters):

            # Step 2: Assign clusters
            distances = self.compute_distance(X, self.centroids)
            self.labels = np.argmin(distances, axis=1)

            # Step 3: Recalculate centroids
            new_centroids = np.array([
                X[self.labels == k].mean(axis=0)
                for k in range(self.K)
            ])

            # Step 4: Convergence check
            if np.linalg.norm(self.centroids - new_centroids) < self.tol:
                break

            self.centroids = new_centroids

        return self.centroids, self.labels

    def inertia(self, X):
        distances = self.compute_distance(X, self.centroids)
        min_dist = np.min(distances, axis=1)
        return np.sum(min_dist ** 2)

# ----------------------------------------------------
# 3. Elbow Method (K = 2 to 10)
# ----------------------------------------------------

wcss = []

for k in range(2, 11):
    model = KMeansScratch(K=k)
    model.fit(X)
    wcss.append(model.inertia(X))

plt.plot(range(2, 11), wcss, marker='o')
plt.xlabel("K (No. of Clusters)")
plt.ylabel("WCSS")
plt.title("Elbow Method (Scratch K-Means)")
plt.show()

# Optimal K found visually = 4

# ----------------------------------------------------
# 4. Run Scratch K-Means Using Optimal K
# ----------------------------------------------------

best_k = 4
model = KMeansScratch(K=best_k)
centroids, labels = model.fit(X)

plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis')
plt.scatter(centroids[:,0], centroids[:,1], color='red', s=200, marker='X')
plt.title("K-Means (Scratch) – Final Clusters")
plt.show()

# ----------------------------------------------------
# 5. Compare With sklearn KMeans
# ----------------------------------------------------

kmeans = KMeans(n_clusters=4, random_state=42)
sk_labels = kmeans.fit_predict(X)
sk_centroids = kmeans.cluster_centers_

plt.scatter(X[:,0], X[:,1], c=sk_labels, cmap='plasma')
plt.scatter(sk_centroids[:,0], sk_centroids[:,1], color='black', s=200, marker='X')
plt.title("K-Means (sklearn)")
plt.show()

# ----------------------------------------------------
# End of Assignment Code
# ----------------------------------------------------
