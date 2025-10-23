import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        """
        Parameters
        ----------
        n_clusters : int
            Number of clusters.
        max_iter : int
            Maximum number of iterations.
        tol : float
            Tolerance for convergence (based on centroid shift).
        random_state : int, optional
            Seed for reproducibility.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None

    def fit(self, X):
        """
        Fit the K-Means algorithm to the data.
        """
        np.random.seed(self.random_state)
        n_samples = X.shape[0]

        # Step 1: Randomly initialize centroids
        random_idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        # self.centroids: (n_clusters, n_features)
        self.centroids = X[random_idx]

        for i in range(self.max_iter):
            # Step 2: Assign clusters (compute distances to centroids)
            # X[:, np.newaxis] (n_samples, 1, n_features)
            # Broadcasting: (n_samples, n_clusters, n_features)
            # distances: (n_samples, n_clusters)  -> distance of each sample to each centroid
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            # labels: (n_samples,)  -> index of nearest centroid for each sample
            labels = np.argmin(distances, axis=1)

            # Step 3: Compute new centroids as mean of assigned points
            new_centroids = np.array([
                X[labels == k].mean(axis=0) if np.any(labels == k)
                else self.centroids[k]  # handle empty cluster
                for k in range(self.n_clusters)
            ])

            # Step 4: Check for convergence
            shift = np.linalg.norm(self.centroids - new_centroids)
            if shift < self.tol:
                break

            self.centroids = new_centroids

        self.labels_ = labels

    def predict(self, X):
        """
        Assign each sample in X to the nearest centroid.
        """
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
