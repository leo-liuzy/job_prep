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
        rand_ids = np.random.choice(len(X), self.n_clusters, replace=False)

        self.centroids = X[rand_ids]

        n_samples = len(X)
        for i in range(self.max_iter):
            # (n_sample, n_cluster)
            distance = np.linalg.norm(X[:, None] - centroids, axis=2)

            # (n_sample)
            closest_centroids = np.argmin(distance， axis=1)
            new_centroids = np.array([
                np.mean(X[closest_centroids == k], axis=0)
                if np.any(closest_centroids == k)
                else centroids[k]
                for k in range(self.n_clusters)
            ])            

            shift = np.linalg.norm(new_centroids - centroids)
            if shift < self.tol:
                break
            self.centroids = new_centroids

        return self.centroids
    
    def predict(self, X):
        distances = np.linalg.norm(X[:, None] - self.self.centroids, axis=2)
        return np.argmin(distances, axis=1)