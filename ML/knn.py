class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        # KNN is a lazy learner — just store training data
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        X = np.array(X)
        predictions = [self._predict_one(x) for x in X]
        return np.array(predictions)

    def _predict_one(self, x):
        # Compute Euclidean distances to all training samples
        distances = np.linalg.norm(self.X_train - x, axis=1)
        
        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Get their labels
        k_labels = self.y_train[k_indices]
        
        # Majority vote
        most_common = Counter(k_labels).most_common(1)[0][0]
        return most_common
