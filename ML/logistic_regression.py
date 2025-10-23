import numpy as np

# -------------------------------
# Sigmoid function
# -------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# -------------------------------
# Logistic Regression class
# -------------------------------
class LogisticRegression:
    def __init__(self, input_dim, lr=0.1):
        self.w = np.zeros((input_dim, 1))  # weights
        self.b = 0.0                        # bias
        self.lr = lr

    def predict_proba(self, X):
        """
        X: (N, D)
        returns: probabilities (N, 1)
        """
        z = X @ self.w + self.b  # linear combination
        return sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def compute_loss(self, X, y):
        """
        Binary cross-entropy loss
        X: (N, D), y: (N, 1)
        """
        N = X.shape[0]
        p = self.predict_proba(X)
        # add epsilon for numerical stability
        eps = 1e-8
        loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        return loss

    def train(self, X, y, epochs=1000, verbose=True):
        """
        X: (N, D), y: (N, 1)
        """
        N = X.shape[0]

        for epoch in range(epochs):
            # Forward pass
            p = self.predict_proba(X)

            # Compute gradients
            dz = p - y  # (N,1)
            dw = (X.T @ dz) / N  # (D,1)
            db = np.sum(dz) / N

            # Gradient descent update
            self.w -= self.lr * dw
            self.b -= self.lr * db

            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                loss = self.compute_loss(X, y)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    # Dummy dataset: AND gate
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([[0],[0],[0],[1]], dtype=float)

    model = LogisticRegression(input_dim=2, lr=0.1)
    model.train(X, y, epochs=1000)

    # Predictions
    print("Predictions:")
    print(model.predict(X))
