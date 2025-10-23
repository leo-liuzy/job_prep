import numpy as np

# ----- Utility functions -----
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(a):
    return a * (1 - a)

def binary_cross_entropy(y, y_hat):
    eps = 1e-10
    return -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))

# ----- Data -----
np.random.seed(42)
N, d = 100, 3   # 100 samples, 3 input features
X = np.random.randn(N, d)
y = (np.random.rand(N, 1) > 0.5).astype(float)  # binary labels

# ----- Initialize parameters -----
hidden_dim = 5
W1 = np.random.randn(d, hidden_dim) * 0.1
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, 1) * 0.1
b2 = np.zeros((1, 1))

lr = 0.1
epochs = 1000

# ----- Training loop -----
for epoch in range(epochs):
    # Forward pass
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    y_hat = sigmoid(z2)

    # Compute loss
    loss = binary_cross_entropy(y, y_hat)

    # Backpropagation
    delta2 = y_hat - y                       # (N, 1)
    dW2 = a1.T @ delta2 / N                  # (5, 1)
    db2 = np.mean(delta2, axis=0, keepdims=True)

    delta1 = (delta2 @ W2.T) * sigmoid_deriv(a1)  # (N, 5)
    dW1 = X.T @ delta1 / N                        # (3, 5)
    db1 = np.mean(delta1, axis=0, keepdims=True)

    # Gradient descent update
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    # Logging
    if epoch % 100 == 0:
        acc = np.mean((y_hat >= 0.5) == y)
        print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {acc:.2f}")
