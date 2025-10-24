import numpy as np

def sigmoid(z):
    z = np.clip(z, -500, 500) # prevent overflow
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, lambda_reg=0.0, lr=0.1, epochs=1000):
    n, d = X.shape
    w = np.zeros((d, 1))
    b = 0.0
    y = y.reshape(-1, 1)
    
    for _ in range(epochs):
        z = X @ w + b
        y_pred = sigmoid(z)
        error = y_pred - y
        dw = (1/n) * (X.T @ error) + (lambda_reg / n) * w
        db = (1/n) * np.sum(error)
        w -= lr * dw
        b -= lr * db

    return w, b