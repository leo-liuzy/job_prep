import numpy as np

def linear_regression(X, y, lr=0.01, epochs=1000):
    n, d = X.shape
    w = np.zeros((d, 1))
    b = 0.0
    y = y.reshape(-1, 1)

    for _ in range(epochs):
        y_pred = X @ w + b
        error = y_pred - y
        dw = (1/n) * X.T @ error
        db = (1/n) * np.sum(error)
        w -= lr * dw
        b -= lr * db
    
    return w, b