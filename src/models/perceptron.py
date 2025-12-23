import numpy as np

class Perceptron:
    def __init__(self, input_dim, lr=0.01):
        self.lr = lr
        self.w = np.zeros(input_dim)
        self.b = 0.0

    def predict(self, X):
        linear_output = X @ self.w + self.b
        return (linear_output >= 0).astype(int)

    def fit(self, X, y, epochs=100):
        for _ in range(epochs):
            for xi, yi in zip(X, y):
                y_hat = self.predict(xi)
                update = self.lr * (yi - y_hat)
                self.w += update * xi
                self.b += update