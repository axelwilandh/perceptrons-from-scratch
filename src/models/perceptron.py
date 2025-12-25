import numpy as np

class Perceptron:
    def __init__(self, input_dim, lr=0.01):
        self.lr = lr
        self.w = np.random.randn(input_dim)
        self.b = 0.0

    def g(self, x):
        return np.tanh(x)

    def forward(self, X):
        """
        Compute the network output for input X
        """
        return self.g(X @ self.w - self.b)

    def predict(self, X):
        out = self.forward(X)
        return np.where(out >= 0, 1, -1)

    def fit(self, X, y, epochs=100):
        for epoch in range(epochs):
            for xi, yi in zip(X, y):
                out = self.forward(xi)
                delta = (yi - out) * (1 - out**2)  # g'(x) = 1 - tanh^2
                self.w += self.lr * delta * xi
                self.b -= self.lr * delta
                
            # compute predictions for the full dataset
            preds = self.predict(X)
            acc = (preds == y).mean()
            print(f"Epoch {epoch+1}/{epochs} – Training accuracy: {acc:.2f}")