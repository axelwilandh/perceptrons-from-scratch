import numpy as np
from src.models.activations import ACTIVATIONS

class Mlp1:
    def __init__(self, input_dim, hidden_dim, activation, lr=0.01):
        self.activation, self.activation_prim = ACTIVATIONS[activation]
        self.lr = lr

        # hidden layer parameters
        self.W1 = np.random.randn(input_dim, hidden_dim)
        self.b1 = 0.1 * np.random.randn(hidden_dim)         #small random bias (to get different starts for neurons)

        # output layer parameters
        self.W2 = np.random.randn(hidden_dim)
        self.b2 = 0.0

        self.history = {
            "loss": [],
            "accuracy": []
        }
    
    def forward(self, X):
        """
        Forward pass.
        Returns output AND cached intermediates for backprop.
        """
        a1 = X @ self.W1 + self.b1        # pre-activation hidden
        h  = self.activation(a1)                     # hidden activations

        a2 = h @ self.W2 + self.b2          # pre-activation output
        out = self.activation(a2)                    # output

        cache = (a1, h, a2)
        return out, cache
    
    def backward(self, xi, yi, out, cache):
        a1, h, a2 = cache

        # output layer gradients
        delta2 = (yi - out) * self.activation_prim(a2)   # scalar

        dW2 = delta2 * h            # (H,)
        db2 = delta2                # scalar

        # hidden layer gradients
        delta1 = self.activation_prim(a1) * self.W2 * delta2    # (H,)

        dW1 = np.outer(xi, delta1)      # (D, H)
        db1 = delta1

        return dW1, db1, dW2, db2

    
    def predict(self, X):
        out, _ = self.forward(X)
        return np.where(out >= 0, 1, -1)

    def fit(self, X, y, epochs=100):
        for epoch in range(epochs):

            for xi, yi in zip(X, y):

                # forward
                out, cache = self.forward(xi)

                # backprop 
                dW1, db1, dW2, db2 = self.backward(xi, yi, out, cache)

                # updates
                self.W2 += self.lr * dW2
                self.b2 += self.lr * db2

                self.W1 += self.lr * dW1
                self.b1 += self.lr * db1

            # monitoring
            outputs, _ = self.forward(X)
            loss = 0.5 * ((y - outputs)**2).mean()
            acc = (self.predict(X) == y).mean()

            self.history["loss"].append(loss)
            self.history["accuracy"].append(acc)

            print(f"Epoch {epoch+1}/{epochs} – Loss: {loss:.4f} – Accuracy: {acc:.2f}")