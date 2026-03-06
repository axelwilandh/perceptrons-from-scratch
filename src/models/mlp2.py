import numpy as np
from src.models.activations import ACTIVATIONS

class Mlp2:
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, activation, lr=0.01):
        self.activation, self.activation_prim = ACTIVATIONS[activation]
        self.lr = lr

        # hidden layer 1 parameters
        self.W1 = np.random.randn(input_dim, hidden_dim_1)
        self.b1 = 0.1 * np.random.randn(hidden_dim_1)         #small random bias (to get different starts for neurons)

        # hidden layer 2 parameters
        self.W2 = np.random.randn(hidden_dim_1, hidden_dim_2)
        self.b2 = 0.1 * np.random.randn(hidden_dim_2)         #small random bias (to get different starts for neurons)

        # output layer parameters
        self.W3 = np.random.randn(hidden_dim_2)
        self.b3 = 0.0

        self.history = {
            "loss": [],
            "accuracy": []
        }
    
    def forward(self, X):          
        """
        Forward pass.
        Returns output AND cached intermediates for backprop.
        """
        a1 = X @ self.W1 + self.b1        # pre-activation hidden 1
        h1  = self.activation(a1)                     # hidden 1 activations

        a2 = h1 @ self.W2 + self.b2          # pre-activation hidden 2
        h2 = self.activation(a2)                    # hidden 2 activations

        a3 = h2 @ self.W3 + self.b3          # pre-activation output
        out = self.activation(a3)                    # output

        cache = (a1, h1, a2, h2, a3)
        return out, cache
    
    def backward(self, xi, yi, out, cache):
        a1, h1, a2, h2, a3 = cache

        # output layer gradients
        delta3 = (yi - out) * self.activation_prim(a3)   # scalar

        dW3 = delta3 * h2            # (H,)
        db3 = delta3                # scalar

        # 2nd hidden layer gradients
        delta2 = self.activation_prim(a2) * (self.W3 * delta3)

        dW2 = np.outer(h1, delta2)      # (3,3) 
        db2 = delta2 

        # 1st hidden layer gradients
        delta1 = self.activation_prim(a1) * (self.W2 @ delta2)    # (H,)

        dW1 = np.outer(xi, delta1)      # (D, H)
        db1 = delta1

        return dW1, db1, dW2, db2, dW3, db3
    
    def predict(self, X):
        out, _ = self.forward(X)
        return np.where(out >= 0, 1, -1)

    def fit(self, X, y, epochs=100):
        for epoch in range(epochs):

            for xi, yi in zip(X, y):

                # forward
                out, cache = self.forward(xi)

                # backprop
                dW1, db1, dW2, db2, dW3, db3 = self.backward(xi, yi, out, cache)

                # updates
                self.W3 += self.lr * dW3
                self.b3 += self.lr * db3

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