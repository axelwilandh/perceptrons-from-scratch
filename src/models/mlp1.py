import numpy as np

class Mlp1:
    def __init__(self, input_dim, hidden_dim, lr=0.01):
        self.lr = lr

        # hidden layer parameters
        self.W1 = np.random.randn(hidden_dim, input_dim)
        self.b1 = np.zeros(hidden_dim)

        # output layer parameters
        self.W2 = np.random.randn(hidden_dim)
        self.b2 = 0.0

    def g(self, x):
        return np.tanh(x)
    
    def g_prim(self, x): 
        return np.cosh(x)**(-2)
    
    def forward(self, X):
        """
        Forward pass.
        Returns output AND cached intermediates for backprop.
        """
        a1 = X @ self.W1.T - self.b1        # pre-activation hidden
        h  = self.g(a1)                     # hidden activations

        a2 = h @ self.W2 - self.b2          # pre-activation output
        out = self.g(a2)                    # output

        cache = (a1, h, a2)
        return out, cache
    
    def predict(self, X):
        out, _ = self.forward(X)
        return np.where(out >= 0, 1, -1)

    def fit(self, X, y, epochs=100):
        for epoch in range(epochs):

            for xi, yi in zip(X, y):

                # ---------- forward ----------
                out, (a1, h, a2) = self.forward(xi)

                # ---------- backprop ----------
                # output layer delta
                delta2 = (yi - out) * self.g_prim(a2)

                # hidden layer delta (vector!)
                delta1 = self.g_prim(a1) * self.W2 * delta2

                # ---------- updates ----------
                self.W2 += self.lr * delta2 * h
                self.b2 -= self.lr * delta2

                self.W1 += self.lr * np.outer(delta1, xi)
                self.b1 -= self.lr * delta1

            # ---------- monitoring ----------
            outputs, _ = self.forward(X)
            loss = 0.5 * ((y - outputs)**2).mean()
            preds = self.predict(X)
            acc = (preds == y).mean()

            print(f"Epoch {epoch+1}/{epochs} – Loss: {loss:.4f} – Accuracy: {acc:.2f}")