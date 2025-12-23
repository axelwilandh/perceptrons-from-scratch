import numpy as np
from src.models.perceptron import Perceptron


def make_linear_data(n=100):
    X = np.random.randn(n, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


if __name__ == "__main__":
    X, y = make_linear_data()

    model = Perceptron(input_dim=2, lr=0.1)
    model.fit(X, y, epochs=20)

    preds = model.predict(X)
    acc = (preds == y).mean()

    print(f"Training accuracy: {acc:.2f}")