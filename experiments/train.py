import numpy as np


def train(model, X, y, epochs=20):
    model.fit(X, y, epochs=epochs)
    predictions = model.predict(X)
    acc = (predictions == y).mean()
    print(f"Training accuracy: {acc:.2f}")
    return model

if __name__ == "__main__":
    train()