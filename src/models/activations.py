import numpy as np

def tanh(x):
    return np.tanh(x)
    
def tanh_prim(x): 
    return 1 - np.tanh(x)**2


def relu(x):
    return np.maximum(0, x)

def relu_prim(x):
    return (x > 0).astype(float)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prim(x):
    s = sigmoid(x)
    return s * (1 - s)

ACTIVATIONS = {
    "relu": (relu, relu_prim),
    "sigmoid": (sigmoid, sigmoid_prim),
    "tanh": (tanh, tanh_prim),
}