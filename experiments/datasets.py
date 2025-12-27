import numpy as np


def make_linear_data(p=500, noise_rate=0.1):
    """
    Generate a 2D binary classification dataset using a linear decision boundary.
    Data points are sampled uniformly from the square [-1, 1]^2 and labeled
    according to the sign of n · x + c, with optional label noise.

    Parameters
    ----------
    p : int
        Number of data points to generate.
    noise_rate : float
        Probability of assigning an incorrect label (where state otherwise obeys the condition).

    Returns
    -------
    X : ndarray of shape (p, 2)
        Input feature vectors.
    y : ndarray of shape (p,)
        Labels in {-1, 1}.
    """
    n = np.array([1.0, 1.0])
    c = 0.0

    states = np.random.rand(p, 2) * 2 - 1
    labels = np.empty(p)

    for mu, state in enumerate(states):
        # assign correct label
        label = 1 if np.dot(n, state) + c > 0 else -1

        # flip label with probability = noise_rate
        if np.random.rand() < noise_rate:
            label *= -1

        labels[mu] = label

    return states, labels

def make_double_linear(p=500, noise_rate=0.1):
    
    n = (np.array([ 0.5 , 0.5]), np.array([ 0.5 , 0.5]))
    c = (0.2,-0.2)

    states = np.random.rand(p, 2) * 2 - 1
    labels = np.empty(p)

    for mu, state in enumerate(states):
        # assign correct label
        label = 1 if (np.dot(n[0],state) + c[0] > 0 and np.dot(n[1],state) + c[1] < 0) else -1

        # flip label with probability = noise_rate
        if np.random.rand() < noise_rate:
            label *= -1

        labels[mu] = label

    return states, labels


def make_squares(p=500, noise_rate=0.1):
    
    n = (np.array([ 0.5 , 0.5]), np.array([ 0.5 , 0.5]))
    c = (0.2,-0.2)

    states = np.random.rand(p, 2) * 2 - 1
    labels = np.empty(p)

    for mu, state in enumerate(states):
        # assign correct label
        label = 1 if state[0] * state[1] > 0 else -1

        # flip label with probability = noise_rate
        if np.random.rand() < noise_rate:
            label *= -1

        labels[mu] = label

    return states, labels

