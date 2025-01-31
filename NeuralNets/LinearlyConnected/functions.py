import numpy as np

def xavier_normal(shape, n_in, n_out):
    """
    Generates a NumPy array with Xavier normal initialization.

    Arguments:
    - shape: Tuple indicating the shape of the weight matrix.
    - n_in: Number of input neurons.
    - n_out: Number of output neurons.

    Returns:
    - NumPy array of given shape sampled from Xavier normal distribution.
    """
    std = np.sqrt(2 / (n_in + n_out))  # Xavier standard deviation
    return np.random.normal(0, std, size=shape)