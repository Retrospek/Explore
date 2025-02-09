import numpy as np
from scipy.special import logsumexp

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

class ReLU:
    def __init__(self):
        pass
    def forward(self, x):
        return np.maximum(0, x)
    def gradient(self, x):
        return np.where(x > 0, 1, 0)
    
class softmax:
    def __init__(self):
        pass
    def forward(self, x):
    # Subtracting max for numerical stability
        x = x.T[0]
        exp_x = np.exp(x - np.max(x, keepdims=True))  
        return np.exp(x - logsumexp(x))
    
    def gradient(self, x):
        return x * (1 - x)
    
class cross_entropy_loss:
    def __init__(self):
        pass
    def cost(self, y_true, y_pred):
        """
        Arguments:
        - y_true: Actual classification, or regression values we're predicting
        - y_pred: Predicted classification

        Returns:
        - Cost of a specific training example
        """
        #print("Prediction: ", y_pred)
        #print("Actual: ", y_true)
        y_pred = np.clip(y_pred, 1e-12, 1.0)
        
        loss = -np.sum(y_true * np.log(y_pred), axis=0)  # Sum over classes

        return np.mean(loss)
    
    def gradient(self, y_true, y_pred):
        return y_pred - y_true
    


