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

def ReLU(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtracting max for stability
    return exp_x / np.sum(exp_x)

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
        y_pred = np.clip(y_pred, 1e-12, 1.0)  # Avoid log(0) instability
        return -np.sum(y_true * np.log(y_pred)) / 1 #y_true.shape[0]
    
    def gradient(self, y_true, y_pred):
        return y_pred - y_true
    

def training_loop(epochs, alpha, X_train, y_train, nn, criterion):
    for epoch in range(epochs):
        epoch_loss = 0
        for x_batch, y_batch in zip(X_train, y_train):
            print(f"y batch shape: {y_batch.shape}")
            print(f"x batch shape: {x_batch.shape}")
            batch_loss = 0
            probabilities = nn.forward(x_batch)
            
            batch_loss += criterion.cost(y_batch, probabilities)
            batch_loss_graduient = criterion.gradient(y_batch, probabilities)
            # Backprop portion
            #temp
            for layer in nn.layersBP:
                layer.weights = layer.weights - alpha * batch_loss
        print(f"Epoch Loss ==> {epoch_loss}")

