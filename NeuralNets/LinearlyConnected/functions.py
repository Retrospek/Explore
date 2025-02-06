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
        sum_cost = 0
        for probs in y_true:
            probs = np.clip(probs, 1e-12, 1.0)
            sum_cost += -np.sum(y_true * np.log(probs)) / 1
        return sum_cost #y_true.shape[0]
    
    def gradient(self, y_true, y_pred):
        return y_pred - y_true
    

def training_loop(epochs, alpha, data, nn, criterion):
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in data:
            x_batch = batch[0]
            y_batch = batch[1]
            #print(f"y batch shape: {y_batch.shape}")
            #print(f"x batch shape: {x_batch.shape}")
            batch_loss = 0
            probabilities = nn.forward(x_batch)
            predicted_classes = np.argmax(probabilities, axis=1)

            batch_loss += criterion.cost(y_batch, predicted_classes)
            print(f"Batch Loss ==> {batch_loss}")

            #
            batch_loss_gradient = criterion.gradient(y_batch, probabilities)
            # Backprop portion
            #temp

            epoch_loss += batch_loss

            nn.backpropogation(probabilities, y_batch, alpha)

        print(f"Epoch Loss ==> {epoch_loss}")

def evaluate(X_test, y_test, nn):
    """
    Arguments:
    - X_test: Test data
    - y_test: Test labels
    - nn: Neural network model

    Returns:
    - Accuracy of the model
    """
    predictions = []
    correct = 0
    for x, y in zip(X_test, y_test):
        probabilities = nn.forward(x)
        prediction = np.argmax(probabilities)
        predictions.append(prediction)
        if prediction == np.argmax(y):
            correct += 1
    return correct / len(y_test), predictions


