import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

# ---------- PROBLEM STATEMENT ---------- # 
"""
    Model: Create a linearly connected (with relu activation function) Neural Network
    Goal: Attempt to classify iris flowers into their separate categories
    Tools: ONLY NUMPY AND MATH

        Considerations: 
        - Because this is a classification task, softmax or sigmoid will be needed
        - However while this is a classification task, ReLU will be utilized for all preceding DNN layers
"""

# ----- STEP (1) ----- #    
"""
Goal: Grab IRIS Data and reshape into numpy array
"""

def getIRIS():
    iris = datasets.load_iris()
    irisDF = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    irisDF['target'] = iris.target
    return irisDF    

def processData(test_size, shuffle):
    data = getIRIS() # Nested function call

    X = data.drop(columns=['target'])
    y = data['target']
    print(f"Columns: {data.columns}")
    print(f"Unique Targets: {data['target'].unique()}")
    Xarray = X.to_numpy() # Convert to numpy as it will make forward pass in neural network far easier
    Yarray = y.to_numpy() # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    print(f"X Shape: {Xarray.shape}")
    print(f"Y Shape: {Yarray.shape}")
    X_train, X_test, y_train, y_test = train_test_split(Xarray, Yarray, random_state=42, test_size=test_size, shuffle=shuffle)

    return X_train, X_test, y_train, y_test



X_train, X_test, y_train, y_test = processData(test_size=0.2, shuffle=True)

# ----- STEP (2) ----- #    
"""
Goal: Construct all necessary functions ==> activation functions, dropout, batch normalization, loss, evaluation, etc.
"""

def sigmoid(x):
    z = 1 / (1 + np.exp(-x))
    return z

def softmax(x):
    exp_values = np.exp(x)
    exp_values_sum = np.sum(exp_values)
 
    return exp_values/exp_values_sum

def ReLU(x):
    if x <= 0:
        return 0
    else:
        return x

def cost(prediction, actual):
    return np.square(prediction - actual)

# ----- STEP (3) ----- #    
"""
Goal: Neural Network Creation => with forward and backward pass and propogation

Architecture: Strict, DNN ==> 3 Hidden Layers

"""
def neuralNet(takeIN, pushOUT): # PYTORCH Inspiration
    """
    Arguments:

    takeIN - Describes the multiple layers and the amount of matrix input they are getting from the previous layer
    pushOUT - Describes the number of neurons in the current layer and how much the next layer should expect

    Note: BTW neural nets are just a bunch of weights and biases combined with non-linear activation function to find
    pretty damn cool trends in data
    """
    np.random.seed(42)

    W1 = np.random.randn(takeIN[0], pushOUT[0])
    b1 = np.random.randn(takeIN[0], 1)

    W2 = np.random.randn(takeIN[1], pushOUT[1])
    b2 = np.random.randn(takeIN[1], 1)

    W3 = np.random.randn(takeIN[2], pushOUT[2])
    b3 = np.random.randn(pushOUT[2], 1)

    WB = {
        'W1': W1,
        'b1': b1,

        'W2':W2,
        'b2':b2,

        'W3':W3,
        'b3':b3       
    }

    return WB

def forward_pass(x, weights):
    W1 = weights['W1']
    b1 = weights['b1']
    
    W2 = weights['W2']
    b2 = weights['b2']

    W3 = weights['W3']
    b3 = weights['b3']

    firstLayer = np.dot(W1, x) + b1#             [****************************] LAYER 1
    a1 = ReLU(firstLayer)#                          \/\/\/\/\/\/\/\/\/\/\/\/
    #                                                   \/\/\/\/\/\/\/\/
    secondLayer = np.dot(W2, a1) + b2#                  [**************] LAYER 2
    a2 = ReLU(secondLayer)#                              \/\/\/\/\/\/\/
    #                                                    \/\/\/\/\/\/\/
    thirdLayer = np.dot(W3, a2) + b3#                      [********] LAYER 3
    prediction = softmax(thirdLayer)#                       \/\/\/\/
    

    cache = {
        'W1': W1,
        'b1':b1,
        'W2':W2,
        'b2':b2,
        'W3':W3,
        'b3':b3
    }
    return prediction, cache

def back_prop(prediction, actual, X, cache, alpha):
    # Output layer error
    dz3 = prediction - actual  # Gradient of loss with respect to output
    dW3 = np.dot(dz3.T, cache['a2'])  # Gradient of weights in layer 3
    db3 = dz3.sum(axis=0)  # Gradient of biases in layer 3

    da2 = np.dot(dz3, cache['W3'])  # Backpropagate through W3
    dz2 = da2 * (cache['a2'] > 0)  # Derivative of ReLU
    dW2 = np.dot(dz2.T, cache['a1'])
    db2 = dz2.sum(axis=0)

    da1 = np.dot(dz2, cache['W2']) 
    dz1 = da1 * (cache['a1'] > 0)  
    dW1 = np.dot(dz1.T, X)
    db1 = dz1.sum(axis=0)

    cache['W3'] -= alpha * dW3
    cache['b3'] -= alpha * db3
    cache['W2'] -= alpha * dW2
    cache['b2'] -= alpha * db2
    cache['W1'] -= alpha * dW1
    cache['b1'] -= alpha * db1

    return cache


def training(DNN, epochs, X_train, y_train, alpha):
    for epoch in range(epochs):
        total_cost = 0
        for i in range(len(X_train)):
            x = X_train[i].reshape(1, -1)
            y = np.zeros((1, 3))
            y[0, y_train[i]] = 1

            prediction, cache = forward_pass(x, DNN)

            total_cost += np.sum(cost(prediction, y))

            DNN = back_prop(prediction, y, x, cache, alpha)

        print(f"Epoch {epoch + 1}, Cost: {total_cost / len(X_train)}")


if __name__ == "__main__":

   takeIN =  [4, 32, 16]
   pushOUT = [32, 16, 1]
   WB = neuralNet(takeIN=takeIN, pushOUT=pushOUT)
   training(DNN=WB, epochs=10, X_train=X_train, y_train=y_train, alpha=0.001)