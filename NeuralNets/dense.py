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
    print(data['target'].unique())
    Xarray = X.to_numpy() # Convert to numpy as it will make forward pass in neural network far easier
    Yarray = y.to_numpy() # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    print(Xarray.shape)
    print(Yarray.shape)
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

def MSE(predictions, targets):
    return np.mean(np.square(predictions - targets))

def weightsBWlayers(X, y):
    weights = np.array()
    for _ in range(X * y):
        weights.append(np.random.randn())

    return weights.reshape(X, y)

def loss(predictions, trues):
    init = np.square(predictions - trues)
    mse = np.sum(init) / len(init)

    return mse


# ----- STEP (3) ----- #    
"""
Goal: Neural Network Creation => with forward and backward pass and propogation

Architecture: input(4) x 32 x 16 x target(1)
"""

def forwardPass(input, w1, w2, w3):
    z1 = input.dot(w1)
    a1 = ReLU(z1)

    z2 = a1.dot(w2)
    a2 = ReLU(z2)

    z3 = a2.dot(w3)
    output = softmax(z3)

    return output

def backPropogation(loss, weights)






