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
    #print(f"Columns: {data.columns}")
    #print(f"Unique Targets: {data['target'].unique()}")
    Xarray = X.to_numpy() # Convert to numpy as it will make forward pass in neural network far easier
    Yarray = y.to_numpy() # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #print(f"X Shape: {Xarray.shape}")
    #print(f"Y Shape: {Yarray.shape}")
    X_train, X_test, y_train, y_test = train_test_split(Xarray, Yarray, random_state=42, test_size=test_size, shuffle=shuffle)

    return X_train, X_test, y_train, y_test



X_train, X_test, y_train, y_test = processData(test_size=0.2, shuffle=True)

# ----- STEP (2) ----- #    
"""
Goal: Create Necessary Functions
"""

def ReLU(x):
    return np.maximum(0, x)

def Softmax(x):
    x = x - np.max(x, axis=0, keepdims=True)  # Subtract max for numerical stability
    exp_x = np.exp(x)  # Exponentiate the stabilized values
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)  # Normalize to get probabilities

def encoder(real):
    if real == 0:
        real_encoded = [1, 0, 0]
    elif real == 1:
        real_encoded = [0, 1, 0]
    else:
        real_encoded = [0, 0, 1]
    
    return real_encoded
def CategoricalCrossEntropyLoss(prediction, real):
    real_encoded = encoder(real)

    return np.sum(np.multiply(real_encoded, np.log10(prediction)))

# ----- STEP (3) ----- #    
"""
Goal: Create Neural Network Class
"""
class Dense():
    def __init__(self, neuronsIN, neuronsOUT):
        self.neuronsIN = neuronsIN
        self.neuronsOUT = neuronsOUT

    def create_layer(self):
        WBs = {

        }
        WBs["W"] = np.random.randn(self.neuronsOUT, self.neuronsIN)
        WBs["b"] = np.random.randn(self.neuronsOUT, 1)
        
        return WBs

class NeuralNetwork():
    def __init__(self, layerDIMs):
        self.layerDIMs = layerDIMs

    def initialize(self):
        layers = {

        }

        for i in range(len(self.layerDIMs)):
            dense_layer = Dense(neuronsIN=self.layerDIMs[i][0], neuronsOUT=self.layerDIMs[i][1])
            WBs = dense_layer.create_layer()
            layers[i] = WBs
        
        return layers
    
def forwardPASS(WeightsBiases, input, target_shape):
    curr_input = input.reshape(-1, 1)  # Initial input is reshaped to column vector (4, 1)
    for layer in WeightsBiases.keys():
        WBS = WeightsBiases[layer]
        
        out = np.dot(WBS["W"], curr_input) + WBS["b"]
        
        # Apply ReLU for hidden layers, or Softmax for the output layer
        if WBS["W"].shape[0] != target_shape:
            out = ReLU(out)
        else:
            out = Softmax(out)
            return out  
        
        curr_input = out
    
    return curr_input

def backpropogation(predictions, actual_encoded, WeightsBiases):
    init_gradient = np.subtract(predictions - actual_encoded)
    
#def training_loop(epochs):

neuralNET = NeuralNetwork(layerDIMs=[
    [4, 32], # INPUT, OUTPUT
    [32, 16],#   ^      ^
    [16, 3]  #   ^      ^
    ])
layers = neuralNET.initialize()

prediction = forwardPASS(WeightsBiases=layers, input=X_train[0], target_shape=3)
print(y_train[0])
print(prediction)
print(CategoricalCrossEntropyLoss(prediction=prediction.flatten(), real=y_train[0]))


