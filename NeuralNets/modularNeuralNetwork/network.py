import math
import numpy as np
from layers import Dense 
from activation_functions import ReLu

class neuralnetwork:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.ReLu = ReLu()

        self.dense1 = Dense(1, 1)

        self.sequence = [
            self.dense1,
            self.ReLu
        ]

    def forward(self, x):
        out = self.dense1.forward(x)
        return out
    
    def backpropogation(self, y, y_hat, criterion):
        self.loss = criterion(y, y_hat)

        for 