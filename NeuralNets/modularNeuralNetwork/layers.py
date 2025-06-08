import math
import numpy as np

class Dense:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = np.random.rand(output_dim, input_dim)
        self.biases = np.random.rand(output_dim)

    def move(self, x):
        self.output = np.dot(self.weights, x) + self.biases
        return self.output

    def gradient(self, x)