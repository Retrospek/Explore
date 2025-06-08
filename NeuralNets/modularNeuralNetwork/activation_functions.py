import math
import numpy as np


class ReLu:
    def __init__(self):
        self.output = None
        self.input = input

    def move(self, x):
        self.output = np.maximum(0, x)
        return self.output
    
    def gradient(self):
        return np.where(self.output > 0, 1, 0)