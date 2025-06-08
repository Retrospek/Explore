import math
import numpy as np

class MSELoss:
    def __init__(self):
        self.mse = 0

        def compute(self, truths, predictions):
            self.mse = np.mean(np.sum(np.square(truths - predictions)))
            return self.mse
        
        def gradient(self, truths, predictions):
            self.gradient = 2 * np.mean(predictions - truths)