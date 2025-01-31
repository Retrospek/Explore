import numpy as np
from NeuralNets.LinearlyConnected.functions import xavier_normal
"""
PLAN:
- Develop a Dense Layer Class
- Implement the basicNet Class, which uses Dense Layer Class Objects
"""

class Dense:
    def __init__(self, inputDim, outputDim):

        # Obvious Shape initializations
        self.inputDim = inputDim
        self.outputDim = outputDim

        # Not so obvious and intuitive weight initializations
        # Math: Wx + b = Boom output (Z) => W must have some m x r dims and x must have r x n, where m can be = to n
        """
        EXAMPLE: x = [[1,2,3,4,5], [6,7,8,9,10]] of dim = (2(BS), 1(rows), 5(column)) and batch_size = 2.
                 But REALLY the dim = (batch_size, input) because the singleton dimension doesn't affect anything
                 Knowing that we don't have column vectors easily in python let's alter the formula to be xW + b = output
                 as dot product will be far easier, and since it's conserved nothing changes
                 W must then have a dimension = (input, output)
        """
        self.weights = xavier_normal((inputDim, outputDim), n_in=inputDim, n_out=outputDim) # (input,output) dim vector for weights
        self.biases = np.zeros((outputDim,)) # Just Zeroed out column vector

    def move(self, input):
        """
        Arguments:
        """

    

class basicNet:
    def __init__(self, ):
        """
        Arguments:
        - Layer Sizes
        - Output Shape
        - Input Shape
        """
