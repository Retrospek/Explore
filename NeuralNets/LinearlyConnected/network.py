import numpy as np
from functions import xavier_normal, ReLU, softmax

"""
PLAN:
- Develop a Dense Layer Class
- Implement the basicNet Class, which uses Dense Layer Class Objects
"""

class Dense:
    def __init__(self, inputDim, outputDim):
        """
        EXAMPLE: x = [[1,2,3,4,5], [6,7,8,9,10]] of dim = (2(BS), 1(rows), 5(column)) and batch_size = 2.
                 But REALLY the dim = (batch_size, input) because the singleton dimension doesn't affect anything
                 Knowing that we don't have column vectors easily in python let's alter the formula to be xW + b = output
                 as dot product will be far easier, and since it's conserved nothing changes
                 W must then have a dimension = (input, output)
        """

        # Obvious Shape initializations
        self.inputDim = inputDim
        self.outputDim = outputDim

        # Not so obvious and intuitive weight initializations
        # Math: Wx + b = Boom output (Z) => W must have some m x r dims and x must have r x n, where m can be = to n
    
        self.weights = xavier_normal((inputDim, outputDim), n_in=inputDim, n_out=outputDim) # (input,output) dim vector for weights
        print(f"Weight Shape: {self.weights.shape}")
        self.biases = np.zeros((outputDim,)) # Just Zeroed out column vector

    def move(self, input):
        """
        Arguments:
        - input of dim (batch_size, input_size)
        - self of the node, which will hold the weights and biases
        """

        Zed = np.dot(input, self.weights) + self.biases

        return Zed


class basicNet:
    def __init__(self, inputShape, outputShape):
        """
        Arguments:
        - Layer Sizes
        - Output Shape
        - Input Shape
        """

        self.dense1 = Dense(inputShape, 32)
        self.dense2 = Dense(32, 32)
        self.dense3 = Dense(32, outputShape)


        # For backpropogatoin we need to find a way to store the layers into one datastructure, so we can access and do back passing
        self.layersBP = [
            self.dense3,
            self.dense2,
            self.dense1
        ]

    def forward(self, x):
        out = self.dense1.move(x)
        out = ReLU(out)

        out = self.dense2.move(out)
        out = ReLU(out)
        
        out = self.dense3.move(out)
        out = softmax(out)

        return out, 
    
    def backpropogation(self, prediction, truth, learning_rate,):
        """
        Notes:
        - You need to use the chain rule in order to find the contribution to the error from each neuron or weight
        - This is the most challenging part that will require some level of for looping from the back, which is the output layer
          back to the first input layer
        """

        # We need to find the gradient of the loss with respect to the output of the last layer
        # We can then use this gradient to find the gradient of the loss with respect to the weights and biases of the last layer

        #Because the cost of softmax it's gradient is just pred - true we can compute easily
        for layer in self.layersBP:
            print(f"Layer Shape: {layer.weights.shape}")
            print(f"Prediction Shape: {prediction.shape}")
            print(f"Truth Shape: {truth.shape}")
            print(f"Layer Biases Shape: {layer.biases.shape}")
            print(f"Layer Weights Shape: {layer.weights.shape}")

            # Compute the gradient of the loss with respect to the weights and biases of the layer
            # We can then use this gradient to update the weights and biases of the layer









        
import numpy as np

"""
# Define batch size and input shape
batch_size = 32
input_dim = 4  # Must match inputShape when initializing basicNet

# Generate dummy input data (batch_size, input_dim)
X_batch = np.random.rand(batch_size, input_dim)  # Random data for testing

# Initialize the network
net = basicNet(inputShape=input_dim, outputShape=3)

# Forward pass with a batch
predictions = net.forward(X_batch)

# Get predicted classes
predicted_classes = np.argmax(predictions, axis=1)

print("Predicted Classes for the batch:", predicted_classes)
print("Predictions Shape:", predictions.shape)  # Should be (32, 3)
"""
