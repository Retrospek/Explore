import numpy as np
from functions import xavier_normal, ReLU, softmax, cross_entropy_loss

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
        self.outputDim= outputDim

        # Not so obvious and intuitive weight initializations
        # Math: Wx + b = Boom output (Z) => W must have some m x r dims and x must have r x n, where m can be = to n
    
        self.weights = np.array(xavier_normal((outputDim, inputDim), n_in=inputDim, n_out=outputDim))
        #print(f"Weight Shape: {self.weights.shape}")
        self.biases = np.array([np.ones((outputDim,))]).T # Just Zeroed out column vector

    def move(self, input):
        self.input = input
        """
        Arguments:
        - input of dim (batch_size, input_size)
        - self of the node, which will hold the weights and biases
        """
        #print(f"Input Shape: {input.shape}")
        #print(f"Weight Shape: {self.weights.shape}")
        #print(f"Bias Shape: {self.biases.shape}")
        Zed = np.dot(self.weights, input) + self.biases
        return Zed

class basicNet:
    def __init__(self, inputShape, outputShape):
        """
        Arguments:
        - Layer Sizes
        - Output Shape
        - Input Shape
        """

        self.dense1 = Dense(inputShape, 4)
        self.dense3 = Dense(4, outputShape)

        self.ReLU = ReLU()
        self.softmax = softmax()

        # For backpropogatoin we need to find a way to store the layers into one datastructure, so we can access and do back passing
        self.layersBP = [
            (self.dense3, self.softmax),
            #(self.dense2, self.ReLU),
            (self.dense1, self.ReLU)
        ]

    def forward(self, x):
        out = self.dense1.move(x)
        out = self.ReLU.forward(out)

        #out = self.dense2.move(out)
        #out = self.ReLU.forward(out)
        
        out = self.dense3.move(out)
        #print(f"Output Shape: {out.shape}")
        out = self.softmax.forward(out)
        #print(f"Softmax Output Shape: {out.shape}")
        return out
    
    def backpropogation(self, learning_rate, lossFunction, y_true, y_pred):
        """
        Notes:
        - You need to use the chain rule in order to find the contribution to the error from each neuron or weight
        - This is the most challenging part that will require some level of for looping from the back, which is the output layer
          back to the first input layer

        Ex:


        Layer 1: Relu activation => f(x)                            FORWARD PASS        BACKWARD PROPOGATION
        Layer 2: Relu uses output from layer 1 => g(f(x))               ||                      /\
        Layer 3: Softmax uses output from layer 2 => h(g(f(x)))         \/                      ||

        */\\
        */\**\/\*
        */\**/\/*
        */\**/\/*
        */\/
        """

        num_classes = len(y_pred)
        y_true_encoded = np.eye(num_classes)[y_true] # Pretty important when using cross entropy loss

        print(y_pred.shape)
        print(y_true_encoded.shape)
        Loss = lossFunction.cost(y_true_encoded, y_pred)
        
        dL_dA = lossFunction.gradient(y_true_encoded, y_pred)

        for layer, activation in self.layersBP: # Changer= layer's weights to change
            dA_dZ = activation.gradient(layer.input)

            dL_dZ = dL_dA * dA_dZ # Chain Rule type shit
            print(f"Layer Input {layer.input.shape}")
            print(f"dL_dZ Shape: {dL_dZ.shape}")
            a = np.array([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], 
                                    [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], 
                                    [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], 
                                    [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]])
            b = np.array([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]).T
            result = np.dot(a, b)
            print(a.shape)
            print(b.shape)
            print(result)
            print(dL_dZ.T.shape)
            print(layer.input.shape)            
            dL_dW = np.dot(dL_dZ.T, layer.input)  # weights grads
            dL_dB = np.sum(dL_dZ, keepdims=True)  # idc about this, but good practice

            # Update weights and biases
            layer.weights -= learning_rate * dL_dW #OOOOooOooOoO nice shit for updating weights
            layer.biases -= learning_rate * dL_dB


            #VERY VERY FUCKING IMPORTANT, need a cache to store these gradients for next layer
            dL_dA = np.dot(layer.weights, dL_dZ)
            
import numpy as np

# Define and input shape
input_dim = 4  # Must match inputShape when initializing basicNet

# Generate dummy input data (batch_size, input_dim)
X_batch = np.array([np.random.rand(input_dim)]).T  # Convert to column vector
# Initialize the network
net = basicNet(inputShape=input_dim, outputShape=3)

# Forward pass with a batch
predictions = net.forward(X_batch)
back = net.backpropogation(learning_rate=0.01, lossFunction=cross_entropy_loss(), y_true=2, y_pred=predictions)
# Get predicted classes
predicted_classes = np.argmax(predictions, axis=0)

print(f"Probabilities: {predictions}")
print("Predictions: ", predicted_classes)  # Should be (3,)
