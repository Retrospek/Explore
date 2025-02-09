from dataSourcing import Dataset, DataLoader, getIRIS
from network import basicNet
from functions import cross_entropy_loss
import numpy as np

def training_loop(epochs, alpha, data, nn, criterion):
    for epoch in range(epochs):
        for batch in data:
            x_batch = batch[0] 
            y_batch = batch[1] 
            batch_losses = 0
            for input, output in zip(x_batch, y_batch):
                input = np.array([input]).T
                #print(output)
                #print(f"x batch shape: {input}")
                #print(f"y batch shape: {output.shape}")
                batch_loss = 0
                
                probabilities = nn.forward(input)  
                #print(f"Probabilities: {probabilities}")

                num_classes = probabilities.shape[0]
                y_true_encoded = np.eye(num_classes)[output]

                batch_loss += criterion.cost(y_true_encoded, probabilities)
                #print(f"Batch Loss ==> {batch_loss}")
                
                nn.backpropogation(learning_rate=alpha, lossFunction=criterion, y_true=output, y_pred=probabilities)
                
                batch_losses += batch_loss
            
            print(f"Batch Loss ==> {batch_losses}")

def evaluate(X_test, y_test, nn):
    """
    Arguments:
    - X_test: Test data
    - y_test: Test labels
    - nn: Neural network model

    Returns:
    - Accuracy of the model
    """
    predictions = []
    correct = 0
    for x, y in zip(X_test, y_test):
        probabilities = nn.forward(x)
        prediction = np.argmax(probabilities)
        predictions.append(prediction)
        if prediction == np.argmax(y):
            correct += 1
    return correct / len(y_test), predictions


data = getIRIS()
dataset = Dataset(data, shuffle=True, batch_size=32)
dataloader = DataLoader(dataset)
batched_data = dataloader.data

net = basicNet(inputShape=4, outputShape=3)
loss = cross_entropy_loss()

training_loop(epochs=100, alpha=0.001, data=batched_data, nn=net, criterion=loss)