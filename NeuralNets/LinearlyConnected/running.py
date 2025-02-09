from dataSourcing import Dataset, DataLoader, getIRIS
from network import basicNet
from functions import cross_entropy_loss, evaluate
import numpy as np

def training_loop(epochs, alpha, data, nn, criterion):
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in data:
            x_batch = batch[0]
            y_batch = batch[1]
            #print(f"y batch shape: {y_batch.shape}")
            #print(f"x batch shape: {x_batch.shape}")
            batch_loss = 0
            probabilities = nn.forward(x_batch)
            predicted_classes = np.argmax(probabilities, axis=1)

            batch_loss += criterion.cost(y_batch, predicted_classes)
            print(f"Batch Loss ==> {batch_loss}")


            batch_loss_gradient = criterion.gradient(y_batch, probabilities)
            # Backprop portion
            #temp

            epoch_loss += batch_loss

            nn.backpropogation(probabilities, y_batch, alpha)

        print(f"Epoch Loss ==> {epoch_loss}")

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

training_loop(epochs=100, alpha=0.01, data=batched_data, nn=net, criterion=loss)