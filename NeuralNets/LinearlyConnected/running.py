from dataSourcing import Dataset, DataLoader, getIRIS
from network import basicNet
from functions import training_loop, cross_entropy_loss
import numpy as np


data = getIRIS()
dataset = Dataset(data, shuffle=True, batch_size=32)
dataloader = DataLoader(dataset)
batched_data = dataloader.data

net = basicNet(inputShape=4, outputShape=3)
loss = cross_entropy_loss()


X, Y = batched_data[0]  # Unpack batched data


TEST_SIZE = 0.2
train_size = np.ceil(len(X) * (1-TEST_SIZE)).astype(int)
# NO need to initialize the test_size as we can just split to the end

X = np.array(X)  
Y = np.array(Y)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = Y[:train_size], Y[train_size:]
print(y_train)
prediction = net.forward(X_train)

training_loop(alpha=0.001, X_train=X_train, y_train=y_train, criterion=loss, epochs=10, nn=net)

print(f"Prediction on X_train Dataset: {prediction}")