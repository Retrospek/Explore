from dataSourcing import Dataset, DataLoader, getIRIS
from network import basicNet
import numpy as np


data = getIRIS()
dataset = Dataset(data, shuffle=True, batch_size=32)
dataloader = DataLoader(dataset)
batched_data = dataloader.data

net = basicNet(inputShape=4, outputShape=1)

X, Y = batched_data[0]  # Unpack batched data


TEST_SIZE = 0.2
train_size = np.ceil(len(X) * (1-TEST_SIZE)).astype(int)
# NO need to initialize the test_size as we can just split to the end

X = np.array(X)  
Y = np.array(Y)

X_train, X_test = X[:train_size], X[train_size:]
y_train = Y[:train_size], Y[train_size:]

prediction = net.forward(X_train)

print(f"Prediction on X_train Dataset: {prediction}")

