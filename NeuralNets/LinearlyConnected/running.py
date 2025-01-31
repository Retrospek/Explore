from dataSourcing import Dataset, DataLoader, getIRIS
from network import basicNet
import numpy as np


data = getIRIS()
dataset = Dataset(data, shuffle=True, batch_size=32)
#print(f"Valid Indices: {dataset.validIdx}")
#print(f"Data Set Valid Batches: {dataset.len()}")
dataloader = DataLoader(dataset)
batched_data = dataloader.data
#print(f"Batches: {len(batched_data)}")
net = basicNet(inputShape=4, outputShape=1)

X, Y = batched_data[0]  # Unpack batched data
X = np.array(X)  # Convert to NumPy array
Y = np.array(Y)

#print(f"X shape: {X.shape}, Y shape: {Y.shape}")  # Verify expected shape
prediction = net.forward(X)  # Now pass correctly formatted X to the network

print(prediction)