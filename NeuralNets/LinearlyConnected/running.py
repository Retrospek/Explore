from dataSourcing import Dataset, DataLoader, getIRIS
from network import basicNet
from functions import training_loop, cross_entropy_loss, evaluate
import numpy as np


data = getIRIS()
dataset = Dataset(data, shuffle=True, batch_size=32)
dataloader = DataLoader(dataset)
batched_data = dataloader.data

net = basicNet(inputShape=4, outputShape=3)
loss = cross_entropy_loss()

#print(batched_data[0])
training_loop(epochs=100, alpha=0.01, data=batched_data, nn=net, criterion=loss)