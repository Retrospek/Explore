import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets


def getIRIS():
    """
    Will grab IRIS data and convert into dataframe format
    """
    iris = datasets.load_iris()
    irisDF = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    irisDF['target'] = iris.target  

    return irisDF    

class Dataset:
    def __init__(self, data, batch_size, shuffle):
        """
        Arguments:
        - data: Dataframe from the getIRIS method
        - batch_size: The size for each batch when putting into model
        - shuffle: If data should be shuffled before being put into the model

        Goal: 
        - Initialization: Creates validIdx variables for later calling with the len method
        """
        self.data = data
        if shuffle:
            self.data = data.sample(frac=1)
        
        self.batch_size = batch_size
        self.target = self.data['target'].values
        self.features = self.data.drop(columns=['target']).values
        self.data = self.data.values    

        valid_indicies = []

        for i in range(0, len(data), batch_size):
            valid_indicies.append(i)
        self.validIdx = np.array(valid_indicies)
        
    def len(self):
        """
        Arguments:
        - self
        """
        return len(self.validIdx)

    def step(self, idx):
        """
        Arguments:
        - idx: for the datalaoder method to continuously call the Dataset class
        """

        if idx <= len(self.validIdx):
            # x_batch, y_batch return statement in an array format
            return [self.features[idx: idx+ self.batch_size], self.target[idx: idx + self.batch_size]]
                                
class DataLoader:
    def __init__(self, dataset):
        """
        Arguments:
        - Dataset Object
        """
        self.dataset = dataset
        self.valid_indices = dataset.validIdx

        self.data = []
        # Now we're going to initialize a data array for the dataloader
        for idx in self.valid_indices:
            if self.dataset.step(idx) != None:
                self.data.append(self.dataset.step(idx)) 
        print(f"Tester 1: {self.data}")
        print(f"Tester 2: {self.data[0]}")
        print(f"Tester 3: {self.data[0][0]}")

        # Now you have stored the data in correct batch pairs
        
    def len(self):
        return len(self.valid_indices)


data = getIRIS()
dataset = Dataset(data, shuffle=True, batch_size=32)

dataloader = DataLoader(dataset)
batched_data = dataloader.data

print(f"First DataLoader Dimension: {len(batched_data)}")
print(f"Second DataLoader Dimension: {len(batched_data[0])}")
print(f"Third DataLoader Dimension: {len(batched_data[0][0])}")