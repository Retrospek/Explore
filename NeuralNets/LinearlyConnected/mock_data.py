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
    print(irisDF.columns)
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
        
        self.data = self.data.values    
        
        self.target = self.data['target']
        self.features = self.data.drop(columns=['target'])

        valid_indicies = []

        for i in range(0, len(self.data), batch_size):
            valid_indicies.append(i)
        self.validIdx = valid_indicies
        
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
        #if idx <= self.validIdx: 
                                

data = getIRIS()
dataset = Dataset(data, test_size=0.2, shuffle=True, batch_size=32)


def DataLoader():



    
