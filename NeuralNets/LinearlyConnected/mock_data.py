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

def Dataset(test_size, shuffle, batch_size):

    """
    Arguments:
    - Test size: Train, Test split
    - Shuffle: True/False => Shuffling the data for non-predictable training patterns
    - batch_size: 

    Steps:
    - Shuffle Data
    - Batch Data
    - Split the batches
    """

    # Load Data
    data = getIRIS()
    data_length = len(data)

    numpyData = data.values
    if shuffle == True:
        shuffledData = np.random.shuffle(numpyData)
    else:
        # not shuffled
        shuffledData = numpyData

    # Find number of batches
    batches = np.ceil(data_length/batch_size).astype(int) # Convert to float 64, so gotta make in int

    valid_indices


def DataLoader():




    
