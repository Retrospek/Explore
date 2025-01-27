import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets


def getIRIS():
    iris = datasets.load_iris()
    irisDF = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    irisDF['target'] = iris.target
    return irisDF    

def processData(test_size, shuffle):
    data = getIRIS() # Nested function call

    X = data.drop(columns=['target'])
    y = data['target']
    #print(f"Columns: {data.columns}")
    #print(f"Unique Targets: {data['target'].unique()}")
    Xarray = X.to_numpy() # Convert to numpy as it will make forward pass in neural network far easier
    Yarray = y.to_numpy() # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #print(f"X Shape: {Xarray.shape}")
    #print(f"Y Shape: {Yarray.shape}")
    X_train, X_test, y_train, y_test = train_test_split(Xarray, Yarray, random_state=42, test_size=test_size, shuffle=shuffle)

    return X_train, X_test, y_train, y_test
