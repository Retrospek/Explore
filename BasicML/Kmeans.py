import numpy as np

def data(m_rows, n_cols):
    """
    Arguments:
    - m_rows: Number of rows in the data set
    - n_cols: Number of columns in the data set
    """

    data = np.random.rand(m_rows, n_cols)
    #^ Creates a random float dataset that has m data points with n features

    return data


class KMeans:
    def __init__(self, k, data, trials, max_iter):
        """
        Arguments: 
        - k: Means the amount of "clusters" or groups I want to separate the data into
        - data: Well the data type shit
        - trials: This is the amount of trials to find the best "trial" with the lowest variance across clusters
        - max_iter: This is basically the number of times per trial kmeans will adjust the clusters
        
        New Variables:
        - variances: This will hold the variances and values for each cluster, so a dict: []
        """

        self.K = k
        self.data = data
        self.trials = trials
        self.max_iter = max_iter

    def iteration(self):
        self.clusters = np.array() # Will hold all the cluster points for each trial, so (trials, k, features)

        for i in range(self.trials):
            centroid_initialization = np.random.rand(self.k, self.data.shape[1])

            # First assign your classes their centroid assignments
            classes = np.array()
            for j in range(self.data.shape[0]):
                distances = np.
                
    def trialing(self):

    def ranker(self):
