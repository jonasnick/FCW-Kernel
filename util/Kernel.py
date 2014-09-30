import numpy as np

def LinearKernel(features):
    return np.dot(features, np.array(features).T)
