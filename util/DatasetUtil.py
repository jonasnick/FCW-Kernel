import numpy as np
import csv
import scipy
from scipy import io

class LINQ:
    @staticmethod
    def readContent(f):
        reader = csv.reader(f, delimiter='\t')
        ids = {}
        features = []
        labels = []
        for i,row in enumerate(reader):
            ids[row[0]] = i
            features.append(row[1:len(row)-1])
            labels.append(row[-1])
        return [ids, np.matrix(features, dtype='int'), np.array(labels)]

    @staticmethod
    def readAdjacencyMatrix(f, size, ids):
        A = np.zeros((size,size), dtype='int')
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            try:
                u = ids[row[0]]
                v = ids[row[1]]
                A[u,v] = 1
                A[v,u] = 1
            except KeyError, e:
                print str(e) + ' missing in content'
        return A

class Matlab:
    @staticmethod
    def save(fname, mdict):
        scipy.io.savemat(fname, mdict)
    @staticmethod
    def load(fname):
        return scipy.io.loadmat(fname, squeeze_me=True)
    @staticmethod
    def savetxt(fname, data):
        np.savetxt(fname, data, fmt='%.7e', delimiter='\t')
        
