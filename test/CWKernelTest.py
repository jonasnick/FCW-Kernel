# Tests if the output of the python CWKernel implementation is the same as the
# output of the reference implementation. The solution outputs, and the
# corresponding generating matlab file is in the data folder.

import numpy as np
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from CWKernel import CWKernel
from util.DatasetUtil import Matlab

# helper function to deal with inconsistencies when reading 0 or 1 dimensional
# matrices
def expand_dim(X):
    if X.ndim == 0:
        X = np.expand_dims(X, 1)
        X = np.expand_dims(X, 2)
    elif X.ndim == 1:
        X = np.expand_dims(X, 2)
    return X


def test_solution(test_nr, K_train, K_test):
    fnameSolution = "./test/data/Solution"+ str(test_nr);
    K_train_s = np.loadtxt(fnameSolution + '_train.txt')
    K_test_s = np.loadtxt(fnameSolution + '_test.txt')

    K_train_s = expand_dim(K_train_s)
    K_test_s = expand_dim(K_test_s)

    return (np.allclose(K_train,K_train_s)
        and np.allclose(K_test, K_test_s))


def test0():
    A = np.matrix([ [0, 1, 0],
                    [1, 0, 1],
                    [0, 1, 0]])

    train_ind = [2]
    observed_labels = [0]
    test_ind = [0,1]
    num_classes = 2
    walk_length = 5

    [K_train, K_test] = CWKernel(A, train_ind, observed_labels,
                                test_ind, num_classes, walk_length, alpha=0)
    return test_solution(0, K_train, K_test)

def test1():
    A = np.matrix([ [0, 1, 0, 1, 0],
                    [1, 0, 1, 1, 0],
                    [0, 1, 0, 0, 1],
                    [1, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0]])

    train_ind = [0,2,3]
    observed_labels = [0, 0, 1]
    test_ind = [1,4]
    num_classes = 2
    walk_length = 5

    [K_train, K_test] = CWKernel(A, train_ind, observed_labels,
                                test_ind, num_classes, walk_length, alpha=1)
    return test_solution(1, K_train, K_test)


def test2():
    A = np.matrix([ [0, 1, 0, 1, 0],
                    [1, 0, 1, 1, 0],
                    [0, 1, 0, 0, 1],
                    [1, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0]])

    train_ind = [0,2,3]
    observed_labels = [0, 0, 1]
    test_ind = [1,4]
    num_classes = 2
    walk_length = 5

    [K_train, K_test] = CWKernel(A, train_ind, observed_labels,
                                test_ind, num_classes, walk_length, alpha=0.5)
    return test_solution(2, K_train, K_test)


def test3():
    param_dict = Matlab.load('test/data/citeseer.mat')
    X_train = param_dict['X_train']
    X_test = param_dict['X_test']
    labels = param_dict['labels']
    A = param_dict['A']

    [K_train, K_test] = CWKernel(A, X_train, labels[X_train], X_test, max(labels)+1, 6, alpha=0.5)

    return test_solution(3, K_train, K_test)


if __name__ == '__main__':
    tests = [test0, test1, test2, test3]
    for i, t in enumerate(tests):
        s = "Test " + str(i)
        if not t():
            s += " failed"
        else:
            s += " passed"
        print s
