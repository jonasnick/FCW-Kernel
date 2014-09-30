"""
 An implementation of the coinciding walk kernel (CWK) described in:

   Neumann, M., Garnett, R., and Kersting, K. Coinciding Walk
   Kernels: Parallel Absorbing Random Walks for Learning with Graphs
   and Few Labels. (2013). To appear in: Proceedings of the 5th
   Annual Asian Conference on Machine Learning (ACML 2013).


 The module also contains an extension of the model called Feature-CWK (FCWK).
"""

import numpy as np
import scipy.sparse as sp

def CWKernel(A, train_ind, observed_labels, test_ind, 
                num_classes, walk_length=10, alpha=0.5):
    """
    Args:
        A: the adjacency matrix for the graph under
                    consideration
        train_ind: a list of indices into A comprising the
                    training nodes
        observed_labels: a list of integer labels corresponding to the
                    nodes in train_ind
        test_ind: a list of indices into A comprising the test
                    nodes
        num_classses: the number of classes
        walk_length: the maximum walk length for the CWK
        alpha: the absorbtion parameter to use in [0, 1]
    Returns:
        A conditionally walk kernel for the train and one for the test indices.
    """

    num_nodes = A.shape[0]
    num_train = len(train_ind)
    num_test = len(test_ind)

    prior = np.ones((1, num_classes), dtype=float) / num_classes
    # TODO: non-uniform prior


    # extend graph with special "label" nodes; see paper for details
    A = np.concatenate((A, np.zeros((num_nodes, num_classes))), axis=1);
    A = np.concatenate((A, np.zeros((num_classes, num_nodes + num_classes))), axis=0);

    # "normal" nodes are only reachable with alpha prob, 
    # otherwise you will walk into sink node 
    A = A.astype(float)
    A[train_ind] = (1 - alpha) * A[train_ind]
    probTrained = np.zeros((num_train, num_classes), dtype = float)
    probTrained[range(num_train), observed_labels] = alpha
    A[train_ind, num_nodes:num_nodes+num_classes] = probTrained 
    # walk stays in sink node
    A[num_nodes:num_nodes+num_classes, num_nodes:num_nodes+num_classes] \
        = np.eye(num_classes, dtype=float) 

    # scale A, differs from original implementation
    A = A/A.sum(axis=1)

    A = sp.csr_matrix(A)

    # initialize class probabilities
    probabilities = np.repeat(prior, num_nodes+num_classes, axis=0)
    probTrained = np.zeros((num_train, num_classes), dtype = float)
    probTrained[range(num_train), observed_labels] = 1
    probabilities[train_ind] = probTrained
    probabilities[num_nodes:num_nodes+num_classes] = np.eye(num_classes, dtype = float)

    K_train = np.zeros((num_train, num_train), dtype=float)
    K_test = np.zeros((num_test, num_train), dtype=float)

    for iteration in range(walk_length+1):
        K_train = K_train + probabilities[train_ind].dot(probabilities[train_ind].T)
        K_test = K_test + probabilities[test_ind].dot(probabilities[train_ind].T)

        # not necessary to perform final propagation
        if(iteration == walk_length): 
            break

        probabilities = A.dot(probabilities)
    
    K_train = (1/(float(walk_length+1))) * K_train
    K_test = (1/(float(walk_length+1))) * K_test

    return [K_train, K_test]

def CWFeatureKernel(A, train_ind, test_ind,
                        walk_length = 10, alpha=0.5, node_kernel=None):
    """
    Args:
        A: the adjacency matrix for the graph under
                    consideration
        train_ind: a list of indices into A comprising the
                    training nodes
        test_ind: a list of indices into A comprising the test
                    nodes
        walk_length: the maximum walk length for the CWK
        alpha: the absorbtion parameter to use in [0, 1]
        node_kernel: 
    Returns:
        A feature-CWK for the train and one for the test indices.
    """

    if node_kernel is None:
        node_kernel = np.eye(num_classes, num_classes)

    num_nodes = A.shape[0]
    num_classes = num_nodes
    num_train = len(train_ind)
    num_test = len(test_ind)
    node_ind = range(num_nodes)

    prior = np.ones((1, num_classes), dtype=float) / num_classes
    # TODO: non-uniform prior

    # extend graph with special "label" nodes; see paper for details
    A = np.concatenate((A, np.zeros((num_nodes, num_classes))), axis=1);
    A = np.concatenate((A, np.zeros((num_classes, num_nodes + num_classes))), axis=0);

    # "normal" nodes are only reachable with alpha prob, 
    # otherwise you will walk into sink node 
    A = A.astype(float)
    A = (1 - alpha) * A
    probTrained = np.zeros((num_nodes, num_classes), dtype = float)
    probTrained[node_ind, node_ind] = alpha
    A[node_ind, num_nodes:num_nodes+num_classes] = probTrained 
    # walk stays in sink node
    A[num_nodes:num_nodes+num_classes, num_nodes:num_nodes+num_classes] \
        = np.eye(num_classes, dtype=float) 

    # scale A
    A = A/A.sum(axis=1)

    A = sp.csr_matrix(A)

    # initialize class probabilities
    probabilities = np.eye(num_nodes, dtype = float)
    probabilities = np.concatenate((probabilities, np.eye(num_classes, dtype=float)), axis=0);

    probabilities = sp.csr_matrix(probabilities)

    K_train = np.zeros((num_train, num_train), dtype=float)
    K_test = np.zeros((num_test, num_train), dtype=float)

    for iteration in range(walk_length+1):
        K_train = K_train + probabilities[train_ind].dot(node_kernel) * probabilities[train_ind].T
        K_test = K_test + probabilities[test_ind].dot(node_kernel) * probabilities[train_ind].T

        # not necessary to perform final propagation
        if(iteration == walk_length): 
            break

        probabilities = A.dot(probabilities)
    
    K_train = (1/(float(walk_length+1))) * K_train
    K_test = (1/(float(walk_length+1))) * K_test

    return [K_train, K_test]
