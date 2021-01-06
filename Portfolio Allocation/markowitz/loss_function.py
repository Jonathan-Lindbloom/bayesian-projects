import numpy as np

def loss(weights, cov, means, q):
    '''
    Loss function for markowitz mean-variance optimization. q \in [0, \infty), represents risk tolerance.
    '''
    term1 = np.matmul( weights.T , np.matmul(cov, weights) )
    term2 = - q*np.matmul( means.T, weights )
    return term1 + term2