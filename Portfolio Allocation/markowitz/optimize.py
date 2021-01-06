import numpy as np
import pandas as pd
from scipy.optimize import minimize
import csv
import sys

from data_processing import process_data
from loss_function import loss

# The solver below sucks, gets trapped in local mimima.
# def find_optimal_weights(loss, cov, means, q):
#     '''
#     Given the loss function, finds and returns the minizing weight vector.
#     '''
#     from scipy.optimize import minimize
    
#     rand_nums = np.random.randint(100, size=5)
#     rand_weight = rand_nums/rand_nums.sum()
#     q = 0.1
#     result = minimize(loss, rand_weight, args=(cov, means, q))
#     optimal_weight = result.x
    
#     return optimal_weight

# The problem is convex, so we use a convex solver instead. 
def find_optimal_weights(loss, cov, means, q):
    '''
    Given the loss function, finds and returns the minizing weight vector.
    '''
    import cvxpy as cp
    
    w = cp.Variable(5)
    risk_tol = cp.Parameter(nonneg=True)
    ret = means.T*w 
    risk = cp.quad_form(w, cov)
    prob = cp.Problem(cp.Minimize(risk - risk_tol*ret), 
                   [cp.sum(w) == 1, 
                    w >= 0])
    risk_tol.value = q
    prob.solve()
    risk_data = cp.sqrt(risk).value
    ret_data = ret.value
    
    return w.value

if __name__ == "__main__":

    # Process data:
    process_data()

    # Load in components
    cov = pd.read_csv("cov_matrix.csv", index_col=0).values
    means = pd.read_csv("ex_returns.csv", index_col=0).values
    tickers = pd.read_csv("cov_matrix.csv", index_col=0).index.to_list()

    # Run optimizer
    q = 0.6 # This sets the risk tolerance. Example of parameter for the loss function.
    result = find_optimal_weights(loss, cov, means, q)

    # Save weights
    weight_dict = dict(zip(tickers, result))

    with open('optimal_weights.csv', 'w') as f:
        f.truncate(0)
        for key in weight_dict.keys():
            f.write("%s,%s\n"%(key,weight_dict[key]))

    # Print out to console
    print()
    print("Optimal Portfolio Weights:")
    print()
    for key, value in weight_dict.items():
        print("{}: {}".format(key, value))
    print()