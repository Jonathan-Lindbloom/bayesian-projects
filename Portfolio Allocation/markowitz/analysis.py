import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

import sys
import os

def compute_port_stdev(weights, cov):
    '''
    Given weights and covariance matrix, computes portfolio annualized volatility.
    '''
    port_var = np.matmul(weights.T, np.matmul(cov, weights))
    return np.sqrt(port_var)

def compute_port_ret(weights, means):
    '''
    Given weights and means, computes portfolio expected return vector.
    '''
    return np.matmul(means.T, weights)


def plot_frontier(cov, means, opt_w, num=100):
    '''
    Given cov matrix and expected returns, plots the portfolio space. 
    '''
    # Compute returns and stdevs for random portfolios
    rand_nums = np.random.randint(100, size=(num, 5))
    rand_nums = rand_nums.astype(float)
    rand_weights = rand_nums.copy()
    for j in range(rand_nums.shape[1]):
        rand_weights[:,j] = np.true_divide(rand_nums[:,j], rand_nums.sum(axis=1))
        
    returns = []
    stdevs = []
    for j in range(num):
        returns.append(compute_port_ret(rand_weights[j,:], means))
        stdevs.append(compute_port_stdev(rand_weights[j,:], cov))
    
    ret_opt = compute_port_ret(opt_w, means)
    stdev_opt = compute_port_stdev(opt_w, cov)

    plt.scatter(stdevs, returns, color="blue", label="Random Portfolios")
    plt.scatter(stdev_opt, ret_opt, color="red", s=40, label="Optimal Portfolio")
    plt.xlabel("Portfolio Volatility")
    plt.ylabel("Portfolio Expected Return")
    plt.title("Risk-Return Tradeoff")
    plt.legend()

    os.chdir(sys.path[0])
    plt.savefig("risk_return_tradeoff", dpi=250)
    plt.show()



if __name__ == "__main__":

    # Navigate to cwd
    os.chdir(sys.path[0])

    # Load in components
    cov = pd.read_csv("cov_matrix.csv", index_col=0).values
    means = pd.read_csv("ex_returns.csv", index_col=0).values

    # Read in dict of optimal weights
    with open('optimal_weights.csv', mode='r') as infile:
        reader = csv.reader(infile)
        opt_w = {rows[0]:float(rows[1]) for rows in reader}
        opt_w = np.array(list(opt_w.values()))

    # Plot risk-return tradeoff
    plot_frontier(cov, means, opt_w, num=4000)





