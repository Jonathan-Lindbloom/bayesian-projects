import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys
import csv

from model import build_model, load_trace
from forecasting import compute_forecast

def get_w_opt():
    '''
    Gets portfolio weight dictionary + array of weights
    '''
    os.chdir(sys.path[0])
    with open('optimal_weights.csv', mode='r') as infile:
        reader = csv.reader(infile)
        weight_dict = {rows[0]:float(rows[1]) for rows in reader}
        opt_w = np.array(list(weight_dict.values()))

    return weight_dict, opt_w

def plot_opt_weights(w_opt, names):
    '''
    Generates a plot of the optimal weights.
    '''
    fig, axs = plt.subplots(1,1)
    axs.bar(names, 100*w_opt)
    axs.set_title("Optimal Portfolio Allocation")
    axs.set_xlabel("Algorithm")
    axs.set_ylabel("% Allocation")
    fig.savefig("opt_weights.png", dpi=250)


def plot_return_paths(w_opt, cum_returns, n_to_plot=1000):
    '''
    Generates a plot of the samples of the optimal portfolio's cumulative return over time.
    Also plots histogram of the final cumulative return.
    '''
    # Cumulative return calculations
    cum_ret_shape = cum_returns.shape
    ndays = cum_ret_shape[1]
    port_ret_samps = np.zeros((cum_ret_shape[0], cum_ret_shape[1]))
    for j in range(ndays):
        port_ret_samps[:, j] = np.matmul(cum_returns[:,j,:], w_opt)

    # Ending return calculation
    ending = port_ret_samps[:,-1]

    # Other calculations
    num_entries = len(ending)
    lower = 100*np.quantile(ending, 0.05)
    upper = 100*np.quantile(ending, 0.95)
    num_nonneg = np.sum(ending >= 0)
    prob_gain = 100*(num_nonneg/num_entries)

    # Now plot
    fig, axs = plt.subplots(2,1,figsize=(13,8))
    for j in range(n_to_plot):
        axs[0].plot(port_ret_samps[j, :], color="blue", alpha=0.1)
    axs[0].set_title("Cumulative Return")
    axs[0].set_xlabel("Days")

    axs[1].hist(ending, bins=100, color="blue")
    axs[1].set_title("Histogram of Final Return. 95% CI of [{:.1f}%, {:.1f}%]. P(gain) = {:.1f}%".format(lower, upper, prob_gain))
    fig.savefig("opt_port_return_paths.png", dpi=250)

def plot_random_comparison(w_opt, ending_returns, num_compare=10000):
    '''
    Plots some metrics of random portfolios for comparison.
    '''
    # Generate random weights
    ndim = ending_returns.shape[1]
    rand_nums = np.random.randint(100, size=(num_compare, ndim))
    rand_nums = rand_nums.astype(float)
    rand_weights = rand_nums.copy()
    for j in range(rand_nums.shape[1]):
        rand_weights[:,j] = np.true_divide(rand_nums[:,j], rand_nums.sum(axis=1))
    
    # Some calculations
    rand_weights = rand_weights.reshape((ndim, num_compare))
    rand_rets = np.matmul(ending_returns, rand_weights)
    ex_rand_rets = rand_rets.mean(axis=0)
    opt_ex_return = np.matmul(ending_returns, w_opt).mean()
    stdev_rand_rets = rand_rets.std(axis=0)
    opt_stdev_return = np.matmul(ending_returns, w_opt).std()

    # Now plot
    fig, axs = plt.subplots(1,2,figsize=(20,5))
    
    # Expected return plot
    axs[0].hist(ex_rand_rets, bins=30)
    axs[0].axvline(opt_ex_return, color="red", label="Opt = {:.1f}%".format(100*opt_ex_return))
    axs[0].set_title("Expected Final Returns of Random Portfolios")
    axs[0].legend()

    # Expected stdev plot
    axs[1].hist(stdev_rand_rets, bins=30)
    axs[1].axvline(opt_stdev_return, color="red", label="Opt = {:.1f}%".format(100*opt_stdev_return))
    axs[1].set_title("Standard Deviation of Final Random Portfolio Returns")
    axs[1].legend()

    fig.savefig("random_comparison.png", dpi=250)


def plot_mean_corr(trace):
    '''
    Generates a plot of the mean correlation array.
    '''
    # Calculate mean cov array
    mean_cov = trace["cov"].mean(axis=0)
    mean_cov_diag_inv_d = np.sqrt(np.diag(mean_cov))**(-1)
    mean_cov_diag_inv = np.zeros(mean_cov.shape)
    np.fill_diagonal(mean_cov_diag_inv, mean_cov_diag_inv_d)
    mean_corr_mat = mean_cov_diag_inv @ mean_cov @ mean_cov_diag_inv

    # Plot
    fig, axs = plt.subplots(1,1,figsize=(10,10))
    im = axs.matshow(mean_corr_mat)
    fig.colorbar(im, orientation='vertical')
    fig.savefig("mean_corr_matrix.png", dpi=250)


if __name__ == "__main__":

    # Load data
    os.chdir(sys.path[0])
    log_rets = pd.read_csv("log_returns.csv", index_col="Date", parse_dates=True)
    data = log_rets.values

    # Build model
    model = build_model(data)

    # Get trace
    trace = load_trace(model)

    # Calculate forecast
    fdays=100
    raw_returns, cum_returns, ending_returns = compute_forecast(trace, fdays=fdays)
    
    # Now get optimal weights and plot
    weight_dict, w_opt = get_w_opt()
    names = ["Algo{}".format(i) for i in range(len(w_opt))]
    
    # Make plots
    plot_return_paths(w_opt, cum_returns, n_to_plot=1000)
    plot_opt_weights(w_opt, names)
    plot_random_comparison(w_opt, ending_returns)
    plot_mean_corr(trace)


