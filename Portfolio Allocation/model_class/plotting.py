import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys
import csv


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


#########################
## Testing-training split
#########################

def plot_test_train_ret(train_df, test_df, model_name):
    '''
    Generates a plot of the returns of the test-train data split, for each asset.
    '''
    names = train_df.columns
    dim = len(train_df.columns)
    fig, axs = plt.subplots(dim, 1, figsize=(2*dim, 10))
    for j in range(dim):
        axs[j].bar(train_df.index, train_df[names[j]], color="blue")
        axs[j].bar(test_df.index, test_df[names[j]], color="red")
        axs[j].set_title(names[j])
        axs[j].set_ylabel("Daily Log Return")
    os.chdir(sys.path[0]+"/plots/")
    fig.savefig("{}_test_train_split_ret.png".format(model_name) , dpi=250)

def plot_test_train_raw(train_df, test_df, model_name):
    '''
    Generates a plot of the balance of the test-train data split, for each asset.
    '''
    names = train_df.columns
    dim = len(train_df.columns)
    fig, axs = plt.subplots(dim, 1, figsize=(2*dim, 10))
    for j in range(dim):
        axs[j].plot(train_df.index, train_df[names[j]], color="blue")
        axs[j].plot(test_df.index, test_df[names[j]], color="red")
        axs[j].set_title(names[j])
        axs[j].set_ylabel("Balance")
    os.chdir(sys.path[0]+"/plots/")
    fig.savefig("{}_test_train_split_raw.png".format(model_name) , dpi=250)




#########################
## Sampling
#########################

def plot_algo_cum_returns(cum_returns, algo_names, model_name, num=100):
    '''
    Plots samples of the forward-looking return paths for each algo.
    '''
    nalgos = cum_returns.shape[2]
    nsamps = cum_returns.shape[0]
    fig, axs = plt.subplots(nalgos,figsize=(3*nalgos, 13))
    fig.suptitle("Cumulative Return", fontsize=14)

    for j in range(nalgos):
        for i in range(num):
            axs[j].plot(cum_returns[-i,:,j], color="blue", alpha=0.1)
            axs[j].set_xlabel("Algo {}".format(j))
            axs[j].set_ylabel("Return")
    plt.tight_layout()

    os.chdir(sys.path[0]+"/plots/")
    fig.savefig("{}_algos_cum_returns.png".format(model_name), dpi=250)



def compute_bands(vals, levels=[1.0, 5.5, 12.5, 25, 75, 87.5, 94.5, 99.0]):
    '''
    Given an array with shape (x, y, z) where x is the dimension for samples, y is the 
    dimension for time, and z the dimension variable, computes the corresponding percentile bands for credible intervals.
    '''
    def scoreatpercentile(vals, p):
        return np.percentile(vals, p, axis=0)
    perc = {p:scoreatpercentile(vals,p) for p in levels}
    median = np.median(vals, axis=0)
    perc["median"] = median
    return perc

def plot_bayes_cone_cum_ret(train_df, test_df, pred_raw_log_returns, model_name, algo_names, fidx="same", prev_days="all"):
    '''
    Plots the bayesian cone for forward-looking cumulative returns wrt each algorithm.
    '''
    # Some params
    dim = pred_raw_log_returns.shape[2]
    cols = train_df.columns
    n_hist_days = train_df.shape[0]
    
    # Handle prediction index
    if fidx == "same":
        idx = test_df.index
    else:
        idx = fdix
    
    # Get cumulative return from training
    hist_cum_returns = np.exp(train_df.values.cumsum(axis=0))-1.0
    hist_idx = train_df.index
    if prev_days == "all":
        pass
    else:
        hist_cum_returns = hist_cum_returns[-prev_days:,:]
        hist_idx = hist_idx[-prev_days:]
    
    # Now compute future returns
    nsamps = pred_raw_log_returns.shape[0]
    mod_train = np.repeat(train_df.values[np.newaxis, :, :], nsamps, axis=0)
    full_samps = np.exp(np.concatenate([mod_train, pred_raw_log_returns], axis=1).cumsum(axis=1))-1.0
    future_samps = full_samps[:, n_hist_days:, :]
    bands = compute_bands(future_samps)
    
    # True returns.
    true_log_rets = np.concatenate([train_df.values, test_df.values])
    true_cum_rets = np.exp(true_log_rets.cumsum(axis=0))-1.0
    future_true = true_cum_rets[n_hist_days:, :]
    
    # Now plot
    fig, axs = plt.subplots(dim, 1, figsize=(13, 4*dim))
    for j in range(dim):
        # Training data
        axs[j].plot(hist_idx, hist_cum_returns[:,j], color="orange", label="Training")
        
        # Predictions
        axs[j].fill_between(idx, bands[1.0][:,j], bands[99.0][:,j], alpha=0.1, color="b", label="98% CI")
        axs[j].fill_between(idx, bands[5.5][:,j], bands[94.5][:,j], alpha=0.3, color="b", label="89% CI")
        axs[j].fill_between(idx, bands[12.5][:,j], bands[87.5][:,j], alpha=0.5, color="b", label="75% CI")
        axs[j].fill_between(idx, bands[25][:,j], bands[75][:,j], alpha=0.8, color="b", label="50% CI")
        axs[j].plot(idx, bands["median"][:,j], alpha=1.0, color="pink", label="Median")
        
        # True path
        axs[j].plot(test_df.index, future_true[:,j], color="red", label="True")
        
        # Other plotting
        axs[j].set_title(algo_names[j])
        axs[j].set_ylabel("Cumulative % Return")
        axs[j].legend()
    
    os.chdir(sys.path[0]+"/plots/")
    fig.savefig("{}_bayes_cone_ret.png".format(model_name), dpi=250)
    


def plot_bayes_cone_bal(train_df, raw_train_df, test_df, raw_test_df, pred_raw_log_returns, model_name, algo_names, fidx="same", prev_days="all"):
    '''
    Plots the bayesian cone for forward-looking balances of each algo.
    '''
    # Some params
    dim = pred_raw_log_returns.shape[2]
    cols = train_df.columns
    n_hist_days = train_df.shape[0]
    
    # Get cumulative return from training
    hist_bals = raw_train_df.values
    hist_idx = raw_train_df.index
    if prev_days == "all":
        pass
    else:
        hist_bals = hist_bals[-prev_days:,:]
        hist_idx = hist_idx[-prev_days:]
        
    # Handle prediction index
    if fidx == "same":
        idx = test_df.index
    else:
        idx = fdix
        
    # Now back out the implied forecasted balances
    nsamps = pred_raw_log_returns.shape[0]
    mod_train = np.repeat(train_df.values[np.newaxis, :, :], nsamps, axis=0)
    full_samps = np.exp(np.concatenate([mod_train, pred_raw_log_returns], axis=1).cumsum(axis=1))-1.0
    full_bals = raw_train_df.values[0,:]*(1.0+full_samps)
    future_bals = full_bals[:, n_hist_days:, :]
    bands = compute_bands(future_bals)
        
    
    fig, axs = plt.subplots(dim, 1, figsize=(13, 4*dim))
    for j in range(dim):
        # Training data
        axs[j].plot(hist_idx, hist_bals[:,j], color="orange", label="Training")
        
        # Predictions
        axs[j].fill_between(idx, bands[1.0][:,j], bands[99.0][:,j], alpha=0.1, color="b", label="98% CI")
        axs[j].fill_between(idx, bands[5.5][:,j], bands[94.5][:,j], alpha=0.3, color="b", label="89% CI")
        axs[j].fill_between(idx, bands[12.5][:,j], bands[87.5][:,j], alpha=0.5, color="b", label="75% CI")
        axs[j].fill_between(idx, bands[25][:,j], bands[75][:,j], alpha=0.8, color="b", label="50% CI")
        axs[j].plot(idx, bands["median"][:,j], alpha=1.0, color="pink", label="Median")
        
        # True path
        axs[j].plot(raw_test_df.index, raw_test_df.values[:,j], color="red", label="True")
        
        # Other plotting
        axs[j].set_title(algo_names[j])
        axs[j].set_ylabel("Balance ($)")
        axs[j].legend()
    
    os.chdir(sys.path[0]+"/plots/")
    fig.savefig("{}_bayes_cone_bal.png".format(model_name))







