import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm

import os
import sys

from model import build_model, load_trace

def compute_forecast(trace, fdays=100):
    nsamps = trace["nu"].shape[0] # number of MCMC samples

    # samples of nu
    nus = trace["nu"]

    # samples of mu
    mus = trace["mu"]

    # samples of the chol fact of the cov matrix
    chols = trace["chol2"]

    # Now, because of shape issues we need some new code to generate multiple samples across the different MCMC samples

    # # Generate samples from the standard multivariate normal distribution.
    dim = len(mus[0])
    samps_per_param = fdays # this can be seen as number of days
    zero_means = np.zeros(mus.shape)
    u = np.random.multivariate_normal(np.zeros(dim), np.eye(dim),
                                    size=(len(zero_means), samps_per_param,))
    # u has shape (len(means), nsamples, dim)

    # # Transform u.
    v = np.einsum('ijk,ikl->ijl', u, chols)
    m = np.expand_dims(zero_means, 1)
    t = v + m
    # Now t is distributed N(0, Cov) but is 3-dim, which is what we want

    # Now we need the normalization constants, which are sqrt(U/nu) where U is chi^2_nu distributed
    prefac = np.random.chisquare(nus)
    prefac /= nus
    prefac = np.sqrt(prefac)

    # Now broadcast to the N(0, Cov) samples
    offset = t/prefac[:,None,None]

    # Now add the true mean
    samps = mus[:,None,:] + offset
    raw_returns = samps

    # Calculate cumulative gains for each algo
    cum_returns = (1.0 + samps[:,:,:]).cumprod(axis=1) - 1.0

    # Slice out the cumulative gain at the final time
    ending_returns = cum_returns[:,-1,:]

    return raw_returns, cum_returns, ending_returns


def plot_ending_ret_hist(ending_returns, fdays=None):
    nalgos = ending_returns.shape[1]
    fig, axs = plt.subplots(nalgos,figsize=(13,3*nalgos))
    if fdays == None:
        fig.suptitle("Ending Cumulative Return", fontsize=14)
    else:
        fig.suptitle("Ending Cumulative Return, {} days in the future".format(fdays), fontsize=14)
    for j in range(nalgos):
        num_entries = len(ending_returns[:,j])
        lower = 100*np.quantile(ending_returns[:,j], 0.05)
        upper = 100*np.quantile(ending_returns[:,j], 0.95)
        num_nonneg = np.sum(ending_returns[:,j] >= 0)
        prob_gain = 100*(num_nonneg/num_entries)

        axs[j].hist(ending_returns[:,j], bins=100)
        axs[j].set_xlabel("Algo {}. 95% CI is [{:.1f}%,{:.1f}%]. Prob(gain) = {:.1f}%".format(j, lower, upper, prob_gain))
        axs[j].set_ylabel("Freq")
    plt.tight_layout()

    os.chdir(sys.path[0])
    if fdays == None:
        plt.savefig("ending_ret_distributions.png", dpi=250)
    else:
        plt.savefig("ending_ret_distributions_fdays_{}.png".format(fdays), dpi=250)

def plot_cum_returns(cum_returns, num=100):
    nalgos = cum_returns.shape[2]
    nsamps = cum_returns.shape[0]
    fig, axs = plt.subplots(nalgos,figsize=(13,3*nalgos))
    fig.suptitle("Cumulative Return", fontsize=14)

    for j in range(nalgos):
        for i in range(num):
            axs[j].plot(cum_returns[-i,:,j], color="blue", alpha=0.1)
            axs[j].set_xlabel("Algo {}".format(j))
            axs[j].set_ylabel("Return")
    plt.tight_layout()

    os.chdir(sys.path[0])
    plt.savefig("cum_returns.png", dpi=250)


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

    # Generate plots
    plot_ending_ret_hist(ending_returns, fdays=fdays)
    plot_cum_returns(cum_returns, num=1000)


