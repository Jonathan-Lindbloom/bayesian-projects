import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pymc3 as pm
import arviz as az
az.style.use("arviz-darkgrid")

import sys
import os

from model import build_model, load_trace

def plot_mu_vec(trace):
    az.plot_trace(trace, var_names=['mu'])
    plt.savefig("mu_traceplot.png", dpi=250)

def plot_cov_mat(trace):
    az.plot_trace(trace, var_names=['cov'])
    plt.savefig("cov_traceplot.png", dpi=250)

def plot_nu(trace):
    az.plot_trace(trace, var_names=['nu'])
    plt.savefig("nu_traceplot.png", dpi=250)



if __name__ == "__main__":
    # Load data
    os.chdir(sys.path[0])
    log_rets = pd.read_csv("log_returns.csv", index_col="Date", parse_dates=True)
    data = log_rets.values

    # Build model
    model = build_model(data)

    # Get trace
    trace = load_trace(model)

    # # Get ppc
    # with model:
    #     ppc = pm.sample_posterior_predictive(trace, var_names=trace.varnames)

    # Plot mu traceplot
    plot_mu_vec(trace)

    # Plot covariance matrix traceplot
    plot_cov_mat(trace)

    # Plot mu traceplot
    plot_nu(trace)


