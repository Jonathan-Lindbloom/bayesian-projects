import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_market_calendars as mcal
import datetime as dt
import scipy as sp

import pymc3 as pm
from theano import tensor as tt
from theano import shared
import arviz as az
import seaborn as sns

import os
import sys


def build_model(data):
    
    # with pm.Model() as model:
    #     ndim = data.shape[1]
    #     chol, corr, stds = pm.LKJCholeskyCov(
    #         "chol", n=ndim, eta=2.0, sd_dist=pm.Exponential.dist(1.0), compute_corr=True
    #     )
    #     cov = pm.Deterministic("cov", chol.dot(chol.T))
    #     mu = pm.Normal("mu", 0.0, 1.5, shape=ndim, testval=data.mean(axis=0))
    #     obs = pm.MvNormal("obs", nu=nu, mu=mu, chol=chol, observed=data)

    with pm.Model() as model:
        # Get dimension of model
        ndim = data.shape[1]

        # Compute correlation matrix
        chol, corr, stds = pm.LKJCholeskyCov(
            "chol", n=ndim, eta=2.0, sd_dist=pm.Exponential.dist(1.0), compute_corr=True
        )
        chol2 = pm.Deterministic("chol2", chol)
        cov = pm.Deterministic("cov", chol.dot(chol.T))
        mu = pm.Normal("mu", 0.0, 1.0, shape=ndim)
        nu1 = pm.HalfNormal("nu_minus_2", sigma=1)
        nu2 = pm.Deterministic("nu", 2.0+nu1)
        obs = pm.MvStudentT("obs", nu=nu2, mu=mu, chol=chol, observed=data, shape=(1,ndim))
    
    return model

def sample_model(model, samples=2000):
    os.chdir(sys.path[0])
    with model:
        trace = pm.sample(samples, cores=1)
    trace_1 = pm.save_trace(trace, directory=sys.path[0]+"/sampling_traces/", overwrite=True)

def load_trace(model):
    trace = pm.load_trace(sys.path[0]+"/sampling_traces/", model=model)
    return trace

if __name__ == "__main__":
    
    # Load in the data
    os.chdir(sys.path[0])
    log_rets = pd.read_csv("log_returns.csv", index_col="Date", parse_dates=True)
    data = log_rets.values

    # Build and sample the model
    model = build_model(data)
    sample_model(model)












