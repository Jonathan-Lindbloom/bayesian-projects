import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_market_calendars as mcal
import datetime as dt
import scipy as sp
import pymc3 as pm

def calc_forecast(data, ppc_samples, fdays):
    '''
    Computes samples for the future volatility and return processes. Output like (datetime_idx, vol_process, return_process).
    '''

    # Compute new date index for new future trading days
    last_trade_day = data.index[-1]
    next_trade_day = last_trade_day + dt.timedelta(days=1)
    new_last_trade_day = last_trade_day + dt.timedelta(days=fdays)
    nyse = mcal.get_calendar('NYSE')
    new_mkt_days = nyse.valid_days(start_date=next_trade_day, end_date=new_last_trade_day).tz_localize(None)
    while len(new_mkt_days) < fdays:
        new_last_trade_day += dt.timedelta(days=1)
        new_mkt_days = nyse.valid_days(start_date=next_trade_day, end_date=new_last_trade_day).tz_localize(None)
        
    # Grab the last vals in each sample for the
    prev_s = ppc_samples["s"][:,-1]

    # Grab the samples for sigma
    sigmas = ppc_samples["sigma"]

    # Grab samples of nu
    nus = ppc_samples["nu"]
    
    # Grab samples of mu
    mus = ppc_samples["mu"]
    
    # Now generate another process that continues the previous
    normal_rvs = np.random.normal(scale=np.array([list(sigmas**2) for j in range(fdays)]).T)

    # Center the first obs on the previous
    normal_rvs[:,0] += prev_s

    # Now use cumsum to create the random walk
    new_s_t = normal_rvs.cumsum(axis=1)

    # Transform into the new vol process
    new_vol_proc = np.exp(-2*new_s_t)

    # Draw samples for the new return paths
    new_rets = pm.StudentT.dist(nu=np.array([list(nus) for j in range(fdays)]).T, mu=np.array([list(mus) for j in range(fdays)]).T, lam=1.0/new_vol_proc).random()
    
    return new_mkt_days, new_vol_proc, new_rets

def compute_bands(vals, levels=[5.5, 12.5, 25, 75, 87.5, 94.5]):
    def scoreatpercentile(vals, p):
        return [sp.stats.scoreatpercentile(temp,p) for temp in vals.T]
    perc = {p:scoreatpercentile(vals,p) for p in levels}
    median = np.median(vals, axis=0)
    perc["median"] = median
    return perc

def plot_training_vol_bands(data, bands):
    plt.fill_between(data.index, bands[5.5], bands[94.5], alpha=0.3, color="b", label="89% CI")
    plt.fill_between(data.index, bands[12.5], bands[87.5], alpha=0.5, color="b", label="75% CI")
    plt.fill_between(data.index, bands[25], bands[75], alpha=0.8, color="b", label="50% CI")
    plt.plot(data.index, bands["median"], alpha=1.0, color="r", label="Median")
    plt.legend()
    
    
def plot_forecast_vol_bands(data, hbands, ndates, nbands, prev_dates=20):
    # Historic volatility
    plt.fill_between(data.index, hbands[5.5], hbands[94.5], alpha=0.3, color="b", label="89% CI")
    plt.fill_between(data.index, hbands[12.5], hbands[87.5], alpha=0.5, color="b", label="75% CI")
    plt.fill_between(data.index, hbands[25], hbands[75], alpha=0.8, color="b", label="50% CI")
    plt.plot(data.index, hbands["median"], alpha=1.0, color="r", label="Median")
    
    # Forecasted volatility
    plt.fill_between(ndates, nbands[5.5], nbands[94.5], alpha=0.3, color="y", label="Forecasted 89% CI")
    plt.fill_between(ndates, nbands[12.5], nbands[87.5], alpha=0.5, color="y", label="Forecasted 75% CI")
    plt.fill_between(ndates, nbands[25], nbands[75], alpha=0.8, color="y", label="Forecasted 50% CI")
    plt.plot(ndates, nbands["median"], alpha=1.0, color="k", label="Forecasted Median")
    
    # Limit plot range
    plt.xlim(data.index[-prev_dates], ndates[-1])
    
    plt.legend()
    
    
    
def plot_ret_forecast(data, ndates, nbands):
    plt.bar(data.index, data)
    
    # Forecasted volatility
    plt.fill_between(ndates, nbands[5.5], nbands[94.5], alpha=0.3, color="y", label="Forecasted 89% CI")
    plt.fill_between(ndates, nbands[12.5], nbands[87.5], alpha=0.5, color="y", label="Forecasted 75% CI")
    plt.fill_between(ndates, nbands[25], nbands[75], alpha=0.8, color="y", label="Forecasted 50% CI")
    plt.bar(ndates, nbands["median"], alpha=1.0, color="k", label="Forecasted Median")
    
    
def plot_price_forecast(hist_price, ndates, price_bands, prev_dates=20):
    # Plot historical
    plt.plot(hist_price.index[-prev_dates:], hist_price[-prev_dates:])
    
    # Plot price forecast
    plt.fill_between(ndates, price_bands[5.5], price_bands[94.5], alpha=0.3, color="y", label="Forecasted 89% CI")
    plt.fill_between(ndates, price_bands[12.5], price_bands[87.5], alpha=0.5, color="y", label="Forecasted 75% CI")
    plt.fill_between(ndates, price_bands[25], price_bands[75], alpha=0.8, color="y", label="Forecasted 50% CI")
    plt.legend()
    