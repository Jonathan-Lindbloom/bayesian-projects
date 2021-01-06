import numpy as np
import pandas as pd
import os
import sys

def process_data():
    '''
    Computes and saves the covariance matrix + vector of expected returns.
    '''
    # Change directory
    os.chdir(sys.path[0]+"/raw_data")
    os.listdir()
    
    # Read in files and grab daily close prices
    df = pd.DataFrame(data={"returns"})
    for n, fname in enumerate(os.listdir()):
        if n == 0:
            ticker = fname.split(".csv")[0]
            df = pd.read_csv(fname, index_col="Date", parse_dates=True)[["Adj Close"]].rename(columns={"Adj Close" : ticker})
        else:
            ticker = fname.split(".csv")[0]
            temp_df = pd.read_csv(fname, index_col="Date", parse_dates=True)[["Adj Close"]].rename(columns={"Adj Close" : ticker})
            df = pd.merge(df, temp_df, left_index=True, right_index=True)
            
    # Convert to daily log returns
    df = np.log(df) - np.log(df.shift(1))
    df = df.dropna()
    
    # Compute cov matrix
    cov = df.cov()
    
    # Compute mean log returns
    means = df.mean()
    
    # Go back up a directory and save
    os.chdir(sys.path[0])
    cov.to_csv("cov_matrix.csv")
    means.to_csv("ex_returns.csv")