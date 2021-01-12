import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp

import pymc3 as pm
import cvxpy as cp

from plotting import *

import os
import sys



class PredictionModel:
    '''
    Abstract model class for pymc3 models of future returns.
    '''
    
    def __init__(self):
        
        self.data_set = False
        self.problem_solved = False # We can't solve the problem more than once, else it breaks
        
    
    #############################################################################################################
    
    # There are several ways we might want to feed data into the models. Option 1 is to generate the returns csv
    # using the raw data. Option 2 is to read in a returns csv that has already been created. 
    
    def process_raw_data(self, overwrite=False, output_fname="log_returns.csv"):
        '''
        Processes the raw input data.
        '''
        # Check that we haven't done this already
        if (self.data_set == True) and (overwrite == False):
            raise Exception("You are about to overwrite data that has already been set. Call with overwrite=True to confirm.")
        
        # Change directory
        os.chdir(sys.path[0]+"/raw_data")

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

        # Save copy of the raw data
        self.raw_df = df.copy()
        
        # Convert to daily log returns
        df = np.log(df) - np.log(df.shift(1))
        df = df.dropna()

        # Go back up a directory and save
        os.chdir(sys.path[0])
        df.to_csv(output_fname)
        
        # Save to object
        self.df = df.copy()
        self.data_set = True
        self.calc_data_stats()
        
    def read_data(self, overwrite=False, input_fname="log_returns.csv"):
        # Check that we haven't done this already.
        if (self.data_set == True) and (overwrite == False):
            raise Exception("You are about to overwrite data that has already been set. Call with overwrite=True to confirm.")
        
        # Read returns from file.
        self.df = pd.read_csv(input_fname, index_col="Date", parse_dates=True)
        self.data_set = True
        self.calc_data_stats()
        
    def calc_data_stats(self):
        '''
        Given the input data, computes some basic statistics.
        '''
        # Get shape of the data
        self.nobs, self.ndim = self.df.shape
        
        # Store time index
        self.data_start_date = min(self.df.index)
        self.data_end_date = max(self.df.index)
        
        # Store names
        self.algo_names = list(self.df.columns)
        
    def train_test_split(self, perc=0.7):
        '''
        Splits data into testing and training segments.
        '''
        # Pick cutoff point
        idx = int(perc*self.nobs)
        
        # Store data
        self.train_df = self.df[:idx].copy()
        self.test_df = self.df[idx:].copy()
        self.train = self.train_df.values
        self.test = self.test_df.values
        self.ntrain = len(self.train_df)
        self.ntest = len(self.test_df)
        
        # Also store raw data copies
        self.raw_train_df = self.raw_df[:idx+1].copy()
        self.raw_test_df = self.raw_df[idx+1:].copy()
        
    #####################################################
    # Running the models
    #####################################################
    
    def sample_model(self, samples=10000, ncores=1, burn=2000):
        '''
        Samples the posterior distribution over model parameters and saves the output.
        '''
        os.chdir(sys.path[0])
        with self.model:
            self.trace = pm.sample(samples, cores=ncores)
        self.trace = self.trace[burn:]
        trace_1 = pm.save_trace(self.trace, directory=sys.path[0]+"/sampling_traces/"+"{}/".format(self.name), overwrite=True)
                                
    def load_trace(self):
        '''
        Loads in a pre-sampled trace for the model.
        '''
        self.trace = pm.load_trace(sys.path[0]+"/sampling_traces/"+"{}/".format(self.name), model=self.model)
    
    #####################################################
    # Optimization
    #####################################################
    
    def build_prob(self, lam=1.0, overwrite=False):
        '''
        Constructs the optimization problem in cvxpy.
        '''
#         if (self.problem_solved == True) and (overwrite == False):
#             raise Exception("Cannot build/solve the same problem more than once. Pass in overwrite=True, or create a new PredictionModel to solve again.")
#         if (self.problem_solved == True) and (overwrite == True):
#             del self._w
#             del self._loss
#             del self.opt_prob
#             del self.opt_weights_array
#             del self.opt_weights
#             del self.opt_weights
        # Setup the loss function
        self.opt_prob_lam = lam
        self._w = cp.Variable(self.ndim, name="optimal_weights")
        self._loss = cp.log_sum_exp(-lam*(self.pred_ending_returns @ self._w))

        self.opt_prob = cp.Problem(cp.Minimize(self._loss), 
                               [cp.sum(self._w) == 1, 
                                self._w >= 0])
        
    def solve_prob(self):
        '''
        Solves the cvxpy problem and stores the solution.
        '''
#         if self.problem_solved == True:
#             raise Exception("Cannot build/solve the same problem more than once. Must create a new PredictionModel to solve again.")
        
        self.opt_prob.solve()
        self.opt_weights_array = self.opt_prob.variables()[0].value.copy()
        self.opt_weights = dict(zip(self.algo_names, self.opt_weights_array))
        self.problem_solved = True
    
       
#     def loss_and_optimize(self, lam=1.0):
#         '''
#         Defines the loss optimization problem to solve with cvxpy, and returns the solution.
#         '''

#         # Setup the loss function
#         w = cp.Variable(self.ndim)
#         loss = cp.log_sum_exp(-lam*(self.pred_ending_returns @ w))
        
       
#         prob = cp.Problem(cp.Maximize(loss), 
#                            [cp.sum(w) == 1, 
#                             w >= 0])

#         # Solve the problem
#         #prob.solve(solver=cp.SCS, max_iters=5000, eps=1e-6)
#         prob.solve(solver=cp.SCS)
#         sol = w.value

#         # Save solution
#         weight_dict = dict(zip(self.algo_names, sol))

#         os.chdir(sys.path[0]+"/optimal_weights/")
#         with open('{}_lam_{}.csv'.format(self.name, lam), 'w') as f:
#             # Overwrite any existing data
#             f.truncate(0)
#             for key in weight_dict.keys():
#                 f.write("%s,%s\n"%(key,weight_dict[key]))

#         return None
    
    #####################################################
    # Generate plots
    #####################################################
    
    def gen_plots(self):
        '''
        Generates a bunch of plots
        '''
        # Test-train split
        plot_test_train_ret(self.train_df, self.test_df, self.name)
        plot_test_train_raw(self.raw_train_df, self.raw_test_df, self.name)
        
        # Plot cumalative return sample paths
        plot_algo_cum_returns(self.pred_cum_returns, self.algo_names, self.name, num=1000)
        plot_bayes_cone_cum_ret(self.train_df, self.test_df, self.pred_raw_log_returns, self.name, self.algo_names, prev_days="all")
        plot_bayes_cone_bal(self.train_df, self.raw_train_df, self.test_df, self.raw_test_df, self.pred_raw_log_returns, self.name, self.algo_names, prev_days="all")
        plot_opt_weights(self.opt_weights, self.name)









class MultStudentT(PredictionModel):
    '''
    Implements the multivariate student t model.
    
    All models must implement build_model and compute_forecast.
    '''
    def __init__(self, name="model1"):
        super().__init__()
        self.name = name
        self.model_type = "multivariate student t"
    
    
    def build_model(self):
        '''
        Defines the model for returns.
        '''
        with pm.Model() as self.model:
            
            # Compute correlation matrix
            chol, corr, stds = pm.LKJCholeskyCov(
                "chol", n=self.ndim, eta=2.0, sd_dist=pm.Exponential.dist(1.0), compute_corr=True
            )
            chol2 = pm.Deterministic("chol2", chol)
            cov = pm.Deterministic("cov", chol.dot(chol.T))
            mu = pm.Normal("mu", 0.0, 1.0, shape=self.ndim)
            nu1 = pm.HalfNormal("nu_minus_2", sigma=1)
            nu2 = pm.Deterministic("nu", 2.0+nu1)
            obs = pm.MvStudentT("obs", nu=nu2, mu=mu, chol=chol, observed=self.train_df, shape=(1,self.ndim))
    
    def compute_forecast(self, fdays="None"):
        '''
        Computes forecasting sample trajectories.
        '''
        if fdays == "None":
            fdays = len(self.test_df)
        
        nsamps = self.trace["nu"].shape[0] # number of MCMC samples

        # samples of nu
        nus = self.trace["nu"]

        # samples of mu
        mus = self.trace["mu"]

        # samples of the chol fact of the cov matrix
        chols = self.trace["chol2"]

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
        self.pred_raw_log_returns = mus[:,None,:] + offset
        self.pred_raw_returns = np.exp(self.pred_raw_log_returns) - 1.0

        # Calculate cumulative gains for each algo

        # But we actually have log returns, so its a cumsum instead of a cumprod.
        self.pred_cum_log_returns = self.pred_raw_log_returns.cumsum(axis=1)
        self.pred_cum_returns = np.exp(self.pred_cum_log_returns) - 1.0
        
        # Slice out the cumulative gain at the final time
        self.pred_ending_returns = self.pred_cum_returns[:,-1,:]
        self.pred_ending_log_returns = self.pred_cum_log_returns[:,-1,:]































