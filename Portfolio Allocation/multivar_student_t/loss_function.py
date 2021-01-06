import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys

from model import build_model, load_trace
from forecasting import compute_forecast


def loss_and_optimize(ending_returns, lam=1.0):
    '''
    Defines the loss optimization problem to solve with cvxpy, and returns the solution.
    '''
    import cvxpy as cp

    # Setup the loss function
    ndim = ending_returns.shape[1]
    w = cp.Variable(ndim)
    loss = cp.sum( -1.0*cp.exp( -lam*( ending_returns @ w ) ) )
    prob = cp.Problem(cp.Maximize(loss), 
                       [cp.sum(w) == 1, 
                        w >= 0])
    
    # Solve the problem
    prob.solve()
    print("Solved the problem!")
    sol = w.value

    # Save solution
    names = ["Algo{}".format(i) for i in range(ndim)]
    weight_dict = dict(zip(names, sol))

    os.chdir(sys.path[0])
    with open('optimal_weights.csv', 'w') as f:
        f.truncate(0)
        for key in weight_dict.keys():
            f.write("%s,%s\n"%(key,weight_dict[key]))
 
    return None

# def loss_func(w, ending_returns, lam=1.0):
#     '''
#     Evaluates the loss function. You want this to be speedy, since it gets called repeatedly by the optimizer.
#     You can in theory throw in whatever kinds of penalizing terms you want.
    
#     Here I've implemented the loss function that Thomas Wiecki showed in his Thalesians talk. Lambda controls
#     the level of risk aversion. Loss is essentially E[-exp(-lam*r)].
#     '''
    
#     loss = (-1.0*np.exp(-lam*np.matmul(ending_returns, w))).sum()
#     return loss

def visualize_loss(lams=[0.1, 0.4, 0.7, 1.0], xmin=-3.0, xmax=3.0, npoints=1000):
    '''
    Helps visualize the loss function.
    '''
    fig, axs = plt.subplots(1, 2, figsize=(13,8))
    
    # subplot 1
    dom = np.linspace(xmin, xmax, npoints)/100
    for lam in lams:
        y = - np.exp(-lam*dom)
        axs[0].plot(dom, y, label="lambda = {}".format(lam))
    axs[0].set_xlabel("Portfolio Return")
    axs[0].set_ylabel("Loss Function Given Portfolio Return")
    axs[0].set_title("Small-scale Loss Function")
    axs[0].legend()
    
    # subplot 2
    dom = np.linspace(xmin, xmax, npoints)
    for lam in lams:
        y = - np.exp(-lam*dom)
        axs[1].plot(dom, y, label="lambda = {}".format(lam))
    axs[1].set_xlabel("Portfolio Return")
    axs[1].set_ylabel("Loss Function Given Portfolio Return")
    axs[1].set_title("Large-scale Loss Function")
    axs[1].legend()
    
    os.chdir(sys.path[0])
    plt.savefig("loss_visualization.png", dpi=250)



if __name__ == "__main__":
    
    # Create loss function visualization
    visualize_loss()

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
    cum_returns, ending_returns = compute_forecast(trace, fdays=fdays)

    # Run optimizer
    loss_and_optimize(ending_returns, lam=1.0)
    