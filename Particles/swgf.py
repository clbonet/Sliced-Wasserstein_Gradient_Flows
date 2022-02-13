import torch
import matplotlib.pyplot as plt
import numpy as np

from copy import deepcopy
from tqdm.auto import trange
from scipy.stats import norm


device = "cuda" if torch.cuda.is_available() else "cpu"


def one_step_flow(n_epochs, x_prev, J, h, num_projections, lr, 
                  device, plot_train, emd1D, sliced_wasserstein):
    """
        Perform gradient descent at time step t
        
        Inputs:
        - n_epochs
        - x_prev: previous sample
        - J: functional (taking (x,rho) as inputs)
        - h: time step
        - num_projections
        - lr
        - delta: volume of each cell of the grid
        - device
        
        Outputs:
        - rho_{k+1}^h
    """
    
    n = x_prev.size(0)
    d = x_prev.size(1)

    x_k = deepcopy(x_prev).requires_grad_(True)

    optimizer = torch.optim.SGD([x_k], lr=lr, momentum=0.9)
    
    train_loss = []
    w_loss = []
    J_loss = []

    for j in range(n_epochs):
        if d>1:
            sw = sliced_wasserstein(x_k, x_prev, num_projections, device, p=2)
        else:
            sw = emd1D(x_k.reshape(1,-1), x_prev.reshape(1,-1), p=2)            
        
        f = J(x_k)
        loss = sw+2*h*f

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        train_loss.append(loss.item())
        w_loss.append(sw.item())
        J_loss.append(2*h*f.item())
                
            
    if plot_train:
        fig, ax = plt.subplots(1,3,figsize=(15,5))
        ax[0].plot(train_loss, label="full loss")
        ax[0].plot(J_loss, label="2hJ")
        ax[0].legend()
        ax[1].plot(w_loss, label="SW")
        ax[1].legend()
        ax[2].plot(J_loss, label="2hJ")
        ax[2].legend()
        plt.suptitle("Loss")
        plt.show()
        
    return x_k.detach()


def SWGF(x0, tau, n_step, n_epochs, J, emd1D, sliced_wasserstein,
         lr=1e-5, num_projections=100, device=device, plot=False, 
         plot_train=False, true_mu=None, true_sigma2=None):
    """
        Inputs:
        - x: samples
        - tau: step size
        - n_step: number of t steps
        - n_epochs: number of epochs for the optimization
        - delta: volume of each cell of the grid
        - J: functional (takes (x,rho) as inputs)
        - lr: learning rate for optimization
        - num_projections
        - device
    """
    Lx = [deepcopy(x0)]
        
    pbar = trange(n_step)

    x = deepcopy(x0)
    
    for k in pbar:
        if isinstance(n_epochs, np.ndarray):
            x = one_step_flow(n_epochs[k], Lx[-1], J, tau, 
                                num_projections, lr, device, 
                                plot_train, emd1D, sliced_wasserstein)
        else:
            x = one_step_flow(n_epochs, Lx[-1], J, tau, 
                                num_projections, lr, device, 
                                plot_train, emd1D, sliced_wasserstein)
        
        Lx.append(x.data)

        if plot:
            t = (k+1)*tau

            plt.scatter(x.detach().cpu()[:,0], x.detach().cpu()[:,1])
            # plt.legend()
            plt.title("t="+str(t))
            plt.show()

    return Lx
