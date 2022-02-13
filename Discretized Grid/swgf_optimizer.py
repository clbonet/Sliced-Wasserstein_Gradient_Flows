import torch
import matplotlib.pyplot as plt
import numpy as np

from copy import deepcopy
from tqdm.auto import trange
from scipy.stats import norm


device = "cuda" if torch.cuda.is_available() else "cpu"


def projection_simplex(v, z=1):
    n_features = v.size(0)
    u,_ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u,0) - z
    ind = torch.arange(n_features).to(device=device) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = torch.clamp(v - theta, min=0)
    return w


def one_step_flow(n_epochs, x_k, x_prev, rho_prev, J, h, num_projections, lr, 
                  delta, device, plot_train, emd1D, sliced_wasserstein):
    """
        Perform gradient descent at time step t
        
        Inputs:
        - n_epochs
        - x_k: actual sample
        - x_prev: previous samples (here, discretized grid, so x_prev=x_k)
        - rho_prev: previous weighs
        - J: functional (taking (x,rho) as inputs)
        - h: time step
        - num_projections
        - lr
        - delta: volume of each cell of the grid
        - device
        
        Outputs:
        - rho_{k+1}^h
    """
    
    n = x_k.size(0)
    d = x_k.size(1)

    rho_k = rho_prev.detach().clone().requires_grad_(True)
    optimizer = torch.optim.SGD([rho_k], lr=lr, momentum=0.9)
    
    train_loss = []
    w_loss = []
    J_loss = []

    L = []
    Lrho = [rho_k.detach().clone()]

    for j in range(n_epochs):
        L.append(deepcopy(rho_k.clone().detach()))
        if d>1:
            sw = sliced_wasserstein(x_k, x_prev, num_projections, device, 
                                    u_weights=rho_k, v_weights=rho_prev, p=2)
        else:
            sw = emd1D(x_k.reshape(1,-1), x_prev.reshape(1,-1), rho_k, rho_prev, p=2)            
        
        f = J(x_k, rho_k, delta)
        loss = sw+2*h*f

        loss.backward()
        optimizer.step()

        rho_k.data = projection_simplex(rho_k.data)
        optimizer.zero_grad()
        
        train_loss.append(loss.item())
        w_loss.append(sw.item())
        J_loss.append(2*h*f.item())
        
        Lrho.append(rho_k.detach().clone())
        
            
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
        
    return rho_k.detach()


def SWGF(x, rho_0, tau, n_step, n_epochs, delta, J, emd1D, sliced_wasserstein,
         lr=1e-5, num_projections=100, device=device, plot_densities=False, 
         plot_train=False, true_mu=None, true_sigma2=None):
    """
        Inputs:
        - x: samples
        - rho_0
        - tau: step size
        - n_step: number of t steps
        - n_epochs: number of epochs for the optimization
        - delta: volume of each cell of the grid
        - J: functional (takes (x,rho) as inputs)
        - lr: learning rate for optimization
        - num_projections
        - device
    """

    rho_k = rho_0/torch.sum(rho_0)
    Lrho = [deepcopy(rho_k)]
        
    pbar = trange(n_step)
    
    for k in pbar:
        if isinstance(n_epochs, np.ndarray):
            rho_k = one_step_flow(n_epochs[k], x, x, Lrho[-1], J, tau, 
                                num_projections, lr, delta, device, 
                                plot_train, emd1D, sliced_wasserstein)
        else:
            rho_k = one_step_flow(n_epochs, x, x, Lrho[-1], J, tau, 
                                num_projections, lr, delta, device, 
                                plot_train, emd1D, sliced_wasserstein)
        
        Lrho.append(rho_k.data)

        if plot_densities:
            t = (k+1)*tau
            y = norm.pdf(x.detach().cpu().numpy(), true_mu(t), np.sqrt(true_sigma2(t)))

            plt.plot(x.detach().cpu().numpy(), Lrho[-1].detach().cpu().numpy(), label="Approximation")
            plt.plot(x.detach().cpu().numpy(), y/np.sum(y), label="True Density")

            plt.legend()
            plt.title("t="+str(t))
            plt.show()

    return Lrho