import torch
import matplotlib.pyplot as plt

from copy import deepcopy
from tqdm.auto import trange

from sw import *


def one_step_flow(n_epochs, rho_prev, J, create_NF, d, h, 
                  num_projections, nh, nl, lr, n_samples, device, k,
                  plot_loss, reset_NF):
    """
        Perform gradient descent at time step t
        
        Inputs:
        - n_epochs
        - rho_prev: previous NF
        - J: functional (taking (x,rho) as inputs)
        - h: time step
        - num_projections
        - nh: number of hidden units
        - nl: number of layers
        - lr
        - device
        - k: step
        
        Outputs:
        - rho_{k+1}^h
    """

    if k>0 and not reset_NF: ## check if it is a NF
        rho_k = deepcopy(rho_prev)
    else:
        rho_k = create_NF(d, nl, nh).to(device)

    optimizer = torch.optim.Adam(rho_k.parameters(), lr=lr)
    
    train_loss = []
    sw_loss = []
    J_loss = []

    for j in range(n_epochs):
        x_k = rho_k.sample(n_samples)

        if k>0:
            x_prev = rho_prev.sample(n_samples)
        else:
            x_prev = rho_prev.sample((n_samples,))

        sw = sliced_wasserstein(x_k, x_prev, num_projections, device, 
                                u_weights=None, v_weights=None, p=2)
        
        f = 2*h*J(x_k,rho_k)
        loss = sw+f
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        train_loss.append(loss.item())
        sw_loss.append(sw.item())
        J_loss.append(f.item())
        
    if plot_loss:
        fig, ax = plt.subplots(1,3,figsize=(15,5))
        ax[0].plot(range(len(train_loss)),train_loss, label="Loss")
        L = range(10, len(train_loss))
        moving_average = []
        for i in range(len(L)):
            moving_average.append(np.mean(train_loss[i:i+10]))
        ax[0].plot(L, moving_average, label="Moving Average")
        ax[0].set_title("Full loss")
        ax[0].legend()
        
        ax[1].plot(sw_loss)
        moving_average = []
        for i in range(len(L)):
            moving_average.append(np.mean(sw_loss[i:i+10]))
        ax[1].plot(L, moving_average)
        ax[1].set_title("SW")
        
        ax[2].plot(J_loss)
        moving_average = []
        for i in range(len(L)):
            moving_average.append(np.mean(J_loss[i:i+10]))
        ax[2].plot(L, moving_average)
        ax[2].set_title("2hJ")
        
        plt.suptitle("k="+str(k))
        plt.show()


    return rho_k


def SWGF(rho_0, tau, n_step, n_epochs, J, create_NF, d=2, nh=64, nl=5, lrs=1e-5, 
         num_projections=100, n_samples=500, device=device, plot_loss=False, reset_NF=False):
    """
        Inputs:
        - rho_0
        - tau: step size
        - n_step: number of t steps
        - n_epochs: number of epochs for the optimization
        - J: functional (takes (x,rho) as inputs)
        - nh: number of hidden units
        - nl: number of layers
        - lr: learning rate for optimization
        - num_projections
        - device
    """

    Lrho = [rho_0] ## For rho_0, distribution class?
        
    pbar = trange(n_step)
    
    for k in pbar:
        if isinstance(n_epochs, np.ndarray):
            n_epoch = n_epochs[k].astype(int)
        else:
            n_epoch = n_epochs

        if isinstance(lrs, np.ndarray):
            lr = lrs[k]
        else:
            lr = lrs

        rho_k = one_step_flow(n_epoch, Lrho[-1], J, create_NF, d, tau, 
                            num_projections, nh, nl, lr, n_samples, device, k,
                            plot_loss, reset_NF)
        
        Lrho.append(rho_k)

    return Lrho
