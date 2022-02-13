import torch
import matplotlib.pyplot as plt

from copy import deepcopy
from tqdm.auto import trange

from sw import *


def one_step_flow(n_epochs, rho_prev, J, create_NF, d, h, 
                  num_projections, nh, nl, lr, n_samples, device, 
                  k, plot_loss, sw_approx, max_sliced, distributional_sliced,
                  reset_NF, use_scheduler):
    """
        Perform gradient descent at time step t
        
        Inputs:
        - n_epochs
        - rho_prev: previous NF
        - J: functional (taking (x,z,log(det(J(z)))) as inputs)
        - h: time step
        - num_projections
        - nh: number of hidden units
        - nl: number of layers
        - lr
        - device
        - k: step
        - plot_loss
        - sw_approx: use the concentration approximation
        - max_sliced: if True, use max SW
        - reset_NF: If True, start from a random initialized NF
        - use_scheduler: If True, use ReduceLROnPlateau Scheduler
        
        Outputs:
        - rho_{k+1}^h
    """

    if k>0 and not reset_NF: ## check if it is a NF
#        rho_k = deepcopy(rho_prev)
        rho_k = create_NF(nh, nl, d=d).to(device)
        rho_k.load_state_dict(deepcopy(rho_prev.state_dict()))
    else:
        rho_k = create_NF(nh, nl, d=d).to(device)

    optimizer = torch.optim.Adam(rho_k.parameters(), lr=lr)
    optimizer.zero_grad()
    
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
        
    
    train_loss = []
    sw_loss = []
    J_loss = []

    for j in range(n_epochs):
        z_k = torch.randn(n_samples, d, device=device)
        x_k, log_det_k = rho_k(z_k)
        x_k = x_k[-1]

        if k>0:
            z0 = torch.randn(n_samples, d, device=device)
            x_prev, log_det_prev = rho_prev(z0)
            x_prev = x_prev[-1]
        else:
            x_prev = rho_prev.sample((n_samples,))

        if sw_approx:
            sw = sw2_approx(x_k, x_prev, device, u_weights=None, v_weights=None)
        elif max_sliced:
            sw = max_SW(x_k, x_prev, device, p=2, u_weights=None, v_weights=None)
        elif distributional_sliced:
            sw = distributional_sw(x_k, x_prev, num_projections, device, u_weights=None, v_weights=None)
        else:
            sw = sliced_wasserstein(x_k, x_prev, num_projections, device, 
                                    u_weights=None, v_weights=None, p=2)
            
        f = J(x_k, z_k, log_det_k)
        loss = sw+2*h*f
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        for flow in rho_k.flows:
            if flow.__class__.__name__ == "ConvexPotentialFlow":
                flow.icnn.convexify() # clamp weights to be >=0
        
        train_loss.append(loss.item())
        sw_loss.append(sw.item())
        J_loss.append(f.item())
        
        if use_scheduler:
            scheduler.step(f)

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
         num_projections=100, n_samples=500, sw_approx=False, max_sliced=False, distributional_sliced=False, reset_NF=False, 
         device=device, use_scheduler=False, plot_loss=False, tqdm_bar=False):
    """
        Inputs:
        - rho_0
        - tau: step size
        - n_step: number of t steps
        - n_epochs: number of epochs for the optimization (can be a list of size
        n_step or an int)
        - J: functional (takes (x,z,log(det(J(z)))) as inputs)
        - create_NF: function which return a BaseNormalizingFlow class taking
        (nh, nl, d) as inputs
        - nh: number of hidden units
        - nl: number of layers
        - lrs: learning rate for optimization (can be a list of size n_step or an int)
        - num_projections
        - n_samples: batch size
        - sw_approx: If true, use the SW_2^2 approximation of SW (without projections)
        - max_sliced: If True, use max-SW 
        - reset_nn: If True, start from an unitialized flow
        - device
        - use_scheduler: If True, use a ReduceLROnPlateau Scheduler
        - plot_loss (default False)
        - tqdm_bar (default False)
    """

    Lrho = [rho_0] ## For rho_0, distribution class
    
    if tqdm_bar:    
        pbar = trange(n_step)
    else:
        pbar = range(n_step)
    
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
                                num_projections, nh, nl, lr, n_samples, 
                                device, k, plot_loss, sw_approx, max_sliced, distributional_sliced, 
                                reset_NF, use_scheduler)
        
        Lrho.append(rho_k)

    return Lrho
