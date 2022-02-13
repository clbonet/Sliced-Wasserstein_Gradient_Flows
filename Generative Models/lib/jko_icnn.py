import torch
import gc
import matplotlib.pyplot as plt
import numpy as np

from copy import deepcopy
from tqdm.auto import trange
from logdet_estimators import stochastic_logdet_gradient_estimator
from icnn.utils_icnn import hessian_ICNN, reverse


def one_step_flow(n_epochs, Lrho, J, create_NN, d, h, w, lr,
                  n_samples, device, k, plot_loss):
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
    if k==0:
        rho_k = create_NN(w=w, d=d, device=device)
    else:
        rho_k = deepcopy(Lrho[-1])

    optimizer = torch.optim.Adam(rho_k.parameters(), lr=lr)

    train_loss = []
    J_loss = []
    w_loss = []

    for j in range(n_epochs):
        x = samples_from_rho_prev(Lrho, n_samples)
        
        ## gradient
        y = rho_k(x)
        nabla_ICNN = torch.autograd.grad(y.sum(), x, retain_graph=True, create_graph=True)[0]

        w = torch.mean(torch.sum(torch.square(x.detach()-nabla_ICNN),dim=1))/(2*h)
        f = J(x, nabla_ICNN)

        loss = w + f        
        loss.backward()         
        optimizer.step()
        optimizer.zero_grad()
        
        rho_k.convexify() # clamp weights to be >=0
        
        train_loss.append(loss.item())
        J_loss.append(f.item())
        w_loss.append(w.item())
        
        if torch.isnan(loss) or torch.isnan(f) or torch.isnan(w):
            print(loss, f, w)
            print(x)
            print(y)
            return

    if plot_loss:
        fig, ax = plt.subplots(1,3,figsize=(15,7))
        ax[0].plot(train_loss,label="loss")
        ax[0].legend()
        ax[1].plot(J_loss,label="J")
        ax[1].legend()
        ax[2].plot(w_loss,label="w/(2h)")
        ax[2].legend()
        plt.show()

    return rho_k


def WGF_ICNN(rho_0, tau, n_step, n_epochs, J, create_NN, d=2, w=256, lrs=1e-5, 
             n_samples=500, device="cpu", plot_loss=False, tqdm_bar=False):
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

    Lrho = [rho_0]
    
    if tqdm_bar:    
        pbar = trange(n_step)
    else:
        pbar = range(n_step)
    
    for k in pbar:
        if isinstance(n_epochs, np.ndarray):
            n_epoch = n_epochs[k]
        else:
            n_epoch = n_epochs

        if isinstance(lrs, np.ndarray):
            lr = lrs[k]
        else:
            lr = lrs

        rho_k = one_step_flow(n_epoch, Lrho, J, create_NN, d, tau, w, lr, 
                              n_samples, device, k, plot_loss)
        
        Lrho.append(rho_k)
    
    return Lrho


def samples_from_rho_prev(Lrho, n_samples):
    zk = Lrho[0].sample((n_samples,))
    zk.requires_grad_(True)
    for k in range(1,len(Lrho)):        
        y = Lrho[k](zk)
        zk = torch.autograd.grad(y.sum(), zk, retain_graph=True, create_graph=True)[0]
    return zk

def samples_from_rho_prev_density(Lrho, n_samples):
    zk = Lrho[0].sample((n_samples,))
    log_prob = Lrho[0].log_prob(zk)
    zk.requires_grad_(True)
    
    for k in range(1, len(Lrho)):
        Lrho[k].eval()
        y = Lrho[k](zk)
        grad_icnn = torch.autograd.grad(y.sum(), zk, retain_graph=True, create_graph=True)[0]
        hessian_icnn = hessian_ICNN(zk, grad_icnn, Lrho[k].training)
        log_prob -= torch.log(torch.abs(torch.det(hessian_icnn)))
        zk = grad_icnn
        
    return zk.detach(), log_prob.detach()


def log_likelihood(x, Lrho):
    ## invert
    xs = [x]
    for k in range(len(Lrho)-1,0,-1):
        x = reverse(Lrho[k], x)
        xs.append(x.clone())

    ## compute log prob by the change of variable formula
    xs = xs[::-1]
    log_prob = Lrho[0].log_prob(xs[0])    
    
    for k in range(1, len(Lrho)):
        zk = xs[k-1].clone()
        zk.requires_grad_(True)
        Lrho[k].eval()
        y = Lrho[k](zk)        
        grad_icnn = torch.autograd.grad(y.sum(), zk, retain_graph=True, create_graph=True)[0]

        hessian_icnn = hessian_ICNN(zk, grad_icnn, Lrho[k].training).detach()
        log_prob -= torch.log(torch.abs(torch.det(hessian_icnn)))

    return log_prob.detach()


def sample_rademacher(*shape):
    return (torch.rand(*shape) > 0.5).float() * 2 - 1


def log_det_Hessian_approx(x, nabla_icnn, training, device):
    n_samples, *dims = x.shape
    v = sample_rademacher(n_samples, np.prod(dims)).to(device)
    
    def hvp_fun(v):
        # v is (bsz, dim)
        v = v.reshape(n_samples, *dims)
        hvp = torch.autograd.grad(nabla_icnn, x, v, create_graph=training, retain_graph=True)[0]

        if not torch.isnan(v).any() and torch.isnan(hvp).any():
            raise ArithmeticError("v has no nans but hvp has nans.")
        hvp = hvp.reshape(n_samples, np.prod(dims))
        
        return hvp
    
    est = stochastic_logdet_gradient_estimator(hvp_fun, v, np.prod(dims))
    return est
    
    
def symKL(L, gm, n_samples=10000, device="cuda"):
    x1 = gm.sample((n_samples,))
    density_true_x1 = gm.log_prob(x1)
    density_rho_x1 = log_likelihood(x1, L)

    x2, density_rho_x2 = samples_from_rho_prev_density(L, n_samples)
    density_true_x2 = gm.log_prob(x2)
    
    kl1 = torch.mean(density_true_x1-density_rho_x1)
    kl2 = torch.mean(density_rho_x2-density_true_x2)
    
    return (kl1+kl2).detach().cpu()
    
    
