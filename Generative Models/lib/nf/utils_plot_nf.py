import torch
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import torch.distributions as D

from scipy.stats import multivariate_normal
from copy import deepcopy

from .utils_nf import log_likelihood, symKL


def plot_density_contourf(rho, device, xmin, xmax, ymin, ymax, title=None):
    xline = torch.linspace(xmin, xmax, 100)
    yline = torch.linspace(ymin, ymax, 100)
    xgrid, ygrid = torch.meshgrid(xline, yline)
    xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)
    
    ## Check if CPF Flow 
    cpf = False
    for flow in rho.flows:
        if flow.__class__.__name__ == "ConvexPotentialFlow":
            cpf = True
            break
            
    rho.eval()
    z_xy, log_s = rho.backward(xyinput.to(device))
    
    if cpf: ## det not available for the inverse
        x, log_det = rho(z_xy[-1])
        zz = (log_likelihood(z_xy[-1],log_det,device)).exp().cpu().detach() 
    else:
        zz = (log_likelihood(z_xy[-1],-log_s,device)).exp().cpu().detach() ## - because I compute likelihood wrt z and log_s is log det J_{T^{-1}}(x)
        
    zgrid = zz.reshape(100,100)

    plt.contourf(xgrid.numpy(), ygrid.numpy(), zgrid.numpy(),cmap="jet")
    plt.colorbar()
    if title is not None:
        plt.title(title)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.show()


def plot_density(rho, device, xmin, xmax, ymin, ymax, title=None):
    xline = torch.linspace(xmin, xmax, 100)
    yline = torch.linspace(ymin, ymax, 100)
    xgrid, ygrid = torch.meshgrid(xline, yline)
    xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

    ## Check if CPF Flow 
    cpf = False
    for flow in rho.flows:
        if flow.__class__.__name__ == "ConvexPotentialFlow":
            cpf = True
            break

    rho.eval()
    z_xy, log_s = rho.backward(xyinput.to(device))
    
    if cpf: ## det not available for the inverse
        x, log_det = rho(z_xy[-1])
        zz = (log_likelihood(z_xy[-1],log_det,device)).exp().cpu().detach() 
    else:
        zz = (log_likelihood(z_xy[-1],-log_s,device)).exp().cpu().detach() ## - because I compute likelihood wrt z and log_s is log det J_{T^{-1}}(x)
     
    zgrid = zz.reshape(100,100)

    plt.scatter(xgrid.numpy(), ygrid.numpy(), c=zgrid.numpy(),cmap="jet")
    plt.colorbar()
    if title is not None:
        plt.title(title)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.show()


def plot_density_ts_contourf(Lrho, device, ts=[0,0.5,1,2,3,4], xmin=-4, xmax=8, ymin=-4, ymax=8, tend=4):
    fig, ax = plt.subplots(2,3,figsize=(15,7))

    xline = torch.linspace(xmin, xmax, 100)
    yline = torch.linspace(ymin, ymax, 100)
    xgrid, ygrid = torch.meshgrid(xline, yline)
    xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

    for ind,t in enumerate(ts):
        i = ind//3
        j = ind%3
        
        k = int(t*(len(Lrho)-1)/tend)

        if k>0:
            ## Check if CPF Flow 
            cpf = False
            for flow in Lrho[k].flows:
                if flow.__class__.__name__ == "ConvexPotentialFlow":
                    cpf = True
                    break
            
            Lrho[k].eval()
            z_xy, log_s = Lrho[k].backward(xyinput.to(device))
    
            if cpf: ## det not available for the inverse
                x, log_det = Lrho[k](z_xy[-1])
                zz = (log_likelihood(z_xy[-1],log_det,device)).exp().cpu().detach() 
            else:
                zz = (log_likelihood(z_xy[-1],-log_s,device)).exp().cpu().detach() ## - because I compute likelihood wrt z and log_s is log det J_{T^{-1}}(x)
     
        else:
            zz = Lrho[0].log_prob(xyinput.to(device)).exp().cpu().detach()

        zgrid = zz.reshape(100,100)

        cb = ax[i,j].contourf(xgrid.numpy(), ygrid.numpy(), zgrid.numpy(),cmap="jet")
        fig.colorbar(cb, ax=ax[i,j])
        ax[i,j].set_title("t="+str(t))
        ax[i,j].set_xlim(xmin, xmax)
        ax[i,j].set_ylim(ymin, ymax)
        
    plt.show()
    
    
def plot_density_ts(Lrho, device, ts=[0,0.5,1,2,3,4],
                     xmin=-4, xmax=8, ymin=-4, ymax=8, tend=4):
    fig, ax = plt.subplots(2,3,figsize=(15,7))

    xline = torch.linspace(xmin, xmax, 100)
    yline = torch.linspace(ymin, ymax, 100)
    xgrid, ygrid = torch.meshgrid(xline, yline)
    xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

    for ind,t in enumerate(ts):
        i = ind//3
        j = ind%3
        
        k = int(t*(len(Lrho)-1)/tend)

        if k>0:
            ## Check if CPF Flow 
            cpf = False
            for flow in Lrho[k].flows:
                if flow.__class__.__name__ == "ConvexPotentialFlow":
                    cpf = True
                    break
                    
            Lrho[k].eval()
            z_xy, log_s = Lrho[k].backward(xyinput.to(device))
    
            if cpf: ## det not available for the inverse
                x, log_det = Lrho[k](z_xy[-1])
                zz = (log_likelihood(z_xy[-1],log_det,device)).exp().cpu().detach() 
            else:
                zz = (log_likelihood(z_xy[-1],-log_s,device)).exp().cpu().detach() ## - because I compute likelihood wrt z and log_s is log det J_{T^{-1}}(x)
     
        else:
            zz = Lrho[0].log_prob(xyinput.to(device)).exp().cpu().detach()
            
        zgrid = zz.reshape(100,100)

        cb = ax[i,j].scatter(xgrid.numpy(), ygrid.numpy(), c=zgrid.numpy(),cmap="jet")
        fig.colorbar(cb, ax=ax[i,j])
        ax[i,j].set_title("t="+str(t))
        ax[i,j].set_xlim(xmin, xmax)
        ax[i,j].set_ylim(ymin, ymax)
        
    plt.show()


def plot_F(Lrho, J, V, device, distr_stationary=None, d=2, n_samples=10000, t_init=0, t_end=4):
    L = []

    for i in range(len(Lrho)):
        if i>0:
            Lrho[i].eval()
            z = torch.randn(n_samples, d, device=device)
            x, log_det = Lrho[i](z)
            L.append(J(x[-1], z, log_det).item())
        else:
            x = Lrho[0].sample((n_samples,))
            L.append(torch.mean(V(x), axis=0)+torch.mean(Lrho[0].log_prob(x),axis=0))
    
    if distr_stationary is not None:
        x_s = distr_stationary.sample((n_samples,))
        log_prob_xs = distr_stationary.log_prob(x_s)
        F_stationary = torch.mean(V(x_s), axis=0)+torch.mean(log_prob_xs, axis=0)
        F_stationary = F_stationary.detach().cpu().numpy()


    absc = np.linspace(t_init, t_end, len(L))
    plt.plot(absc, L, label=r"$F(\hat{\rho}_t)$")
    if distr_stationary is not None:
        plt.plot(absc, F_stationary*np.ones(len(absc)), '--', label=r"$F(\rho^*)$")
    plt.xlabel("t")
    plt.legend()
    plt.show()


def plot_F_gaussians(Lrho, J, V, true_mu, true_sigma, mu_stationary, sigma_stationary,
                     device, d=2, n_samples=10000, t_init=0, t_end=4, dilation=1):
    L = []
    L_true = []

    absc = np.linspace(t_init, t_end, len(Lrho))

    for i in range(len(Lrho)):
        t = absc[i]*dilation

        if i>0:
            Lrho[i].eval()
            z = torch.randn(n_samples, d, device=device)
            x, log_det = Lrho[i](z)
            L.append(J(x[-1], z, log_det).item())
        else:
            x = Lrho[0].sample((n_samples,))
            L.append(torch.mean(V(x), axis=0)+torch.mean(Lrho[0].log_prob(x),axis=0))
        
        mu_t = true_mu(t)
        Sigma_t = true_sigma(t)
        x_t = multivariate_normal.rvs(mu_t, Sigma_t, n_samples)
        log_prob = multivariate_normal.logpdf(x_t, mu_t, Sigma_t)
        x_t = torch.tensor(x_t, device=device, dtype=torch.float)
        log_prob = torch.tensor(log_prob, device=device, dtype=torch.float)
        L_true.append(torch.mean(V(x_t), axis=0)+torch.mean(log_prob, axis=0))

    x_s = multivariate_normal.rvs(mu_stationary, sigma_stationary, n_samples)
    log_prob_s = multivariate_normal.logpdf(x_s, mu_stationary, sigma_stationary)
    x_s = torch.tensor(x_s, device=device, dtype=torch.float)
    log_prob_s = torch.tensor(log_prob_s, device=device, dtype=torch.float)
    F_stationary = torch.mean(V(x_s), axis=0)+torch.mean(log_prob_s, axis=0)

    if dilation!=1:
        plt.plot(absc, L_true, label=r"$F(\rho_{"+str(dilation)+r"t})$")
    else:
        plt.plot(absc, L_true, label=r"$F(\rho_t)$")
    plt.plot(absc, L, label=r"$F(\hat{\rho}_{t})$")
    plt.plot(absc, np.ones(len(absc))*F_stationary.cpu().numpy(), '--', label=r"$F(\rho^*)$")
    plt.xlabel("t")
    plt.legend()
    plt.show()


def plot_mean(Lrho, true_mu, mu_stationary, t_init, t_end, device, d=2, dilation=1):
    L_mu = []
    L = []
    absc = np.linspace(t_init, t_end, len(Lrho))

    for i in range(len(Lrho)):
        L_mu.append(true_mu(absc[i]*dilation))

        if i==0:
            x = Lrho[i].sample((10000,))
            L.append(torch.mean(x, axis=0).cpu().detach().numpy())
        else:
            z = torch.randn(10000, d, device=device)
            x, _ = Lrho[i](z)
            L.append(torch.mean(x[-1],axis=0).cpu().detach().numpy())

    L_mu = np.array(L_mu)
    L = np.array(L)

    fig, ax = plt.subplots(1,2,figsize=(15,7))

    ax[0].plot(absc, L[:,0], label=r"$\mu_{approx}^1(t)$")
    if dilation==1:
        ax[0].plot(absc, L_mu[:,0], label=r"$\mu_{analytical}^1(t)$")
    else:
        ax[0].plot(absc, L_mu[:,0], label=r"$\mu_{analytical}^1("+str(dilation)+r"t)$")
    ax[0].plot(absc, mu_stationary[0]*np.ones(len(absc)), '--', label=r"$\mu^1_*$")
    ax[0].set_xlabel("t")
    ax[0].legend()

    ax[1].plot(absc, L[:,1], label=r"$\mu_{approx}^2(t)$")
    if dilation==1:
        ax[1].plot(absc, L_mu[:,1], label=r"$\mu_{analytical}^2(t)$")
    else:
        ax[1].plot(absc, L_mu[:,1], label=r"$\mu_{analytical}^2("+str(dilation)+r"t)$")
    ax[1].plot(absc, mu_stationary[1]*np.ones(len(absc)), '--', label=r"$\mu^2_*$")
    ax[1].legend()
    ax[1].set_xlabel("t")
    plt.show()


def plot_SymKL(Lrho, distr_stationary, t_init, t_end, h, device, n_samples=10000, log10=False):    
    absc = np.linspace(t_init+h, t_end, len(Lrho)-1)
    L_stationary = []

    for i in range(1,len(Lrho)):
        s = symKL(Lrho[i], distr_stationary, n_samples, device)
        L_stationary.append(s)

    L_stationary = np.array(L_stationary)

    if log10:
        plt.plot(absc, np.log10(L_stationary))
    else:
        plt.plot(absc, L_stationary)
    plt.title(r"SymKL($\rho^*,\hat{\rho}_t$)")
    plt.xlabel("t")
    plt.ylabel(r"$\log_{10}\mathrm{SymKL}$")
    plt.show()        
        
        
def cov(X):
    """
        https://github.com/pytorch/pytorch/issues/19037
    """
    D = X.shape[-1]
    mean = torch.mean(X, dim=-1).unsqueeze(-1)
    X = X - mean
    return 1/(D-1) * X @ X.transpose(-1, -2)


def plot_var(Lrho, true_sigma, sigma_stationary, t_init, t_end, device, d=2, dilation=1):
    L_var = []
    L = []
    absc = np.linspace(t_init, t_end, len(Lrho))

    for i in range(len(Lrho)):
        L_var.append(true_sigma(absc[i]*dilation))

        if i==0:
            x = Lrho[i].sample((10000,))
            L.append(cov(x.T).cpu().detach().numpy())
        else:
            z = torch.randn(10000, d, device=device)
            x, _ = Lrho[i](z)
            L.append(cov(x[-1].T).cpu().detach().numpy())

    L_var = np.array(L_var)
    L = np.array(L)

    fig, ax = plt.subplots(2,2,figsize=(15,7))

    for i in range(2):
        for j in range(2):
            if dilation==1:
                ax[i,j].plot(absc, L_var[:,i,j], label=r"$[\Sigma(t)]_{"+str(i)+str(j)+"}$")
            else:
                ax[i,j].plot(absc, L_var[:,i,j], label=r"$[\Sigma("+str(dilation)+"t)]_{"+str(i)+str(j)+"}$")
            ax[i,j].plot(absc, L[:,i,j], label=r"$[\hat{\Sigma}(t)]_{"+str(i)+str(j)+"}$")
            ax[i,j].plot(absc, sigma_stationary[i,j]*np.ones(len(absc)), '--', label=r"$[\Sigma^*]_{"+str(i)+str(j)+"}$")
            ax[i,j].set_xlabel("t")
    
            ax[i,j].legend()
    plt.show()
