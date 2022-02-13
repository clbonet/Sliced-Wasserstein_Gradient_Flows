import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
from jko_icnn import *


def plot_mean(Lrho, true_mu, mu_stationary, t_init, t_end, device):
    L_mu = []
    L = []
    absc = np.linspace(t_init ,t_end, len(Lrho))
    
    n_samples = 1000

    for i in range(len(Lrho)):
        L_mu.append(true_mu(absc[i]))

        x = samples_from_rho_prev(Lrho[:i+1], n_samples).detach().cpu().numpy()
        L.append(np.mean(x, axis=0))

    L_mu = np.array(L_mu)
    L = np.array(L)

    fig, ax = plt.subplots(1,2,figsize=(15,7))

    ax[0].plot(absc, L_mu[:,0], label=r"$\mu_{analytical}^1$")
    ax[0].plot(absc, L[:,0], label=r"$\mu_{approx}^1$")
    ax[0].plot(absc, mu_stationary[0]*np.ones(len(absc)), '--', label=r"$\mu^1_*$")
    ax[0].set_xlabel("t")
    ax[0].legend()

    ax[1].plot(absc, L_mu[:,1], label=r"$\mu_{analytical}^2$")
    ax[1].plot(absc, L[:,1], label=r"$\mu_{approx}^2$")
    ax[1].plot(absc, mu_stationary[1]*np.ones(len(absc)), '--', label=r"$\mu^2_*$")
    ax[1].legend()
    ax[1].set_xlabel("t")
    plt.show()


def plot_F(Lrho, J, V, true_mu, true_sigma, mu_stationary, sigma_stationary,
           device, d=2, n_samples=1000, t_init=0, t_end=4):
    L = []
    L_true = []
    
    absc = np.linspace(t_init, t_end, len(Lrho))

    for i in range(len(Lrho)):
        t = absc[i]

        L.append(J(Lrho[:i+1], n_samples).cpu())

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

    plt.plot(absc, L, label=r"$F(\hat{\rho}_t)$")
    plt.plot(absc, L_true, label=r"$F(\rho_t)$")
    plt.plot(absc, np.ones(len(absc))*F_stationary.cpu().numpy(), '--', label=r"$F(\rho^*)$")
    plt.xlabel("t")
    plt.legend()
    plt.show()
    

def plot_density(Lrho, ts, xmin, xmax, ymin, ymax, t_end, device):
    fig, ax = plt.subplots(2,3,figsize=(15,7))

    xline = torch.linspace(xmin, xmax, 100)
    yline = torch.linspace(ymin, ymax, 100)
    xgrid, ygrid = torch.meshgrid(xline, yline)
    xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

    for ind,t in enumerate(ts):
        i = ind//3
        j = ind%3

        k = int(t*(len(Lrho)-1)/t_end)

        zz = log_likelihood(xyinput.to(device), Lrho[:k+1]).exp().detach()
        zgrid = zz.reshape(100,100).cpu()
        
        cb = ax[i,j].scatter(xgrid.numpy(), ygrid.numpy(), c=zgrid.numpy(),cmap="jet")
        fig.colorbar(cb, ax=ax[i,j])
        ax[i,j].set_title("t="+str(t))
        ax[i,j].set_xlim(xmin, xmax)
        ax[i,j].set_ylim(ymin, ymax)

    plt.show()


def plot_SymKL(Lrho, true_mu, true_Sigma, mu_stationary, Sigma_stationary, 
            t_init, t_end, device, log=False, log10=False):
    absc = np.linspace(t_init, t_end, len(Lrho))
    
    n_samples = 5000
    
    L_t = []
    L_stationary = []
    
    for i in range(1,len(Lrho)):
        t = absc[i]
#         print(i)
#         print(torch.cuda.memory_summary())
        mu_t = true_mu(t)
        Sigma_t = true_Sigma(t)
        
        x1 = multivariate_normal.rvs(mu_t, Sigma_t, n_samples)
        density_t_x1 = multivariate_normal.pdf(x1, mu_t, Sigma_t)
        density_rho_x1 = log_likelihood(torch.tensor(x1,dtype=torch.float,device=device),
                                        Lrho[:i+1]).detach().cpu().numpy()

        x2, density_rho_x2 = samples_from_rho_prev_density(Lrho[:i+1], n_samples)
        x2 = x2.detach().cpu().numpy()
        density_rho_x2 = density_rho_x2.detach().cpu().numpy()
        density_t_x2 = multivariate_normal.pdf(x2, mu_t, Sigma_t)

        x1s = multivariate_normal.rvs(mu_stationary, Sigma_stationary, n_samples)
        density_s_x1s = multivariate_normal.pdf(x1s, mu_stationary, Sigma_stationary)
        density_rho_x1s = log_likelihood(torch.tensor(x1s,dtype=torch.float,device=device),
                                         Lrho[:i+1]).detach().cpu().numpy()
                             
        density_s_x2 = multivariate_normal.pdf(x2, mu_stationary, Sigma_stationary)
        
        klt1 = np.mean(np.log(density_t_x1)-density_rho_x1)
        klt2 = np.mean(density_rho_x2-np.log(density_t_x2))
        L_t.append(klt1+klt2)
        
        kls1 = np.mean(np.log(density_s_x1s)-density_rho_x1s)
        kls2 = np.mean(density_rho_x2-np.log(density_s_x2))
        L_stationary.append(kls1+kls2)

    if not log10:
        fig, ax = plt.subplots(1,2,figsize=(15,7))
        ax[0].plot(absc[1:], L_t)
        ax[0].set_xlabel("t")
        ax[0].set_title(r"SymKL($\rho_t,\hat{\rho}_t$)")
        
        ax[1].plot(absc[1:], L_stationary)
        ax[1].set_xlabel("t")
        ax[1].set_title(r"SymKL($\rho^*,\hat{\rho}_t$)")

        if log:
            # ax[0].set_yscale("log")
            ax[1].set_yscale("log")

        plt.show()

    else:
        L_t = np.array(L_t)
        L_stationary = np.array(L_stationary)           
        
        fig, ax = plt.subplots(1,2,figsize=(15,7))
        ax[0].plot(absc[1:], np.log10(L_t))
        ax[0].set_xlabel("t")
        ax[0].set_ylabel(r"$\mathrm{log}_{10}$SymKL")
        ax[0].set_title(r"SymKL($\rho_t,\hat{\rho}_t$)")
        
        ax[1].plot(absc[1:], np.log10(L_stationary))
        ax[1].set_xlabel("t")
        ax[1].set_ylabel(r"$\mathrm{log}_{10}$SymKL")
        ax[1].set_title(r"SymKL($\rho^*,\hat{\rho}_t$)")
        plt.show()
