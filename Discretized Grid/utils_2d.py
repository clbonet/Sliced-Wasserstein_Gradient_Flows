import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import torch

from scipy.stats import multivariate_normal


def plot_density(xs, Lrho, xmin=-4, xmax=8, ymin=-4, ymax=8, t1=4):
    fig, ax = plt.subplots(2,3,figsize=(15,7))
    inds = [0,0.5,1,2,3,4]

    for ind,t in enumerate(inds):
        i = ind//3
        j = ind%3
        
        k = int(t*(len(Lrho)-1)/t1)
        cb = ax[i,j].scatter(xs[:,0], xs[:,1], c=Lrho[k],cmap="jet")
        ax[i,j].set_ylim(ymin,ymax)
        ax[i,j].set_xlim(xmin,xmax)
        fig.colorbar(cb, ax=ax[i,j])
        ax[i,j].set_title("t="+str(t))
        
    plt.show()


def plot_true_density(xs, true_mu, true_Sigma, xmin=-4, xmax=8, ymin=-4, ymax=8):
    fig, ax = plt.subplots(2,3,figsize=(15,7))
    inds = [0,0.5,1,2,3,4]

    for ind,t in enumerate(inds):
        i = ind//3
        j = ind%3
        
        m = true_mu(t)
        s = true_Sigma(t)
        density = multivariate_normal.pdf(xs,m,s)
        density /= np.sum(density)
        
        cb = ax[i,j].scatter(xs[:,0], xs[:,1], c=density,cmap="jet")
        ax[i,j].set_ylim(ymin,ymax)
        ax[i,j].set_xlim(xmin,xmax)
        fig.colorbar(cb, ax=ax[i,j])
        ax[i,j].set_title("t="+str(t))

    plt.show()



def W(m1,m2,sigma1,sigma2):
    dm = np.linalg.norm(m1-m2)**2
    sigma1_12 = sp.linalg.sqrtm(sigma1)
    A = sigma1_12@sigma2@sigma1_12
    B = np.trace(sigma1+sigma2)-2*np.trace(sp.linalg.sqrtm(A))
    return dm+B


def plot_W2(xs, Lrho, true_mu, true_Sigma,
            mu_stationary, sigma2_stationary, 
            t0=0, t1=4, log=False):
    absc = np.linspace(t0,t1,len(Lrho))
    
    Ws = []
    Ws2 = []
    for i in range(len(Lrho)):
        s1 = true_Sigma(absc[i])
        s2 = np.cov(xs.T,aweights=Lrho[i])
        mu_x = np.sum(Lrho[i][:,None]*xs,axis=0)
        Ws.append(W(mu_x,true_mu(absc[i]),s1,s2))

        s1 = sigma2_stationary
        Ws2.append(W(mu_x,mu_stationary,s1,s2))
        
    fig, ax = plt.subplots(1,2,figsize=(15,7))
    
    ax[1].plot(absc,Ws2)
    ax[1].set_title(r"$W_2^2(\rho^*,\hat{\rho})$")
    if log:
        ax[1].set_yscale("log")
    
    ax[0].plot(absc,Ws)
    ax[0].set_title(r"$W_2^2(\rho,\hat{\rho})$")

    plt.show()


def get_W2(xs, Lrho, true_mu, true_Sigma,
            mu_stationary, sigma2_stationary, 
            t0=0, t1=4):
    """
        returns the squared Wasserstein distance between the true gaussians, 
        and the estimated gaussians at each time step.
        Also returns it between the distribution at time t and the stationary distribution.
    """
    absc = np.linspace(t0,t1,len(Lrho))
    
    Ws = []
    Ws2 = []
    for i in range(len(Lrho)):
        s1 = true_Sigma(absc[i])
        s2 = np.cov(xs.T,aweights=Lrho[i])
        mu_x = np.sum(Lrho[i][:,None]*xs,axis=0)
        Ws.append(W(mu_x,true_mu(absc[i]),s1,s2))

        s1 = sigma2_stationary
        Ws2.append(W(mu_x,mu_stationary,s1,s2))
    
    return Ws, Ws2


def plot_F(F, V, x, rho, delta, device, distr_stationary=None, d=2, n_samples=10000, 
           t_init=0, t_end=4):
    L = []

    for i in range(len(rho)):
        L.append(F(torch.tensor(x,device=device),
                   torch.tensor(rho[i],device=device),
                   delta).detach().cpu().numpy())
    
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
    
    
def plot_F_gaussians(F,x,rho,delta,true_mu,true_sigma,mu_stationary, 
                     sigma_stationary,V,device,t0=0,t1=4,dilation=1,
                     n_samples=10000):
    L = []
    L_true = []
    
    absc = np.linspace(t0, t1, len(rho))
    
    for i in range(len(rho)):
        L.append(F(torch.tensor(x,device=device),
                   torch.tensor(rho[i],device=device),
                   delta).detach().cpu().numpy())
        
        t = absc[i]*dilation
        
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


def plot_error(xs, Lrho, true_mu, true_Sigma, xmin=-4, xmax=8, ymin=-4, ymax=8, t1=4):
    fig, ax = plt.subplots(2,3,figsize=(15,7))
    inds = [0,0.5,1,2,3,4]

    max_error = 0
    for ind,t in enumerate(inds):
        k = int(t*(len(Lrho)-1)/t1)

        m = true_mu(t)
        s = true_Sigma(t)
        density = multivariate_normal.pdf(xs,m,s)
        density /= np.sum(density)

        cpt = np.max(np.abs(density-Lrho[k]))
        if cpt>max_error:
            max_error = cpt

    for ind,t in enumerate(inds):
        i = ind//3
        j = ind%3
        
        m = true_mu(t)
        s = true_Sigma(t)
        density = multivariate_normal.pdf(xs,m,s)
        density /= np.sum(density)

        k = int(t*(len(Lrho)-1)/t1)
        
        cb = ax[i,j].scatter(xs[:,0], xs[:,1], c=np.abs(density-Lrho[k]),cmap="jet", vmax=max_error)
        ax[i,j].set_ylim(ymin, ymax)
        ax[i,j].set_xlim(xmin, xmax)
        ax[i,j].set_title("t="+str(t))

    fig.colorbar(cb, ax=ax, location="right",shrink=0.75)
    plt.show()
    

def plot_mean(xs, Lrho, true_mu, mu_stationary, d=2, t_init=0, t_end=4, dilation=1):
    absc = np.linspace(t_init,t_end,len(Lrho))
    
    L_mu = []
    L_x = []
    for i in range(len(Lrho)):
        L_mu.append(true_mu(absc[i]*dilation))
        mu_x = np.sum(Lrho[i][:,None]*xs,axis=0)
        L_x.append(mu_x)
        
    L_mu = np.array(L_mu)
    L_x = np.array(L_x)    
    
    fig, ax = plt.subplots(1,2,figsize=(15,7))

    ax[0].plot(absc, L_x[:,0], label=r"$\mu_{approx}^1(t)$")
    if dilation==1:
        ax[0].plot(absc, L_mu[:,0], label=r"$\mu_{analytical}^1(t)$")
    else:
        ax[0].plot(absc, L_mu[:,0], label=r"$\mu_{analytical}^1("+str(dilation)+r"t)$")
    ax[0].plot(absc, mu_stationary[0]*np.ones(len(absc)), '--', label=r"$\mu^1_*$")
    ax[0].set_xlabel("t")
    ax[0].legend()

    ax[1].plot(absc, L_x[:,1], label=r"$\mu_{approx}^2(t)$")
    if dilation==1:
        ax[1].plot(absc, L_mu[:,1], label=r"$\mu_{analytical}^2(t)$")
    else:
        ax[1].plot(absc, L_mu[:,1], label=r"$\mu_{analytical}^2("+str(dilation)+r"t)$")
    ax[1].plot(absc, mu_stationary[1]*np.ones(len(absc)), '--', label=r"$\mu^2_*$")
    ax[1].legend()
    ax[1].set_xlabel("t")
    plt.show()
    
    
    
def cov2(X, prob):
    mean = np.sum(X*prob[:,None], axis=0) # torch.mean(X, dim=-1).unsqueeze(-1)
    X = (X - mean[None,:])
    xx = np.matmul(X.reshape(-1,2,1), X.reshape(-1,1,2))
    var = np.sum(xx * prob.reshape(-1,1,1), axis=0)
    return var


def plot_var(xs, Lrho, true_sigma, sigma_stationary, d=2, t_init=0, t_end=4, dilation=1):
    absc = np.linspace(t_init,t_end,len(Lrho))

    L_var = []
    L = []

    for i in range(len(Lrho)):
        L_var.append(true_sigma(absc[i]*dilation))
        C = cov2(xs, Lrho[i])
        L.append(C)

    L_var = np.array(L_var)
    L = np.array(L)

    fig, ax = plt.subplots(2,2,figsize=(15,7))

    for i in range(2):
        for j in range(2):
            ax[i,j].plot(absc, L_var[:,i,j])
            if dilation==1:
                ax[i,j].plot(absc, L[:,i,j])
            else:
                ax[i,j].plot(absc, L[:,i,j])
            ax[i,j].plot(absc, sigma_stationary[i,j]*np.ones(len(absc)), '--')
            ax[i,j].set_xlabel("t")

    plt.show()
