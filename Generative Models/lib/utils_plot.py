import torch
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import torch.distributions as D

from scipy.stats import multivariate_normal
from copy import deepcopy


def plot_true_density(distr, device, xmin=-4, xmax=8, ymin=-4, ymax=8):
    xline = np.linspace(xmin, xmax, 100)
    yline = np.linspace(ymin, ymax, 100)
    xgrid, ygrid = np.meshgrid(xline, yline)
    xyinput = np.concatenate([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)],axis=1)

    zz = distr.log_prob(torch.tensor(xyinput, dtype=torch.float32, device=device)).exp().detach().cpu()
    zgrid = zz.reshape(100, 100)

    plt.scatter(xyinput[:,0], xyinput[:,1], c=zz, cmap="jet")
    plt.colorbar()
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.show()


def plot_true_density_gaussians(true_mu, true_Sigma, ts=[0,0.5,1,2,3,4], xmin=-4, xmax=8, ymin=-4, ymax=8):
    fig, ax = plt.subplots(2,3,figsize=(15,7))
    xline = np.linspace(xmin, xmax, 100)
    yline = np.linspace(ymin, ymax, 100)
    xgrid, ygrid = np.meshgrid(xline, yline)
    xyinput = np.concatenate([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)],axis=1)

    for ind,t in enumerate(ts):
        i = ind//3
        j = ind%3
        
        m = true_mu(t)
        s = true_Sigma(t)
        density = multivariate_normal.pdf(xyinput,m,s)

        cb = ax[i,j].scatter(xyinput[:,0], xyinput[:,1], c=density, cmap="jet")
        ax[i,j].set_ylim(ymin,ymax)
        ax[i,j].set_xlim(xmin,xmax)
        fig.colorbar(cb, ax=ax[i,j])
        ax[i,j].set_title("t="+str(t))

    plt.show()
    

def plot_true_density_gaussians_contourf(true_mu, true_Sigma, xmin=-4, xmax=8, ymin=-4, ymax=8):
	fig, ax = plt.subplots(2,3,figsize=(15,7))

	xline = np.linspace(xmin, xmax, 100)
	yline = np.linspace(ymin, ymax, 100)
	xgrid, ygrid = np.meshgrid(xline, yline)
	xyinput = np.concatenate([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)],axis=1)

	for ind,t in enumerate([0,0.5,1,2,3,4]):
		i = ind//3
		j = ind%3

		zz = multivariate_normal.pdf(xyinput,true_mu(t),true_Sigma(t))
		zgrid = zz.reshape(100,100)

		cb = ax[i,j].contourf(xgrid, ygrid, zgrid,cmap="jet")
		fig.colorbar(cb, ax=ax[i,j])
		ax[i,j].set_title("t="+str(t))
		ax[i,j].set_xlim(xmin, xmax)
		ax[i,j].set_ylim(ymin, ymax)

	plt.show()


def plot_landscape1d(flows, rho_0, loss, J, h, k, d, 
                     normalized=False, alpha0=-1, alpha1=2):
    """
        flows: list of 2 nn
    """
    alphas = np.linspace(alpha0,alpha1,100)

    nf = deepcopy(flows[0])
    score_line = []

    for i in range(100):
        for name, p in nf.named_parameters():
            new_param = (1-alphas[i])*flows[0].state_dict()[name].data + alphas[i]*flows[1].state_dict()[name].data
            nf.state_dict()[name].copy_(new_param)

        score_line.append(loss(rho_0, nf, J, h, k, d).item())
        
    if normalized:
        score_line = np.array(score_line)
        normalized_score = (score_line-np.min(score_line))
        normalized_score /= np.max(normalized_score)
        
        plt.plot(alphas, normalized_score)
        plt.show()
    else:
        plt.plot(alphas, score_line)
        plt.show()


