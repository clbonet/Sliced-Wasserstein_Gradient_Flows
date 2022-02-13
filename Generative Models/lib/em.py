import torch
import torch.distributions as D
import numpy as np


def EulerMaruyama(V, n_particles=1000, d=2, dt=1e-3, t_end=4, device="cpu", init_distr=None):
    n_steps = int(t_end/dt)

    normal = D.MultivariateNormal(torch.zeros(d, device=device),torch.eye(d, device=device))

    if init_distr is None:
        init_distr = D.MultivariateNormal(torch.zeros(d, device=device), torch.eye(d,device=device))

    x0 = init_distr.sample((n_particles,))
    xk = x0.clone()
    for k in range(n_steps):
        xk.requires_grad_(True)
        grad_V = torch.autograd.grad(V(xk).sum(), xk)[0]
        W = normal.sample((n_particles,))
        xk = xk.detach()
        xk += -grad_V*dt+np.sqrt(2*dt)*W
        
    return xk


def symKL_kernel(gm, kernel, n=10000, device="cpu"):
    x1 = gm.sample((n,))
    density_gm_x1 = gm.log_prob(x1).detach().cpu().numpy()
    density_kernel_x1 = kernel.logpdf(x1.T.detach().cpu())

    x2 = kernel.resample((n,))
    density_gm_x2 = gm.log_prob(torch.tensor(x2.T, dtype=torch.float32, device=device)).detach().cpu().numpy()
    density_kernel_x2 = kernel.logpdf(x2)

    symKL = np.mean(density_gm_x1-density_kernel_x1) + np.mean(density_kernel_x2-density_gm_x2)
    return symKL
