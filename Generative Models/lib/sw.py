import torch

import numpy as np
import torch.nn.functional as F
import torch.nn as nn


device = "cuda" if torch.cuda.is_available() else "cpu"


def emd1D(u_values, v_values, u_weights=None, v_weights=None,p=1, require_sort=True):
    n = u_values.shape[-1]
    m = v_values.shape[-1]

    device = u_values.device
    dtype = u_values.dtype

    if u_weights is None:
        u_weights = torch.full((n,), 1/n, dtype=dtype, device=device)

    if v_weights is None:
        v_weights = torch.full((m,), 1/m, dtype=dtype, device=device)

    if require_sort:
        u_values, u_sorter = torch.sort(u_values, -1)
        v_values, v_sorter = torch.sort(v_values, -1)

        u_weights = u_weights[..., u_sorter]
        v_weights = v_weights[..., v_sorter]

    zero = torch.zeros(1, dtype=dtype, device=device)
    
    u_cdf = torch.cumsum(u_weights, -1)
    v_cdf = torch.cumsum(v_weights, -1)

    cdf_axis, _ = torch.sort(torch.cat((u_cdf, v_cdf), -1), -1)
    
    u_index = torch.searchsorted(u_cdf, cdf_axis)
    v_index = torch.searchsorted(v_cdf, cdf_axis)

    u_icdf = torch.gather(u_values, -1, u_index.clip(0, n-1))
    v_icdf = torch.gather(v_values, -1, v_index.clip(0, m-1))

    cdf_axis = torch.nn.functional.pad(cdf_axis, (1, 0))
    delta = cdf_axis[..., 1:] - cdf_axis[..., :-1]

    if p == 1:
        return torch.sum(delta * torch.abs(u_icdf - v_icdf), axis=-1)
    if p == 2:
        return torch.sum(delta * torch.square(u_icdf - v_icdf), axis=-1)  
    return torch.sum(delta * torch.pow(torch.abs(u_icdf - v_icdf), p), axis=-1)


def sliced_cost(Xs, Xt, projections=None,u_weights=None,v_weights=None,p=1):
    if projections is not None:
        Xps = (Xs @ projections).T
        Xpt = (Xt @ projections).T
    else:
        Xps = Xs.T
        Xpt = Xt.T

    return torch.mean(emd1D(Xps,Xpt,
                       u_weights=u_weights,
                       v_weights=v_weights,
                       p=p))


def sliced_wasserstein(Xs, Xt, num_projections, device,
                       u_weights=None, v_weights=None, p=1):
    num_features = Xs.shape[1]

    # Random projection directions, shape (num_features, num_projections)
    projections = np.random.normal(size=(num_features, num_projections))
    projections = F.normalize(torch.from_numpy(projections), p=2, dim=0).type(Xs.dtype).to(device)

    return sliced_cost(Xs,Xt,projections=projections,
                       u_weights=u_weights,
                       v_weights=v_weights,
                       p=p)


def sw2_approx(X1, X2, device, u_weights=None, v_weights=None):
    n = X1.size(0)
    m = X2.size(0)
    d = X1.size(1)
    
    dtype = X1.dtype
    
    if u_weights is None:
        u_weights = torch.full((n,), 1/n, dtype=dtype, device=device).reshape(-1,1)
    else:
        u_weights = u_weights.reshape(-1,1)

    if v_weights is None:
        v_weights = torch.full((m,), 1/m, dtype=dtype, device=device).reshape(-1,1)
    else:
        v_weights = v_weights.reshape(-1,1)
    
    m1 = torch.sum(X1 * u_weights, dim=0)
    m2 = torch.sum(X2 * v_weights, dim=0)
    norms1 = torch.linalg.norm(X1-m1, dim=1)**2
    norms2 = torch.linalg.norm(X2-m2, dim=1)**2
    
    cpt1 = torch.sum(torch.square(m1-m2))/d
    cpt2 = (torch.sum(norms1 * u_weights[:,0])**(1/2)-torch.sum(norms2 * v_weights[:,0])**(1/2))**2

    return cpt1+cpt2/d


#def get_optimal_proj(Xs, Xt, theta, u_weights=None, v_weights=None,
                     #p=2, lr=1e-2, epochs=50):
    #theta.requires_grad_(True)
    #optimizer = torch.optim.Adam([theta], lr=lr)
        
    #for k in range(epochs):
        #Xps = (Xs @ theta).T
        #Xpt = (Xt @ theta).T
        
        #w2 = emd1D(Xps, Xpt, u_weights=u_weights, v_weights=v_weights, p=p)
        #w2.backward(retain_graph=True)
        
        #optimizer.step()
        #optimizer.zero_grad()
        
        #theta.data = theta.data/torch.sqrt(torch.sum(torch.square(theta.data))) ## projection
        
    #return theta.detach(), w2


#def max_SW(Xs, Xt, device, u_weights=None, v_weights=None, p=2, 
           #lr=1e-2, epochs=50):
    #num_features = Xs.shape[1]
    
    #Random projection directions, shape (num_features, num_projections)
    #projection = np.random.normal(size=(num_features, 1))
    #projection = F.normalize(torch.from_numpy(projection), p=2, dim=0).type(Xs.dtype).to(device)
    
    #theta_star, w2 = get_optimal_proj(Xs, Xt, theta=projection,
                                      #u_weights=u_weights, v_weights=v_weights, 
                                      #p=p, lr=lr, epochs=epochs)
    
    #return emd1D((Xs@theta_star).T, (Xt@theta_star).T, u_weights=u_weights, v_weights=v_weights, p=p)
    

def get_optimal_proj(Xs, Xt, theta, u_weights=None, v_weights=None,
                     p=2, lr=1e-4, epochs=50):
    theta.requires_grad_(True)
    
    L = []
    
    eta_max = 1
    gamma = 0.5
    beta_eta = 0.5
    c = 0.001
    
    for k in range(epochs):
        Xps = (Xs @ theta).T
        Xpt = (Xt @ theta).T
        
        w2 = emd1D(Xps, Xpt, u_weights=u_weights, v_weights=v_weights, p=p)
        grad_w2 = torch.autograd.grad(w2, [theta])[0]

        ## Line search
        norm_grad = torch.sum(torch.square(grad_w2))
        eta = eta_max
        with torch.no_grad():
            while True:
                theta2 = theta.detach().clone()-eta*grad_w2
                theta2.data = theta2.data/torch.sqrt(torch.sum(torch.square(theta2.data)))
                                
                Xps = (Xs @ theta2).T
                Xpt = (Xt @ theta2).T
                current_w2 = emd1D(Xps, Xpt, u_weights=u_weights, v_weights=v_weights, p=p)
                                
                if current_w2 <= w2-c*eta*norm_grad:
                    break
                else:
                    eta *= beta_eta
                
                if eta<1e-20:
                    break
                
        theta = theta-eta*grad_w2
        theta.data /= torch.sqrt(torch.sum(torch.square(theta.data))) ## projection
        
        L.append(w2.item())
        
#     plt.plot(L)
#     plt.show()
        
    return theta.detach(), w2


def max_SW(Xs, Xt, device, u_weights=None, v_weights=None, p=2, 
           lr=1e-4, epochs=50):
    num_features = Xs.shape[1]
    
    # Random projection directions, shape (num_features, num_projections)
    projection = np.random.normal(size=(num_features, 1))
    projection = F.normalize(torch.from_numpy(projection), p=2, dim=0).type(Xs.dtype).to(device)
        
    theta_star, w2 = get_optimal_proj(Xs, Xt, theta=projection,
                                      u_weights=u_weights, v_weights=v_weights, 
                                      p=p, lr=lr, epochs=epochs)

    return emd1D((Xs@theta_star).T, (Xt@theta_star).T, u_weights=u_weights, v_weights=v_weights, p=p)    
    
    
class MLP(nn.Module):
    def __init__(self, d_in, nh, d_out, n_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(d_in,nh))
        for i in range(n_layers):
            self.layers.append(nn.Linear(nh,nh))
        self.layers.append(nn.Linear(nh,d_out))

    def forward(self, x):
        for layer in self.layers:
            x = F.leaky_relu(layer(x),0.2)
        return x

def dual_dsw(Xps, Xpt, f_theta, f_theta2, C ,lambd, u_weights, v_weights, p):
    sw = torch.mean(emd1D(Xps,Xpt,u_weights=u_weights,v_weights=v_weights,p=p))**(1/p)
    return sw -lambd * torch.mean(torch.matmul(f_theta, f_theta2.T)) + lambd * C

def get_optimal_distr(Xs, Xt, num_projections, device,
                      u_weights=None, v_weights=None, p=2,
                      C=1, lambd=0.5, n_epochs=200, lr=1e-5):
    num_features = Xs.shape[1]

    f = MLP(num_features, 512, num_features, 2).to(device)
    
    optimizer = torch.optim.Adam(f.parameters(), lr=lr)
    
    L = []
    
    for k in range(n_epochs):
        projections = np.random.normal(size=(num_features, num_projections))
        projections = F.normalize(torch.from_numpy(projections), p=2, dim=0).type(Xs.dtype).to(device)
        
        f_theta = f(projections.T).T
        
        Xps = (Xs @ f_theta).T
        Xpt = (Xt @ f_theta).T
        
        projections2 = np.random.normal(size=(num_features, num_projections))
        projections2 = F.normalize(torch.from_numpy(projections2), p=2, dim=0).type(Xs.dtype).to(device)
        f_theta2 = f(projections2.T).T
        
        loss = -dual_dsw(Xps, Xpt, f_theta, f_theta2, C, lambd, u_weights, v_weights, p)
        
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        
        L.append(loss.item())
        
    #plt.plot(L)
    #plt.show()
        
    return f
        

def distributional_sw(Xs, Xt, num_projections, device, 
                      u_weights=None, v_weights=None, p=2,
                      C=1, lambd=0.5, n_epochs=200, lr=1e-4):
    
    f = get_optimal_distr(Xs, Xt, num_projections, device, 
                          u_weights, v_weights, p, 
                          C, lambd, n_epochs, lr)
    
    projections = np.random.normal(size=(Xs.shape[1], num_projections))
    projections = F.normalize(torch.from_numpy(projections), p=2, dim=0).type(Xs.dtype).to(device)
        
    f_theta = f(projections.T).T
    
    return sliced_cost(Xs,Xt,projections=projections,
                       u_weights=u_weights,
                       v_weights=v_weights,
                       p=p)
