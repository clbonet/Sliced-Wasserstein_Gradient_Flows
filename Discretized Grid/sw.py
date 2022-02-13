import torch

import numpy as np
import torch.nn.functional as F


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
