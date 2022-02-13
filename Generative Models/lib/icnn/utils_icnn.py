import torch
import gc


def hessian_ICNN(x, nabla_icnn, training):
    n_samples = x.size(0) ## batch_size
    
    nabla_ICNN = nabla_icnn.reshape(n_samples, -1)
    H = []
    for i in range(nabla_ICNN.shape[1]):
        H.append(torch.autograd.grad(nabla_ICNN[:, i].sum(), x, 
                                     create_graph=training, retain_graph=True)[0])

    # H: (batch_size, d, d)
    H = torch.stack(H, dim=1)    
        
    return H


def reverse(icnn, y, max_iter=10000, lr=1.0, tol=1e-12, x=None):
    """
        https://github.com/CW-Huang/CP-Flow/blob/d01303cb4ebeb5a0bbfca638ffaf5b7a8ec22fb1/lib/flows/cpflows.py#L43
    """
    if x is None:
        x = y.clone().detach().requires_grad_(True)

    def closure():
        # Solves x such that f(x) - y = 0
        # <=> Solves x such that argmin_x F(x) - <x,y>
        F = icnn(x)
        loss = torch.sum(F) - torch.sum(x * y)
        x.grad = torch.autograd.grad(loss, x)[0].detach()
        return loss

    optimizer = torch.optim.LBFGS([x], lr=lr, line_search_fn="strong_wolfe", max_iter=max_iter, 
                                  tolerance_grad=tol, tolerance_change=tol)

    optimizer.step(closure)

    torch.cuda.empty_cache()
    gc.collect()

    return x
