import torch
import torch.distributions as D


def log_likelihood(z, log_det, device):
    """
        Log likelihood
        Inputs:
        - z
        - $\log|\det J_T(z)| = -\log|\det J_{T^{-1}}(x)|$
    """
    d = z.size(1)
    distr = D.MultivariateNormal(torch.zeros(d,device=device),torch.eye(d,device=device))
    
    prior = distr.log_prob(z)
    return prior-log_det
    
    
def symKL(rho, distr, n_samples, device):
    ## Check if CPF Flow 
    cpf = False
    for flow in rho.flows:
        if flow.__class__.__name__ == "ConvexPotentialFlow":
            cpf = True
            break

    x1 = distr.sample((n_samples,))
    logprob_distr_x1 = distr.log_prob(x1).detach().cpu()

    rho.eval()
    if cpf: ## det not available for the inverse
        z, _ = rho.backward(x1)
        x, log_det = rho(z[-1])
        logprob_rho_x1 = log_likelihood(z[-1], log_det, device).detach().cpu()
	    
        log_det = log_det.detach()
        for i in range(len(x)):
            x[i] = x[i].detach()
    else: 
        z, log_det = rho.backward(x1)
        logprob_rho_x1 = log_likelihood(z[-1], -log_det, device).detach().cpu()
    
    z = torch.randn(n_samples, z[-1].size(1), device=device)
    x2, log_det = rho(z)
    x2 = x2[-1]
    logprob_rho_x2 = log_likelihood(z, log_det, device).detach().cpu()
    logprob_distr_x2 = distr.log_prob(x2).detach().cpu()
    
    s = torch.mean(logprob_distr_x1-logprob_rho_x1) + torch.mean(logprob_rho_x2-logprob_distr_x2)
    return s
    
