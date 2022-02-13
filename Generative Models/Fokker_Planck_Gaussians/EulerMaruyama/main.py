import argparse
import sys
import torch
import torch.distributions as D
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde

sys.path.append("../../lib")

from em import *
from utils_gaussians import *


parser = argparse.ArgumentParser()
parser.add_argument("--ntry", type=int, default=15, help="number of restart")
parser.add_argument("--t", type=float, help="t_end")
parser.add_argument("--td", help="If True, multiplies t_end by d", action="store_true")
parser.add_argument("--stationary", help="If True, compare to the stationary distribution", action="store_true")
parser.add_argument("--high_dim", help="If True, in higher dimension", action="store_true")
args = parser.parse_args()


if __name__=="__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("device =", device,flush=True)

    if args.high_dim:
        ds = [20, 30, 40, 50, 75, 100]
    else:
        ds = range(2,13)

    symKLs_em = np.zeros((len(ds), args.ntry, 3))
    Fs = np.zeros((len(ds), args.ntry, 3))

    t_end = args.t

    for ind, d in enumerate(ds):
        if args.td:
            t1 = t_end*d
            print(t1/d,t_end,d)
        else:
            t1 = t_end

        for k in range(args.ntry):
            A_np = np.loadtxt("../data/A_d"+str(d)+"_k"+str(k), delimiter=",")
            A = torch.tensor(A_np, dtype=torch.float, device=device)

            b_np = np.loadtxt("../data/b_d"+str(d)+"_k"+str(k), delimiter=",")
            b = torch.tensor(b_np, dtype=torch.float, device=device)

            mu0 = torch.zeros(d, dtype=torch.float, device=device)
            sigma0 = torch.tensor(np.eye(d),device=device,dtype=torch.float)

            A2 = A.detach().cpu().numpy()
            mu02 = mu0.detach().cpu().numpy()
            sigma02 = sigma0.detach().cpu().numpy()
            A2_ = np.linalg.inv(A2)
            b2 = b.detach().cpu().numpy()

            if args.stationary:
                m = torch.tensor(b2, dtype=torch.float, device=device)
                s = torch.tensor(A2_, dtype=torch.float, device=device)
            else:
                m = torch.tensor(true_mu(t1, A2, b2, mu02), 
                                dtype=torch.float, device=device)
                s = torch.tensor(true_Sigma(t1, A2, sigma02, d), 
                                dtype=torch.float, device=device)
            true_distr = D.MultivariateNormal(m,s)


            def V(x):
                x = x[:,:,None]
                y = torch.matmul(A,x-b[:,None])
                z = torch.matmul(torch.transpose(x-b[:,None],1,2),y)
                return z[:,0,0]/2

            for j, n in enumerate([1000,10000,50000]):
                x = EulerMaruyama(V, n, d=d, t_end=t1, device=device)
                kernel = gaussian_kde(x.T.detach().cpu())
                symKLs_em[ind, k, j] = np.log10(symKL_kernel(true_distr, kernel, device=device))
                
                                
                x_t = true_distr.sample((10000,))
                log_prob = true_distr.log_prob(x_t)
                f_true = torch.mean(V(x_t), axis=0)+torch.mean(log_prob, axis=0)
                
                x2 = kernel.resample((10000,))
                log_prob = torch.tensor(kernel.logpdf(x2), device=device, dtype=torch.float)
                f = torch.mean(V(torch.tensor(x2.T, device=device, dtype=torch.float)), axis=0) + torch.mean(log_prob, axis=0)
                
                Fs[ind, k ,j] = torch.abs(f-f_true).detach().cpu().numpy()

            print("d",d,"k",k,symKLs_em[ind,k,:],flush=True)

    dim = "high_d_" if args.high_dim else "low_d_"

    for j, n in enumerate([1000,10000,50000]):
        m = np.mean(symKLs_em[:,:,j],axis=1)
        s = np.std(symKLs_em[:,:,j],axis=1)
        plt.plot(ds, m,label="EM  with "+str(n)+" particles")
        plt.fill_between(ds,m-2*s/np.sqrt(args.ntry),m+2*s/np.sqrt(args.ntry),alpha=0.5)

    plt.grid(True)
    plt.legend()
    plt.xlabel("d")
    plt.ylabel(r"$\log_{10}\mathrm{SymKL}$")
    if args.stationary:
        plt.savefig("./results/gaussians_stationary_"+dim+".png", format="png")
    elif args.td:
        plt.title(r"$t="+str(args.t)+r"d$")
        plt.savefig("./results/gaussians_"+dim+"t_"+str(args.t)+"d.png", format="png")
    else:
        plt.title(r"$t="+str(args.t)+r"$")
        plt.savefig("./results/gaussians_"+dim+"t_"+str(args.t)+".png", format="png")
    plt.close("all")

    for j, n in enumerate([1000,10000,50000]):
        if args.stationary:
            np.savetxt("./results/symKLs_EM_"+dim+"stationary_n"+str(n), symKLs_em[:,:,j], delimiter=",")
        elif args.td:
            np.savetxt("./results/symKLs_EM_"+dim+"t"+str(args.t)+"d_n"+str(n), symKLs_em[:,:,j], delimiter=",")
        else:
            np.savetxt("./results/symKLs_EM_"+dim+"t"+str(args.t)+"_n"+str(n), symKLs_em[:,:,j], delimiter=",")

            
    for j, n in enumerate([1000,10000,50000]):
        m = np.mean(Fs[:,:,j],axis=1)
        s = np.std(Fs[:,:,j],axis=1)
        plt.plot(ds, m,label="EM  with "+str(n)+" particles")
        plt.fill_between(ds,m-2*s/np.sqrt(args.ntry),m+2*s/np.sqrt(args.ntry),alpha=0.5)
            
            
    plt.grid(True)
    plt.legend()
    plt.xlabel("d")
    plt.ylabel(r"Error")
    if args.stationary:
        plt.savefig("./results/F_gaussians_stationary_"+dim+".png", format="png")
    elif args.td:
        plt.title(r"$t="+str(args.t)+r"d$")
        plt.savefig("./results/F_gaussians_"+dim+"t_"+str(args.t)+"d.png", format="png")
    else:
        plt.title(r"$t="+str(args.t)+r"$")
        plt.savefig("./results/F_gaussians_"+dim+"t_"+str(args.t)+".png", format="png")
    plt.close("all")

    
    for j, n in enumerate([1000,10000,50000]):
        if args.stationary:
            np.savetxt("./results/F_EM_"+dim+"stationary_n"+str(n), Fs[:,:,j], delimiter=",")
        elif args.td:
            np.savetxt("./results/F_EM_"+dim+"t"+str(args.t)+"d_n"+str(n), Fs[:,:,j], delimiter=",")
        else:
            np.savetxt("./results/F_EM_"+dim+"t"+str(args.t)+"_n"+str(n), Fs[:,:,j], delimiter=",")
