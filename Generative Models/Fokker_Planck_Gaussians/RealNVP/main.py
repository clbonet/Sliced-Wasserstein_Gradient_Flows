import argparse
import sys
import torch
import torch.distributions as D
import numpy as np

sys.path.append("../../lib")

from nf.realnvp import *
from nf.utils_nf import log_likelihood, symKL
from swgf_mynf import *
from utils_gaussians import *


parser = argparse.ArgumentParser()
parser.add_argument("--ntry", type=int, default=15, help="number of restart")
parser.add_argument("--epochs", type=int, default=200, help="Number of epochs in the inner loop")
parser.add_argument("--n_projs", type=int, default=1000, help="Number of projections")
parser.add_argument("--batch_size", type=int, default=1024, help="batch_size")
parser.add_argument("--ts",  type=float, nargs="+", help="List of time to evaluatet")
parser.add_argument("--tau", type=float, default=0.05, help="step size")
parser.add_argument("--fixed_t", help="If True, use tau/d as step size", action="store_true")
parser.add_argument("--stationary", help="Compare to the stationary distribution", action="store_true")
parser.add_argument("--high_dim", help="If True, in higher dimension", action="store_true")
parser.add_argument("--sw_approx", help="If True, use the concentration approximation of SW", action="store_true")
parser.add_argument("--reset", help="If True, use a randomly initialized NF at each outer iteration", action="store_true")
parser.add_argument("--fixed_t2", help="If True, use tau/d as step size", action="store_true")
args = parser.parse_args()


if __name__=="__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("device =", device,flush=True)

    ts = args.ts
    
    if args.high_dim:
        ds = [20, 30, 40, 50, 75, 100]
    else:
        ds = range(2,13)

    symKLs_sw = np.zeros((len(ts), len(ds), args.ntry))
    Fs = np.zeros((len(ts), len(ds), args.ntry))

    t_init = 0

    for ind, d in enumerate(ds):

        if args.fixed_t:
            h = args.tau/d
            t_end = np.max(ts)/d+h
        if args.fixed_t2:
            h = args.tau/(2*d)
            t_end = np.max(ts)/d + h
            print("tau",h, flush=True)
        else:
            h = args.tau
            t_end = np.max(ts)+h
                        
        n_steps = int(np.ceil((t_end-t_init)/h))
        print("n_steps", n_steps, flush=True)

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

            mu_stationary = b2
            sigma_stationary = A2_

            def V(x):
                x = x[:,:,None]
                y = torch.matmul(A,x-b[:,None])
                z = torch.matmul(torch.transpose(x-b[:,None],1,2),y)
                return z[:,0,0]/2

            def J(x, z, log_det):
                h = torch.mean(log_likelihood(z, log_det, device),axis=0) ## entropy
                return torch.mean(V(x),axis=0)+h

            mu0 = torch.tensor(np.zeros(d), dtype=torch.float, device=device)
            sigma0 = torch.eye(d, dtype=torch.float, device=device)
            rho_0 = D.MultivariateNormal(mu0,sigma0)

            if args.fixed_t or args.fixed_t2:
                lrs = 1e-4 * np.ones(n_steps)
                lrs[0] = 5e-3
            else:
                lrs = 1e-3 * np.ones(n_steps)
                lrs[0] = 5e-3

            Lrho = SWGF(rho_0, h, n_step=n_steps, n_epochs=args.epochs, d=d,
                        J=J, create_NF=create_RealNVP, nh=100, nl=5, lrs=lrs, 
                        num_projections=args.n_projs, n_samples=args.batch_size, 
                        sw_approx=args.sw_approx, reset_NF=args.reset)

            for i, t in enumerate(ts):
                ind_t = np.int(t/args.tau)
                if not args.fixed_t and not args.fixed_t2:
                    t *= d
                
                if args.stationary:
                    m = torch.tensor(mu_stationary, dtype=torch.float, device=device)
                    s = torch.tensor(sigma_stationary, dtype=torch.float, device=device)
                    ind_t = -1
                else:
                    m = torch.tensor(true_mu(t, A2, b2, mu02), 
                                    dtype=torch.float, device=device)
                    s = torch.tensor(true_Sigma(t, A2, sigma02, d), 
                                    dtype=torch.float, device=device)

                true_distr = D.MultivariateNormal(m,s)
                symKLs_sw[i, ind, k] = np.log10(symKL(Lrho[ind_t], true_distr, 10000, device))
                
                z_k = torch.randn(10000, d, device=device)
                x_k, log_det_k = Lrho[ind_t](z_k)
                f = J(x_k[-1], z_k, log_det_k)
                
                x_t = true_distr.sample((10000,))
                log_prob = true_distr.log_prob(x_t)
                f_true = torch.mean(V(x_t), axis=0)+torch.mean(log_prob, axis=0)
                
                Fs[i, ind, k] = torch.abs(f-f_true).detach().cpu().numpy()

            print("d",d,"k",k,flush=True)
            print("SW", symKLs_sw[:, ind, k],flush=True)
            print()
            

    dim = "high_d" if args.high_dim else "low_d"
    
    if args.sw_approx and args.reset:
        xp = "approx+reset"
    elif args.sw_approx:
        xp = "approx"
    elif args.reset:
        xp = "reset"
    elif args.fixed_t2:
        xp = "projs_less_iter"
    else:
        xp = "projs"

    for i, t in enumerate(ts):
        m = np.mean(symKLs_sw[i,:,:], axis=1)
        s = np.std(symKLs_sw[i,:,:], axis=1)
        plt.plot(ds, m, label="RealNVP-SWGF")
        plt.fill_between(ds,m-2*s/np.sqrt(args.ntry),m+2*s/np.sqrt(args.ntry),alpha=0.5)

        plt.grid(True)
        plt.legend()
        plt.xlabel("d")
        plt.ylabel(r"$\log_{10}\mathrm{SymKL}$")
        if args.fixed_t and not args.stationary:
            plt.title(r"$t="+str(t)+r"$")
            plt.savefig("./results/gaussians_"+xp+"_"+dim+"_t_"+str(t)+".png",format="png")
        elif not args.stationary:
            plt.title(r"$t="+str(t)+r"d$")
            plt.savefig("./results/gaussians_"+xp+"_"+dim+"_t_"+str(t)+"d.png",format="png")
        else:
            plt.savefig("./results/gaussians_stationary_"+"xp"+"_"+dim+".png", format="png")
        plt.close("all")

        if args.fixed_t:
            np.savetxt("./results/symKLs_"+xp+"_"+dim+"_t"+str(t), symKLs_sw[i,:,:], delimiter=",")
        else:
            np.savetxt("./results/symKLs_"+xp+"_"+dim+"_t"+str(t)+"d", symKLs_sw[i,:,:], delimiter=",")
            
            
        m = np.mean(Fs[i,:,:], axis=1)
        s = np.std(Fs[i,:,:], axis=1)
        plt.plot(ds, m, label="RealNVP-SWGF")
        plt.fill_between(ds,m-2*s/np.sqrt(args.ntry),m+2*s/np.sqrt(args.ntry),alpha=0.5)

        plt.grid(True)
        plt.legend()
        plt.xlabel("d")
        plt.ylabel(r"Error")
        if args.fixed_t and not args.stationary:
            plt.title(r"$t="+str(t)+r"$")
            plt.savefig("./results/F_gaussians_"+xp+"_"+dim+"_t_"+str(t)+".png",format="png")
        elif not args.stationary:
            plt.title(r"$t="+str(t)+r"d$")
            plt.savefig("./results/F_gaussians_"+xp+"_"+dim+"_t_"+str(t)+"d.png",format="png")
        else:
            plt.savefig("./results/F_gaussians_stationary_"+"xp"+"_"+dim+".png", format="png")
        plt.close("all")

        if args.fixed_t:
            np.savetxt("./results/Fs_"+xp+"_"+dim+"_t"+str(t), Fs[i,:,:], delimiter=",")
        else:
            np.savetxt("./results/Fs_"+xp+"_"+dim+"_t"+str(t)+"d", Fs[i,:,:], delimiter=",")
            