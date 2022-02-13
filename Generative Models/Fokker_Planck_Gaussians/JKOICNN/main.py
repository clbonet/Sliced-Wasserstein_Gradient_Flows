import argparse
import torch
import sys
import torch.distributions as D
import numpy as np

from scipy.stats import gaussian_kde
from scipy.linalg import expm, sqrtm

sys.path.append("../../lib")

from icnn.icnn import *
from jko_icnn import *
from utils_gaussians import *


parser = argparse.ArgumentParser()
parser.add_argument("--ntry", type=int, default=15, help="number of restart")
parser.add_argument("--epochs", type=int, default=500, help="Number of epochs in the inner loop")
parser.add_argument("--batch_size", type=int, default=1024, help="batch_size")
parser.add_argument("--ts",  type=float, nargs="+", help="List of time to evaluate t")
parser.add_argument("--tau", type=float, default=0.05, help="step size")
parser.add_argument("--stationary", help="Compare to the stationary distribution", action="store_true")
args = parser.parse_args()


if __name__=="__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("device =", device,flush=True)

    h = args.tau

    ts = args.ts
    t_init = 0
    t_end = np.max(ts)+h
    
    n_steps = int(np.ceil((t_end-t_init)/h))

    ds = range(2,13)

    symKLs_icnn = np.zeros((len(ts), len(ds), args.ntry))
    Fs = np.zeros((len(ts), len(ds), args.ntry))

    for ind, d in enumerate(ds):
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


            def J(x, nabla_icnn):
                h = -log_det_Hessian_approx(x, nabla_icnn, training=True, device=device)
                v = V(nabla_icnn)
                f = torch.mean(v+h, dim=0)
                return f
            
            def J2(Lrho, n_samples):
                print(len(Lrho))
                x, log_prob = samples_from_rho_prev_density(Lrho, n_samples)
                h = torch.mean(log_prob, dim=0)

                f = torch.mean(V(x), dim=0) + h
                return f

            mu0 = torch.tensor(np.zeros(d), dtype=torch.float, device=device)
            sigma0 = torch.eye(d, dtype=torch.float, device=device)
            rho_0 = D.MultivariateNormal(mu0,sigma0)

            lrs = 5e-3 

            Lrho = WGF_ICNN(rho_0, h, n_step=n_steps, n_epochs=args.epochs, J=J, create_NN=create_ICNN, d=d, w=64, lrs=5e-3, 
                            n_samples=args.batch_size, plot_loss=False, device=device)


            for i, t in enumerate(ts):
                ind_t = np.int(t/args.tau)
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

                if args.stationary:
                    symKLs_icnn[i, ind, k] = np.log10(symKL(Lrho, true_distr, 10000, device))
                else:
                    symKLs_icnn[i, ind, k] = np.log10(symKL(Lrho[:ind_t+1], true_distr, 10000, device))
                    
                if args.stationary:
                    f = J2(Lrho, 10000)
                else:
                    f = J2(Lrho[:ind_t+1], 10000)
                
                x_t = true_distr.sample((10000,))
                log_prob = true_distr.log_prob(x_t)
                f_true = torch.mean(V(x_t), axis=0)+torch.mean(log_prob, axis=0)

                Fs[i, ind, k] = torch.abs(f-f_true).detach().cpu().numpy()


            print("d",d,"k",k,flush=True)
            print("SW_ICNN", symKLs_icnn[:, ind, k],flush=True)
            print()


    for i, t in enumerate(ts):
        m = np.mean(symKLs_icnn[i,:,:], axis=1)
        s = np.std(symKLs_icnn[i,:,:], axis=1)
        plt.plot(ds, m, label="JKO-ICNN")
        plt.fill_between(ds,m-2*s/np.sqrt(args.ntry),m+2*s/np.sqrt(args.ntry),alpha=0.5)

        plt.grid(True)
        plt.legend()
        plt.xlabel("d")
        plt.ylabel(r"$\log_{10}\mathrm{SymKL}$")
        if not args.stationary:
            plt.title(r"$t="+str(t)+r"$")
            plt.savefig("./results/gaussians_t_"+str(t)+".png",format="png")
        else:
            plt.savefig("./results/gaussians_stationary.png", format="png")
        plt.close("all")

        np.savetxt("./results/symKLs_ICNN_t"+str(t), symKLs_icnn[i,:,:], delimiter=",")
        

        np.savetxt("./results/FS_ICNN_t"+str(t), Fs[i,:,:], delimiter=",")
        
        m = np.mean(Fs[i,:,:], axis=1)
        s = np.std(Fs[i,:,:], axis=1)
        plt.plot(ds, m, label="JKO-ICNN")
        plt.fill_between(ds,m-2*s/np.sqrt(args.ntry),m+2*s/np.sqrt(args.ntry),alpha=0.5)

        plt.grid(True)
        plt.legend()
        plt.xlabel("d")
        plt.ylabel(r"Error")
        if not args.stationary:
            plt.title(r"$t="+str(t)+r"$")
            plt.savefig("./results/F_gaussians_t_"+str(t)+".png",format="png")
        else:
            plt.savefig("./results/F_gaussians_stationary.png", format="png")
        plt.close("all")

        
        np.savetxt("./results/Fs_ICNN_t"+str(t), Fs[i,:,:], delimiter=",")
        
        
