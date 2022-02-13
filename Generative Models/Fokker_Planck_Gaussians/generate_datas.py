import argparse
import numpy as np

from sklearn.datasets import make_spd_matrix


parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=15, help="Number  to generate")
parser.add_argument("--dataset", type=str, choices=["gaussian", "mixture"], default="gaussian")
parser.add_argument("--l", type=float, default=10, help="Parameter to draw mean for mixtures")

args = parser.parse_args()

def generate_gaussian(d):
    A = make_spd_matrix(d)
    b = np.random.randn(d)
    return A, b

if __name__=="__main__":
    Ms = [5 for k in range(12)]
    l = args.l

    for d in range(2,13):
#    for d in [20, 30, 40, 50, 75, 100]:
        for k in range(args.n):
            if args.dataset == "gaussian":
                A, b = generate_gaussian(d)        
                np.savetxt("./data/b_d"+str(d)+"_k"+str(k), b, delimiter=",")
                np.savetxt("./data/A_d"+str(d)+"_k"+str(k), A, delimiter=",")
            elif args.dataset == "mixture":
                mus = -l/2 + l * np.random.rand(Ms[d-1],d)
                np.savetxt("./data/mus_d"+str(d)+"_k"+str(k), mus, delimiter=",")
