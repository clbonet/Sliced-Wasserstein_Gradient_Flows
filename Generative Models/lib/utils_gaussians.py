import numpy as np

from scipy.linalg import expm, sqrtm


def true_mu(t, A2, b2, mu02):
    B = expm(-A2*t)
    return b2+B @(mu02-b2)

def true_Sigma(t, A2, sigma02, d):
    e = expm(-2*A2*t)
    B = expm(-t*A2)
    A2_ = np.linalg.inv(A2)
    A12_ = sqrtm(A2_)
    cpt1 = B@sigma02@B.T
    cpt2 = A12_@(np.eye(d)-e)@A12_.T
    return cpt1+cpt2
