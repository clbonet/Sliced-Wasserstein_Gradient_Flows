import torch
import gc
import sys

from .NF_base import *

sys.path.append("../icnn")

from icnn.icnn import *
from icnn.utils_icnn import *


class ConvexPotentialFlow(BaseNormalizingFlow):
    def __init__(self, icnn):
        super().__init__()
        self.icnn = icnn
    
    def forward(self, x):
        x.requires_grad_(True)
        y = self.icnn(x)
        z = torch.autograd.grad(y.sum(), x, retain_graph=True, create_graph=True)[0] ## nabla_ICNN
        
        hessian_icnn = hessian_ICNN(x, z, self.training)
        log_det = torch.log(torch.abs(torch.det(hessian_icnn)))
        
        if not self.icnn.training:
            z = z.detach()
            log_det = log_det.detach()
        
        return z, log_det
    
    def backward(self, z):
        x = reverse(self.icnn, z)
        return x, 0

def create_CPF(nh=32, nl=5, d=2):
    flows = []
    for i in range(nl):
        icnn = create_ICNN(device=device,d=d,w=nh,nh=3)
        flows.append(ConvexPotentialFlow(icnn))

    model = NormalizingFlows(flows).to(device)
    return model
    
