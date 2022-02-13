import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np


class ConvexQuadratic(nn.Module):
    '''
        Convex Quadratic Layer
        (https://github.com/iamalexkorotin/Wasserstein2Barycenters/blob/cfd2e0d2614ec5d802071ec92076c6e52290215d/src/layers.py#L6)

        Appendix B.2 [1]
        [1] Korotin, Alexander, et al. "Wasserstein-2 generative networks." arXiv preprint arXiv:1909.13082 (2019).
    '''
    __constants__ = ['in_features', 'out_features', 'quadratic_decomposed', 'weight', 'bias']

    def __init__(self, in_features, out_features, bias=True, rank=1):
        super(ConvexQuadratic, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        self.quadratic_decomposed = nn.Parameter(torch.Tensor(
            torch.randn(in_features, rank, out_features)
        ))
        self.weight = nn.Parameter(torch.Tensor(
            torch.randn(out_features, in_features)
        ))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        quad = ((input.matmul(self.quadratic_decomposed.transpose(1,0)).transpose(1, 0)) ** 2).sum(dim=1)
        linear = F.linear(input, self.weight, self.bias)
        return quad + linear


class DenseICNN(nn.Module):
    '''
        Fully Conncted ICNN with input-quadratic skip connections
        (https://github.com/iamalexkorotin/Wasserstein2Barycenters/blob/main/src/icnn.py)

        Appendix B.2 [1]
        [1] Korotin, Alexander, et al. "Wasserstein-2 generative networks." arXiv preprint arXiv:1909.13082 (2019).
    '''
    def __init__(
        self, in_dim, 
        hidden_layer_sizes=[32, 32, 32],
        rank=1, activation='celu', dropout=0.03,
        strong_convexity=1e-6
    ):
        super(DenseICNN, self).__init__()
        
        self.strong_convexity = strong_convexity
        self.hidden_layer_sizes = hidden_layer_sizes
        self.droput = dropout
        self.activation = activation
        self.rank = rank
        
        self.quadratic_layers = nn.ModuleList([
            nn.Sequential(
                ConvexQuadratic(in_dim, out_features, rank=rank, bias=True),
                nn.Dropout(dropout)
            )
            for out_features in hidden_layer_sizes
        ])
        
        sizes = zip(hidden_layer_sizes[:-1], hidden_layer_sizes[1:])
        self.convex_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, out_features, bias=False),
                nn.Dropout(dropout)
            )
            for (in_features, out_features) in sizes
        ])
        
        self.final_layer = nn.Linear(hidden_layer_sizes[-1], 1, bias=False)

    def forward(self, input):
        output = self.quadratic_layers[0](input)
        for quadratic_layer, convex_layer in zip(self.quadratic_layers[1:], self.convex_layers):
            output = convex_layer(output) + quadratic_layer(input)
            if self.activation == 'celu':
                output = torch.celu(output)
            elif self.activation == 'softplus':
                output = F.softplus(output)
            elif self.activation == 'relu':
                output = F.relu(output)
            else:
                raise Exception('Activation is not specified or unknown.')
        
        return self.final_layer(output) + .5 * self.strong_convexity * (input ** 2).sum(dim=1).reshape(-1, 1)
    
    def push(self, input):
        """
            Compute the gradient
        """
        output = torch.autograd.grad(
            outputs=self.forward(input), inputs=input,
            create_graph=True, retain_graph=True,
            only_inputs=True,
            grad_outputs=torch.ones((input.size()[0], 1)).cuda().float()
        )[0]
        return output    
    
    def convexify(self):
        """
            Enfore positivity of weights 
            (to have convexity)
        """
        for layer in self.convex_layers:
            for sublayer in layer:
                if (isinstance(sublayer, nn.Linear)):
                    sublayer.weight.data.clamp_(0)
        self.final_layer.weight.data.clamp_(0)
              

def create_ICNN(device="cuda", w=256, d=2, nh=2):
    model = DenseICNN(d, hidden_layer_sizes=[w for k in range(nh)], activation="softplus", 
                      dropout=0, strong_convexity=1e-3)

    # Manual weights init
    for p in model.parameters():
        p.data = torch.randn(p.shape, device=device, dtype=torch.float32) / 2.
        
    ## Pretraining for \nabla\psi(x)=x
    X0_sampler = D.MultivariateNormal(torch.zeros(d, device=device),torch.eye(d,device=device)) 
    # distributions.StandardNormalSampler(dim=2, requires_grad=True)
    D_opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.8, 0.9), weight_decay=1e-7)

    model.train(True)
    loss = np.inf

    for iteration in range(10000):
        X = (X0_sampler.sample((500,))).detach() * 4
        X.requires_grad_(True)

        loss = F.mse_loss(model.push(X), X.detach())
        loss.backward()
        D_opt.step(); D_opt.zero_grad()
        model.convexify()

        if loss.item() < 1e-2:
            break

    return model
