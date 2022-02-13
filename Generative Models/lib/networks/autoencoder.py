"""
    Same architecture as https://github.com/aliutkus/swf/blob/master/code/networks/autoencoder.py
"""

import torch

from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_shape, d=48):
        super().__init__()
        self.input_shape = input_shape
        self.conv1 = nn.Conv2d(input_shape[0], 3, 
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(3, input_shape[-1], 
                               kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(input_shape[-1], input_shape[-1],
                               kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(input_shape[-1], input_shape[-1],
                               kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(int(input_shape[-1]**3/4), d)
        
        
    def forward(self, x):
        if len(x.shape) == 3:
            x = x[None, ...]

        out = torch.relu(self.conv1(x))
        out = torch.relu(self.conv2(out))
        out = torch.relu(self.conv3(out))
        out = torch.relu(self.conv4(out))
        out = out.view(out.size(0), -1)
        return torch.relu(self.fc1(out))


class Decoder(nn.Module):
    def __init__(self, input_shape, bottleneck_size=48):
        super().__init__()
        self.input_shape = input_shape
        d = input_shape[-1]

        self.fc4 = nn.Linear(bottleneck_size, int(d/2 * d/2 * d))
        self.deconv1 = nn.ConvTranspose2d(d, d,
                                          kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(d, d,
                                          kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(d, d,
                                          kernel_size=2, stride=2, padding=0)
        self.conv5 = nn.Conv2d(d, self.input_shape[0],
                               kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        d = self.input_shape[-1]
        out = torch.relu(self.fc4(x))
        out = out.view(-1, d, int(d/2), int(d/2))
        out = torch.relu(self.deconv1(out))
        out = torch.relu(self.deconv2(out))
        out = torch.relu(self.deconv3(out))
        return torch.sigmoid(self.conv5(out))


class AE(nn.Module):
    def __init__(self, input_shape, d=48):
        super().__init__()
        self.encoder = Encoder(input_shape, d)
        self.decoder = Decoder(input_shape, d)
    
    def forward(self, x):
        z = self.encoder(x)        
        y = self.decoder(z)
        return y
    
    
