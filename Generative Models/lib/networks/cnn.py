import torch

import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, no_of_channels=1, noise_dim=100, gen_dim=32):
        super(CNN, self).__init__()
        
        self.dense1 = nn.Linear(noise_dim, 128*7*7)
        
        self.conv1 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        
  
    def forward(self, z):
        '''
        forward pass of the generator: 
        Input is a noise tensor and output is generated images resembling Training data.
        '''
        z = z.view(-1, 100)
        x = F.leaky_relu(self.dense1(z), 0.2)
        x = x.view(-1, 128, 7, 7)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.sigmoid(self.conv3(x))
        return x
