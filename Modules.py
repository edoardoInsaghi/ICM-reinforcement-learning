import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
        


class Net(nn.Module):

    def __init__(self, channels_in, action_space, size=84, algo="fdql"):
        super(Net, self).__init__()
        
        self.blocks = nn.ModuleList(
            [ConvBlock(channels_in, 32, kernel_size=8, stride=4)] + 
            [ConvBlock(32, 64, kernel_size=4, stride=2)] + 
            [ConvBlock(64, 64, stride=1)]
        )

        self.fc1target = nn.Linear(3136, 512)
        self.fc1online = nn.Linear(3136, 512)
        self.fc2target = nn.Linear(512, action_space)
        self.fc2online = nn.Linear(512, action_space)

        if algo == "fdql":
            for p in self.fc2target.parameters():
                p.requires_grad = False

        if algo == "ac":
            self.fc2target = nn.Linear(512, 1)  
        

    def forward(self, x, model):
        B, C, H, W = x.shape

        for block in self.blocks:
            x = block(x)
            
        x = x.view(B, -1)
        
        if model == "online":
            x = F.relu(self.fc1online(x))
            x = self.fc2online(x)
        
        else:
            x = F.relu(self.fc1target(x))
            x = self.fc2target(x)

        return x
        
