import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
        


class Net(nn.Module):

    def __init__(self, channels_in, action_space, size=84, algo="fdql"):
        super(Net, self).__init__()

        self.algo = algo
        
        self.blocks = nn.ModuleList(
            [ConvBlock(channels_in, 32, kernel_size=8, stride=4)] + 
            [ConvBlock(32, 64, kernel_size=4, stride=2)] + 
            [ConvBlock(64, 64, stride=1)]
        )

        self.fc1 = nn.Sequential (
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, action_space)
        )

        self.fc2 = nn.Sequential (
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, action_space) if algo != "ac" and algo != "dueling" else nn.Linear(512, 1)
        )

        if algo == "dueling":

            self.fc1_2 = nn.Sequential (
                nn.Flatten(),
                nn.Linear(3136, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, action_space)
            )

            self.fc2_2 = nn.Sequential (
                nn.Flatten(),
                nn.Linear(3136, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 1)
            )

        if algo == "fdql":
            for p in self.fc2.parameters():
                p.requires_grad = False

        
    def forward(self, x, model=1):

        for block in self.blocks:
            x = block(x)

        if self.algo == "dueling":
            if model == 1:
                return self.fc2(x) + self.fc1(x) - self.fc1(x).mean()
            else:
                return self.fc2_2(x) + self.fc1_2(x) - self.fc1_2(x).mean()
        
        if model == 1:
            return self.fc1(x) 
        else:
            return self.fc2(x)
        


class Reverse_Dynamics_Module(nn.Module):

    def __init__(self, size=84, action_space=12, device=torch.device('cpu')):
        super(Reverse_Dynamics_Module, self).__init__()
        self.size = size
        self.action_space = action_space
        self.device = device

        self.blocks = nn.ModuleList(
            [ConvBlock(4, 32, stride=2, padding=1)] + 
            [ConvBlock(32, 32, stride=2, padding=1)] + 
            [ConvBlock(32, 32, stride=2, padding=1)] +
            [ConvBlock(32, 32, stride=2, padding=1)]
        )

        self.fc = nn.Sequential(
            nn.Linear(32*6*6*2, 256),
            nn.ReLU(),
            nn.Linear(256, action_space)
        )

    def forward(self, state, next_state):

        x1 = torch.tensor(state, dtype=torch.float32).to(self.device)
        x2 = torch.tensor(next_state, dtype=torch.float32).to(self.device).unsqueeze(0)
        for block in self.blocks:
              x1 = block(x1)
              x2 = block(x2)

        x = torch.cat((x1.view(1,-1), x2.view(1,-1)), dim=1)
        return self.fc(x)
        

    def get_latent_state(self, state):
        self.eval()
        return self.blocks(state).Flatten()