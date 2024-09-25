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

    def __init__(self, channels_in, action_space, size=84, algo="fdql", learn_states=False):
        super(Net, self).__init__()

        self.algo = algo
        
        if not learn_states:
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

        else:
            self.RDM = Reverse_Dynamics_Module(size=size, action_space=action_space)

            self.fc1 = nn.Sequential (
                nn.Flatten(),
                nn.Linear(288, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, action_space)
            )

            self.fc2 = nn.Sequential (
                nn.Flatten(),
                nn.Linear(288, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, action_space) if algo != "ac" and algo != "dueling" else nn.Linear(512, 1)
            )

            if algo == "dueling":

                self.fc1_2 = nn.Sequential (
                    nn.Flatten(),
                    nn.Linear(288, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Linear(512, action_space)
                )

                self.fc2_2 = nn.Sequential (
                    nn.Flatten(),
                    nn.Linear(288, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Linear(512, 1)
                )

        if algo == "fdql":
            for p in self.fc2.parameters():
                p.requires_grad = False
        
        self._initialize_weights()


    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

        
    def forward(self, x, model=1):

        if hasattr(self, "RDM"):
            x = self.RDM.get_latent_state(x)
        else:
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

    def __init__(self, size=84, action_space=12):
        super(Reverse_Dynamics_Module, self).__init__()
        self.size = size
        self.action_space = action_space

        self.blocks = nn.ModuleList(
            [ConvBlock(4, 32, stride=2, padding=1)] + 
            [ConvBlock(32, 32, stride=2, padding=1)] + 
            [ConvBlock(32, 32, stride=2, padding=1)] +
            [ConvBlock(32, 8, stride=2, padding=1)]
        )

        self.fc = nn.Sequential(
            nn.Linear(8*6*6*2, 256),
            nn.ReLU(),
            nn.Linear(256, action_space)
        )
        self._initialize_weights()


    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)


    def forward(self, state, next_state):
        B, C, H, W = state.shape
        for block in self.blocks:
              state = block(state)
              next_state = block(next_state)

        x = torch.cat((state.view(B,-1), state.view(B,-1)), dim=1)
        return self.fc(x)
        
    @torch.no_grad()
    def get_latent_state(self, state):
        B, C, H, W = state.shape
        self.eval()
        for block in self.blocks:
            state = block(state)
        return state.view(B, -1)
        