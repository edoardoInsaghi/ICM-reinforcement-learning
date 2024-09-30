import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


'''
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
'''



class FDQN_NET(nn.Module):
    def __init__(self, channels_in, action_space):
        super(FDQN_NET, self).__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(channels_in, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_space)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_space)
        )
        
        for p in self.fc2.parameters():
            p.requires_grad = False

        self._initialize_weights()


    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)


    def forward(self, x, model=1):
        B, C, H, W = x.shape
        x = self.backbone(x)
        x = x.view(B, -1)
        if model == 1:
            return self.fc1(x)
        else:
            return self.fc2(x)
        



class DDQN_NET(nn.Module):
    def __init__(self, channels_in, action_space):
        super(DDQN_NET, self).__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(channels_in, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_space)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_space)
        )

        self._initialize_weights()


    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)


    def forward(self, x, model=1):
        B, C, H, W = x.shape
        x = self.backbone(x)
        x = x.view(B, -1)
        
        if model == 1:
            return self.fc1(x)
        else:
            return self.fc2(x)


class AC_NET(nn.Module):
    def __init__(self, channels_in, action_space):
        super(AC_NET, self).__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(channels_in, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 6 * 6, 512),
            nn.ReLU(),
            nn.Linear(512, action_space),
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 6 * 6, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        
        self._initialize_weights()
            
            
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)


    def forward(self, x, model):
        x = self.backbone(x)
        x = x.view(x.shape[0], -1)
        
        if model == 1:
            return self.fc1(x)
        else:
            return self.fc2(x)
        


class DUELING_NET(nn.Module):
    def __init__(self, channels_in, action_space):
        super(DUELING_NET, self).__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(channels_in, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_space)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.fc1_2 = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_space)
        )

        self.fc2_2 = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self._initialize_weights()


    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)


    def forward(self, x, model=1):
        B, C, H, W = x.shape
        x = self.backbone(x)
        x = x.view(B, -1)
        if model == 1:
            return self.fc2(x) + self.fc1(x) - self.fc1(x).mean()
        else:
            return self.fc2_2(x) + self.fc1_2(x) - self.fc1_2(x).mean()