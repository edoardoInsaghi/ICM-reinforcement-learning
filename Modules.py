import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np




class FDQN_NET(nn.Module):
    def __init__(self, channels_in, action_space):
        super(FDQN_NET, self).__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(channels_in, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 6 * 6, 512),
            nn.ReLU(),
            nn.Linear(512, action_space)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(32 * 6 * 6, 512),
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
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(1600, 512),
            nn.ReLU(),
            nn.Linear(512, action_space)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1600, 512),
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
    def __init__(self, channels_in, num_actions):
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
        self.fc = nn.Linear(32 * 6 * 6, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)
        self._initialize_weights()


    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x, model):
        
        x = self.backbone(x)

        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc(x))
        
        if model == 1:
            output = self.actor_linear(x)
        else:
            output = self.critic_linear(x)
        
        return output       


class DUELING_NET(nn.Module):
    def __init__(self, channels_in, action_space):
        super(DUELING_NET, self).__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(channels_in, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
            
        self.fc1 = nn.Sequential(
            nn.Linear(1600, 512),
            nn.ReLU(),
            nn.Linear(512, action_space)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1600, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.fc1_2 = nn.Sequential(
            nn.Linear(1600, 512),
            nn.ReLU(),
            nn.Linear(512, action_space)
        )

        self.fc2_2 = nn.Sequential(
            nn.Linear(1600, 512),
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
        
    @torch.no_grad()
    def get_value(self, x):
        self.eval()
        B, C, H, W = x.shape
        x = self.backbone(x)
        x = x.view(B, -1)
        return [self.fc2(x), self.fc2_2(x)]
    
    @torch.no_grad()
    def get_adv(self, x):
        self.eval()
        B, C, H, W = x.shape
        x = self.backbone(x)
        x = x.view(B, -1)
        return [self.fc1(x) - self.fc1(x).mean(), self.fc1_2(x) - self.fc1_2(x).mean()]



class Reverse_Dynamics_Module(nn.Module):

    def __init__(self, channels_in=4, action_space=12):
        super(Reverse_Dynamics_Module, self).__init__()


        self.backbone = nn.Sequential(
            nn.Conv2d(channels_in, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1152, 288),
            nn.ReLU(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(288*2, 256),
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
        B = state.shape[0]
        state, next_state = self.backbone(state), self.backbone(next_state)
        x = torch.cat((state, next_state), dim=1)
        return self.fc(x)
        
    @torch.no_grad()
    def get_latent_state(self, state):
        B, C, H, W = state.shape
        self.eval()
        state = self.backbone(state)
        return state.view(B, -1)
    
    
    
class ForwardModule(nn.Module):
    def __init__(self, action_space=12):
        super(ForwardModule, self).__init__()
        
        self.action_space = action_space
        
        self.fc = nn.Sequential(
            nn.Linear(288 + action_space, 256),
            nn.ReLU(),
            nn.Linear(256, 288)
        )
        
    
    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        return self.fc(x)