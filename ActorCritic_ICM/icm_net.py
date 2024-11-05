import torch.nn as nn
import torch.nn.functional as F
import torch


class Reverse_Dynamics_Module(nn.Module):
    def __init__(self, channels_in=4, action_space=12):
        super(Reverse_Dynamics_Module, self).__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(channels_in, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(1152, 288),
            nn.ReLU(),)
        
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


    def forward(self, latent_state, latent_next_state):
        x = torch.cat((latent_state, latent_next_state), dim=1)
        return self.fc(x)
        
    @torch.no_grad()
    def get_latent_state(self, state):
        B, C, H, W = state.shape
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