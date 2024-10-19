import torch.nn as nn
import torch.nn.functional as F


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