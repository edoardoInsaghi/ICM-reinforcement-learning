import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
        


class Net(nn.Module):
    def __init__(self, channels_in, action_space, size=84):
        super(Net, self).__init__()
        
        self.blocks = nn.ModuleList(
            [ConvBlock(channels_in, 32)] + 
            [ConvBlock(32, 64, stride=2)] + 
            [ConvBlock(64, 128, stride=2)] +
            [ConvBlock(128, 16)]
        )

        self.fc1 = nn.Linear(4624, 256)
        self.fc2target = nn.Linear(256, action_space)
        self.fc2online = nn.Linear(256, action_space)

        for p in self.fc2target.parameters():
            p.requires_grad = False
        

    def forward(self, x, model):
        B, C, H, W = x.shape

        for block in self.blocks:
            x = block(x)
        
        x = x.view(B, -1)
        x = F.relu(self.fc1(x))

        x = self.fc2online(x) if model=="online" else self.fc2target(x)

        return x


class Agent():
    def __init__(self, action_space, gamma=0.9, batch_size=32, size=84, max_memory=int(1e4), device="cpu"):
        self.action_space = action_space
        self.gamma = gamma
        self.device = device
        self.memory = []
        self.max_memory = max_memory
        self.net = Net(4, action_space, size=size).to(device)
        self.batch_size = batch_size
        self.loss = torch.nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-4)

    def cache(self, state, next_state, action, reward, done):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])

        self.memory.append({"State":state, "Next_state":next_state, "Action":action, "Reward":reward, "Done":done})
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)

    def sample_from_memory(self, n=None):
        n = n if n is not None else self.batch_size
        idx = np.random.choice(len(self.memory), n, replace=False)
        samples = [self.memory[i] for i in idx]
        states = torch.stack([s["State"] for s in samples])
        next_states = torch.stack([s["Next_state"] for s in samples])
        actions = torch.stack([s["Action"] for s in samples])
        rewards = torch.stack([s["Reward"] for s in samples])
        dones = torch.stack([s["Done"] for s in samples])

        return states, next_states, actions, rewards, dones


    def td_estimate(self, state, action):
        self.net.eval()
        state = state.to(self.device)
        action = action.to(self.device)
        q = self.net(state, "online")[np.arange(0, self.batch_size), action]
        return q
    
    @torch.no_grad()
    def td_target(self, reward, next_state):
        self.net.eval()
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        q_online = self.net(next_state, "online")
        a = torch.argmax(q_online, dim=1)
        q_target = self.net(next_state, "target")[np.arange(0, self.batch_size), a]
        return reward + self.gamma * q_target
    
    def update_q_online(self, td_estimate, td_target):
        self.net.train()
        loss = self.loss(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def update_target(self):
        self.net.fc2target.load_state_dict(self.net.fc2online.state_dict())
 
    def act(self, state, epsilon=0.0):
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_space)
        else:
            self.net.eval()
            state = state.to(self.device)
            q = self.net(state, "online")
            return torch.argmax(q).item()



if __name__ == "__main__":
    mario = Agent(14)

    tdestimate = mario.td_estimate(torch.randn(32, 4, 84, 84), 1)
    tdtarget = mario.td_target(1, torch.randn(32, 4, 84, 84))
    loss = mario.update_q_online(tdestimate, tdtarget)
    mario.update_target()
    action = mario.act(torch.randn(1, 4, 84, 84), 0.1)
    print(tdestimate, tdestimate.shape)
    print(tdtarget, tdtarget.shape)
    print(loss, action)


    



    
        