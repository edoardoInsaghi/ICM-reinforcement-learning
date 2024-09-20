from cv2 import log
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Modules import *
from abc import abstractmethod


class Agent():

    def __init__(self, action_space, gamma=0.9, batch_size=32, size=84, max_memory=int(1e5), 
                 device="cpu", learn_every=4, warmup=1000, lr=0.00025,
                 epsilon = 0.15, epsilon_min=0.01, epsilon_decay=0.999997):
        
        self.lr = lr
        self.counter = 0
        self.learn_every = learn_every
        self.warmup = warmup
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = device
        self.memory = []
        self.max_memory = max_memory
        self.batch_size = batch_size
        self.loss = torch.nn.MSELoss(reduction="mean")


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
        n = n if n is not None else self.batch_size - 10
        idx = np.random.choice(len(self.memory), n, replace=False)
        samples = [self.memory[i] for i in idx] + [self.memory[-i] for i in range(1, 11)]
        states = torch.stack([s["State"] for s in samples]).to(self.device) 
        next_states = torch.stack([s["Next_state"] for s in samples]).to(self.device)
        actions = torch.stack([s["Action"] for s in samples]).to(self.device)
        rewards = torch.stack([s["Reward"] for s in samples]).to(self.device)
        dones = torch.stack([s["Done"] for s in samples]).to(self.device)

        return states, next_states, actions, rewards, dones
    
    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def learn(self):
        pass



class FDQN_Agent(Agent):

    def __init__(self, action_space, gamma=0.9, batch_size=32, size=84, max_memory=int(1e4), 
                 device="cpu", learn_every=4, warmup=1000, lr=0.00025,
                 epsilon = 0.15, epsilon_min=0.01, epsilon_decay=0.999997, sync_every=1000):
        
        super().__init__(action_space, gamma, batch_size, size, max_memory, device, 
                         learn_every, warmup, lr, epsilon, epsilon_min, epsilon_decay)

        self.sync_every = sync_every
        self.net = Net(4, action_space, size, "fdqn").to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)


    def td_estimate(self, state, action):
        self.net.eval()
        q = self.net(state, 1)[np.arange(0, self.batch_size), action]
        return q
    

    @torch.no_grad()
    def td_target(self, reward, next_state):
        self.net.eval()
        q_target = self.net(next_state, 2)
        q_target = torch.max(q_target, dim=1).values
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
 

    def act(self, state):
        self.counter += 1
        if self.counter == self.warmup:
            print("Warmup done")
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space)
        else:
            self.net.eval()
            state = state.to(self.device)
            q = self.net(state, 1)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            return torch.argmax(q).item()
        

    def learn(self):
        if self.counter % self.sync_every == 0:
            self.update_target()

        if self.counter % self.learn_every != 0:
            return None, None
        
        if self.counter < self.warmup:
            return None, None
        
        state, next_state, action, reward, done = self.sample_from_memory()

        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state)
        loss = self.update_q_online(td_est, td_tgt)

        return td_est.mean().item(), loss
    


class DDQN_Agent(Agent):

    def __init__(self, action_space, gamma=0.9, batch_size=32, size=84, max_memory=int(1e4), 
                device="cpu", learn_every=4, warmup=1000, lr=0.00025,
                epsilon = 0.15, epsilon_min=0.01, epsilon_decay=0.999997):
    
        super().__init__(action_space, gamma, batch_size, size, max_memory, device, 
                         learn_every, warmup, lr, epsilon, epsilon_min, epsilon_decay)
        

        self.net = Net(4, action_space, size, "ddqn").to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        

    def update_q(self, state, next_state, action, reward):
        self.net.train()

        # Select best action from both networks
        q1 = self.net(next_state, 1)
        action1 = torch.argmax(q1, dim=1)
        q2 = self.net(next_state, 2)
        action2 = torch.argmax(q2, dim=1)

        target1 = self.gamma * self.net(state, 2)[np.arange(0, self.batch_size), action1] + reward
        target2 = self.gamma * self.net(state, 1)[np.arange(0, self.batch_size), action2] + reward

        q1t = self.net(state, 1)[np.arange(0, self.batch_size), action]
        q2t = self.net(state, 2)[np.arange(0, self.batch_size), action]

        loss1 = self.loss(target2, q1t)
        loss2 = self.loss(target1, q2t) 

        if np.random.rand() < 0.5:
            self.optimizer.zero_grad()
            loss1.backward()
            self.optimizer.step()
        else:
            self.optimizer.zero_grad()
            loss2.backward()
            self.optimizer.step()

        return (q1t.mean().item() + q2t.mean().item()) / 2, (loss1.item() + loss2.item()) / 2


    def act(self, state):
        self.counter += 1
        if self.counter == self.warmup:
            print("Warmup done")
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space)
        else:
            self.net.eval()
            state = state.to(self.device)
            if np.random.rand() < 0.5:
                q = self.net(state, 1)
            else:
                q = self.net(state, 2)

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            return torch.argmax(q).item()
        

    def learn(self):
        if self.counter < self.warmup:
            return None, None
        
        if self.counter % self.learn_every != 0:
            return None, None

        state, next_state, action, reward, done = self.sample_from_memory()

        mean_q, loss = self.update_q(state, next_state, action, reward)

        return mean_q, loss 
    



class AC_Agent(Agent):
    def __init__(self, action_space, gamma=0.9, batch_size=32, size=84, max_memory=int(1e4), 
                 device="cpu", learn_every=4, warmup=1000, lr=0.00025,
                 epsilon = 0.15, epsilon_min=0.01, epsilon_decay=0.999997):
        
        super().__init__(action_space, gamma, batch_size, size, max_memory, device, 
                         learn_every, warmup, lr, epsilon, epsilon_min, epsilon_decay)
        
        self.net = Net(4, action_space, size, "ac").to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        
    
    def act(self, state):
        self.counter += 1
        if self.counter == self.warmup:
            print("Warmup done")
        
        self.net.eval()
        state = state.to(self.device)
        p = self.net(state, 1)
        p = torch.softmax(p, dim=1)
        action = torch.multinomial(p, 1).item()
        return action
    

    def learn(self):
        if self.counter < self.warmup:
            return None, None
        
        if self.counter % self.learn_every != 0:
            return None, None

        state, next_state, action, reward, done = self.sample_from_memory()

        self.net.train()
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        p = self.net(state, 1)
        p = torch.softmax(p, dim=1)
        vt = self.net(state, 2)
        vtnext = self.net(next_state, 2)
        advantage = (reward + self.gamma * vtnext - vt)
        log_p = torch.log(p[np.arange(0, self.batch_size), action.squeeze()])
        policy_loss = -log_p.unsqueeze(1) * advantage
        value_loss = self.loss(vt, reward + self.gamma * vtnext)

        self.optimizer.zero_grad()
        policy_loss.mean().backward(retain_graph=True)
        value_loss.backward()
        self.optimizer.step()
        
        return vt.mean().item(), [policy_loss.mean().item() + value_loss.mean().item()]


