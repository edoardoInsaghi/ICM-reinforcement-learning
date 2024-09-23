import matplotlib.pyplot as plt
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
        
        self.counter = 0        
        self.memory = []
        self.max_memory = max_memory   
        self.action_space = action_space

        self.gamma = gamma # Reward Discount
        self.epsilon = epsilon # Exploration Rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.warmup = warmup
        self.lr = lr
        self.learn_every = learn_every
        self.batch_size = batch_size
        self.device = device
        self.loss = torch.nn.MSELoss(reduction="mean")


    def cache(self, state, next_state, action, reward, done):
        state = state.clone().detach()
        next_state = next_state.clone().detach()
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
        self.net.fc2.load_state_dict(self.net.fc1.state_dict())
 

    def act(self, state, show_stats=False):
        self.counter += 1
        self.net.eval()
        q = self.net(state, 1)
        if self.counter == self.warmup:
            print("Warmup done")
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_space)
            color = 'g'
        else:
            action = torch.argmax(q).item()
            color = 'r' 

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if show_stats and self.counter > self.warmup and self.counter % 2 == 0:
            plot_stats(q.squeeze().detach().cpu().numpy(), action, color)

        return action
        

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
        self.optimizer1 = optim.Adam([{"params":self.net.blocks.parameters()}, {"params":self.net.fc1.parameters()}], lr=lr)
        self.optimizer2 = optim.Adam([{"params":self.net.blocks.parameters()}, {"params":self.net.fc2.parameters()}], lr=lr)
        

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

        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()

        loss1.backward(retain_graph=True)
        loss2.backward()

        self.optimizer1.step()
        self.optimizer2.step()

        return (q1t.mean().item() + q2t.mean().item()) / 2, [loss1.item(), loss2.item()]


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
                 device="cpu", learn_every=4, warmup=1000, lr=0.00025, beta=0.01,
                 epsilon = 0.15, epsilon_min=0.01, epsilon_decay=0.999997, temperature=1.0, min_temperature=0.1, temperature_decay=0.999997):
        
        super().__init__(action_space, gamma, batch_size, size, max_memory, device, 
                         learn_every, warmup, lr, epsilon, epsilon_min, epsilon_decay)
        
        self.net = Net(4, action_space, size, "ac").to(device)
        self.optimizer1 = optim.Adam(self.net.parameters(), lr=lr)
        self.loss = torch.nn.SmoothL1Loss(reduction="mean")
        # self.optimizer2 = optim.Adam([{"params":self.net.blocks.parameters()}, {"params":self.net.fc2.parameters()}], lr=lr)

        self.temperature = temperature
        self.min_temperature = min_temperature
        self.temperature_decay = temperature_decay
        self.beta = beta

    
    def act(self, state, show_stats=True):
        self.counter += 1
        color = 'r'
        if self.counter == self.warmup:
            print("Warmup done")
        
        self.net.eval()
        state = state.to(self.device)
        p = self.net(state, 1)
        p = torch.softmax(p, dim=1)
        action = torch.multinomial(p, 1).item()

        if show_stats and self.counter > self.warmup and self.counter % 2 == 0:
            plot_stats(p.squeeze().detach().cpu().numpy(), action, color)

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
        value_loss = self.loss(vt, reward + self.gamma * vtnext)

        vtnew = self.net(state, 2)
        vtnextnew = self.net(next_state, 2)
        advantage = (reward + self.gamma * vtnextnew - vtnew)
        log_p = torch.log(p[np.arange(0, self.batch_size), action.squeeze()])
        policy_loss = -log_p.unsqueeze(1) * advantage
        
        a = p[np.arange(0, self.batch_size), action.squeeze()]
        entropy = -torch.sum(p[np.arange(0, self.batch_size), action.squeeze()] * log_p)
        total_loss = policy_loss + value_loss + entropy * self.beta

        #self.optimizer2.zero_grad()
        self.optimizer1.zero_grad()

        #value_loss.mean().backward(retain_graph=True)
        #policy_loss.mean().backward()

        total_loss.mean().backward()
        for param in self.net.parameters():
            param.grad.data.clamp(-1, 1)

        #self.optimizer2.step()
        self.optimizer1.step()

        return vt.mean().item(), [policy_loss.mean().item(), value_loss.mean().item()]


class DUELING_Agent(Agent):
    def __init__(self, action_space, gamma=0.9, batch_size=32, size=84, max_memory=int(1e4), 
            device="cpu", learn_every=4, warmup=1000, lr=0.00025,
            epsilon = 0.15, epsilon_min=0.01, epsilon_decay=0.9):

        super().__init__(action_space, gamma, batch_size, size, max_memory, device, 
            learn_every, warmup, lr, epsilon, epsilon_min, epsilon_decay)
        
        self.net = Net(4, action_space, size, "dueling").to(device)
        self.optimizer1 = optim.Adam([{"params":self.net.blocks.parameters()}, {"params":self.net.fc1.parameters()}, {"params":self.net.fc2.parameters()}], lr=lr)
        self.optimizer2 = optim.Adam([{"params":self.net.blocks.parameters()}, {"params":self.net.fc1_2.parameters()}, {"params":self.net.fc2_2.parameters()}], lr=lr)
        self.scheduler1 = torch.optim.lr_scheduler.CyclicLR(self.optimizer1, base_lr=1e-5, max_lr=1e-3, step_size_up=2000, cycle_momentum=False)
        self.scheduler2 = torch.optim.lr_scheduler.CyclicLR(self.optimizer2, base_lr=1e-5, max_lr=1e-3, step_size_up=2000, cycle_momentum=False)
            

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

        # print(f"Q1 from network 1: {q1.mean().item()}")
        # print(f"Q2 from network 2: {q2.mean().item()}")

        loss1 = self.loss(target2, q1t)
        loss2 = self.loss(target1, q2t) 

        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()

        loss1.backward(retain_graph=True)
        loss2.backward()

        self.optimizer1.step()
        self.optimizer2.step()

        self.scheduler1.step()
        self.scheduler2.step()

        return [q1t.mean().item(), q2t.mean().item()], [loss1.item(), loss2.item()]


    def act(self, state):
        self.counter += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if self.counter == self.warmup:
            print("Warmup done")
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space)
        else:
            self.net.eval()
            state = state.to(self.device)
            if np.random.rand() < 0.5:
                return self.net(state, 1).argmax().item()
            else:
                return self.net(state, 2).argmax().item()
        

    def learn(self):
        if self.counter < self.warmup:
            return None, None
        
        if self.counter % self.learn_every != 0:
            return None, None

        state, next_state, action, reward, done = self.sample_from_memory()

        mean_q, loss = self.update_q(state, next_state, action, reward)

        return mean_q, loss 
    


def plot_stats(q, action, color, v=None):
    plt.clf()
    bars = plt.bar(range(len(q)), q, width=0.4)
    bars[action].set_color(color)
    if v is not None:
        plt.bar(range(len(v)), v, width=0.4, color="g")
    plt.xlabel('Actions')
    plt.ylabel('Q-values')
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)
    plt.show(block=False)




"""
class REINFORCE_Agent(Agent):

    def __init__(self, action_space, gamma=0.9, batch_size=32, size=84, max_memory=int(1e4), 
                 device="cpu", learn_every=4, warmup=1000, lr=0.00025,
                 epsilon = 0.15, epsilon_min=0.01, epsilon_decay=0.999997):
        
        super().__init__(action_space, gamma, batch_size, size, max_memory, device, 
                         learn_every, warmup, lr, epsilon, epsilon_min, epsilon_decay)
        
        self.net = Net(4, action_space, size, "reinforce").to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.max_memory = np.inf
        self.baseline = 0.0

    
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

        cum_rewards = 0
        total_loss = 0
        n = len(self.memory)
        for batch in range(0, n, self.batch_size):
            states = torch.stack([self.memory[i]["State"] for i in range(batch, min(batch+self.batch_size, n))]).to(self.device)
            actions = torch.stack([self.memory[i]["Action"] for i in range(batch, min(batch+self.batch_size, n))]).to(self.device)
            rewards = torch.stack([self.memory[i]["Reward"] for i in range(batch, min(batch+self.batch_size, n))]).to(self.device)

            cum_rewards += rewards.sum().item()

            self.net.train()
            p = self.net(states, 1)
            p = torch.softmax(p, dim=1)
            log_p = torch.log(p[np.arange(0, len(actions.squeeze())), actions.squeeze()])
            loss = -log_p.unsqueeze(1) * (rewards - self.baseline)
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            total_loss += loss.mean().item()

        self.baseline = cum_rewards / n

        return cum_rewards, total_loss
"""