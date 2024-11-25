import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import numpy as np
from ActorCritic_ICM.ac_net import *
from torch.distributions import Categorical 



class AC_Agent():
    def __init__(self, action_space, args, device):
        
        self.action_space = action_space
        self.gamma = args.gamma 
        self.lr = args.lr
        
        self.device = device
        self.net = AC_NET(4, action_space).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=args.lr)

        self.beta = args.beta
        self.cluster = args.cluster
        self.log_policies = []
        self.values = []
        self.rewards = []
        self.entropies = []
        self.last_state = None
        self.done = False
        self.info = None
        self.counter = 0
        self.v = []

    
    def plot_stats(self, q, action, color, v, ax):
        ax1, ax2 = ax
        self.v.append(v)
        if len(self.v) > 100:
            self.v = self.v[1:]
        ax1.clear()
        ax2.clear()
        bars1 = ax1.bar(range(len(q)), q, width=0.4)
        bars1[action].set_color(color)
        ax1.set_xlabel('Actions')
        ax1.set_ylabel('Q-values')
        ax1.set_ylim((0, 1))
        ax2.plot(self.v, label="Value")
        ax2.set_ylim((-5, 15))
        plt.draw()
        plt.pause(0.001)
        plt.show(block=False)


    def act(self, state, show_stats=True, ax=None):
        self.counter += 1
        color = 'r'
        
        with torch.no_grad():
            state = state.to(self.device)
            logits = self.net(state, model=1)
            p = torch.softmax(logits, dim=1)

            v = self.net(state, model=2).item()
            
            m = Categorical(p)
            action = m.sample().item()
        
        if show_stats and self.counter % 1 == 0:
            self.plot_stats(p.squeeze().detach().cpu().numpy(), action, color, v, ax)
            
        
        return action
    
    
    def get_experience(self, env, state, local_steps, device, show_stats=True, ax=None):
        self.log_policies = []
        self.values = []
        self.rewards = []
        self.entropies = []
        self.last_state = None
        self.done = None
        
        for _ in range(local_steps):
            
            action = self.act(state, show_stats=show_stats, ax=ax)
            logits= self.net(state, model=1)
            value = self.net(state, model=2)
            policy = torch.softmax(logits, dim=1)
            log_policy = torch.log_softmax(logits, dim=1)
            entropy = -(policy * log_policy).sum(1, keepdim=True)
            
            state, reward, self.done, self.info = env.step(action)
            state = torch.tensor(np.asarray(state) / 255.0, dtype=torch.float32, device=device).unsqueeze(0).to(self.device)
            
            self.values.append(value)
            self.log_policies.append(log_policy[0, action])
            self.rewards.append(reward)
            self.entropies.append(entropy)

            if self.done:
                break
            
            if show_stats:
                env.render()
            
        self.last_state = state
        return self.done, self.last_state
        
    
    def learn(self):
        self.net.train()

        actor_loss = 0.0
        critic_loss = 0.0
        entropy_loss = 0.0
        gae = torch.zeros((1, 1), dtype=torch.float).to(self.device)

        if self.done:
            R = torch.zeros(1, 1, device=self.device)
        else:
            R = self.net(self.last_state, model=2)
                
        R = R.to(self.device)
        next_value = R
        
        for value, log_policy, reward, entropy in list(zip(self.values, self.log_policies, self.rewards, self.entropies))[::-1]:
            gae = gae * self.gamma 
            gae = gae + reward + self.gamma * next_value.detach() - value.detach()
            next_value = value
            R = R * self.gamma + reward
            actor_loss = actor_loss + (log_policy * gae)  
            critic_loss = critic_loss + ((R - value) ** 2 / 2)
            entropy_loss = entropy_loss + entropy

        total_loss = -actor_loss + critic_loss - self.beta * entropy_loss
        self.optimizer.zero_grad() 
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.optimizer.step()
        
        return next_value.item(), total_loss.item()
            
            
    def learn2(self):
        self.net.train()

        actor_loss = 0.0
        critic_loss = 0.0
        entropy_loss = 0.0
        gae = torch.zeros((1, 1), dtype=torch.float).to(self.device)

        if self.done:
            R = torch.zeros(1, 1, device=self.device)
        else:
            R = self.net(self.last_state, model=2)
                
        R = R.to(self.device)
        next_value = R
        
        for value, log_policy, reward, entropy in list(zip(self.values, self.log_policies, self.rewards, self.entropies))[::-1]:
            gae = gae * self.gamma 
            gae = gae + reward + self.gamma * next_value.detach() - value.detach()
            next_value = value
            R = R * self.gamma + reward
            actor_loss = actor_loss + (log_policy * gae)  
            critic_loss = critic_loss + ((R - value) ** 2 / 2)
            entropy_loss = entropy_loss + entropy

        total_loss = -actor_loss + critic_loss - self.beta * entropy_loss
        
        return total_loss
