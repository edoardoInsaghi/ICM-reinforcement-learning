import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ActorCritic_ICM.ac_net import AC_NET
from ActorCritic_ICM.icm_net import Reverse_Dynamics_Module, ForwardModule
from abc import abstractmethod
from torch.distributions import Categorical 


class ICM_Agent():
    def __init__(self, action_space, args, device):
        
        self.forward_net = ForwardModule(action_space).to(device)
        self.ac_net = AC_NET(4, action_space).to(device)
        self.reverse = Reverse_Dynamics_Module(4, action_space).to(device)
        self.optimizer = optim.Adam(list(self.forward_net.parameters()) + 
                                    list(self.ac_net.parameters()) + list(self.reverse.parameters()), 
                                    lr=args.lr)
        self.action_space = action_space
        self.gamma = args.gamma
        self.beta = args.beta
        self.log_policies = []
        self.values = []
        self.rewards = []
        self.entropies = []
        self.actions = []
        self.states = []
        self.last_state = None
        self.done = False
        self.cluster = args.cluster
        self.info = None
        self.counter = 0
        self.device = device
        
        
    
    def plot_stats(self, q, action, color, v=None, ax=None):
        plt.clf()
        bars = plt.bar(range(len(q)), q, width=0.4)
        bars[action].set_color(color)
        plt.xlabel('Actions')
        plt.ylabel('Policy-values')
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)
        plt.show(block=False)   
    
        
        
    def act(self, state, height, show_stats=True, ax=None):
        self.counter += 1
        color = 'r'
        
        with torch.no_grad():
            state = state.to(self.device)
            logits= self.ac_net(state, model=1)
            p = torch.softmax(logits, dim=1)
            m = Categorical(p)
            action = m.sample().item()
        
        if show_stats and self.counter % 2 == 0:
            self.plot_stats(p.squeeze().detach().cpu().numpy(), action, color)
            
        
        return action
    
    
    def get_experience(self, env, state, local_steps, device, show_stats=True, ax=None):
        self.log_policies = []
        self.values = []
        self.rewards = []
        self.entropies = []
        self.actions = []
        self.states = []
        self.last_state = None
        self.done = None
        
        for _ in range(local_steps):
            
            action = self.act(state, height=None, ax=ax, show_stats=show_stats)
            logits= self.ac_net(state, model=1)
            value = self.ac_net(state, model=2)
            policy = torch.softmax(logits, dim=1)
            log_policy = torch.log_softmax(logits, dim=1)
            entropy = -(policy * log_policy).sum(1, keepdim=True)
            
            a = torch.zeros(1, self.action_space).to(self.device)
            a[0, action] = 1.0
            latent_state = self.reverse.get_latent_state(state)
            
            next_state, reward, self.done, self.info = env.step(action)
            next_state = torch.tensor(np.asarray(next_state) / 255.0, dtype=torch.float32, device=device).unsqueeze(0).to(self.device)

            
            forward_out = self.forward_net(latent_state, a)
            latent_next_state = self.reverse.get_latent_state(next_state)
            reward += torch.nn.functional.mse_loss(forward_out, latent_next_state, reduction='sum') 
            
            self.states.append(next_state)
            self.actions.append(action)
            self.values.append(value)
            self.log_policies.append(log_policy[0, action])
            self.rewards.append(reward)
            self.entropies.append(entropy)
            
            state = next_state

            if self.done:
                break
            
            if show_stats:
                env.render()
            
        self.last_state = state
        return self.done, self.last_state
        
    
    def learn(self):
        self.ac_net.train()
        self.reverse.train()
        self.forward_net.train()
        
        if len(self.states) == 1:
            return 0, 0
        
        forward_loss = 0.0
        reverse_loss = 0.0
        actor_loss = 0.0
        critic_loss = 0.0
        entropy_loss = 0.0
        gae = torch.zeros((1, 1), dtype=torch.float).to(self.device)

        if self.done:
            R = torch.zeros(1, 1, device=self.device)
        else:
            R = self.ac_net(self.last_state, model=2)
                
        R = R.to(self.device)
        next_value = R
        
        for i, (value, action, state, log_policy, reward, entropy) in enumerate(list(zip(self.values, self.actions, self.states, self.log_policies, self.rewards, self.entropies))[-2::-1]):
            gae = gae * self.gamma 
            gae = gae + reward + self.gamma * next_value.detach() - value.detach()
            next_value = value
            R = R * self.gamma + reward
            
            next_state = self.states[i+1]
            a = torch.zeros(1, self.action_space).to(self.device)
            a[0, action] = 1.0
            latent_state = self.reverse.get_latent_state(state)
            latent_next_state = self.reverse.get_latent_state(next_state)
            a_hat = self.reverse(latent_state, latent_next_state)
            forward_out = self.forward_net(latent_state, a)
            
            reverse_loss = reverse_loss + torch.nn.functional.cross_entropy(a_hat, torch.tensor(action).unsqueeze(0).to(self.device))
            forward_loss = forward_loss + torch.nn.functional.mse_loss(forward_out, latent_next_state, reduction='sum') * 1/2
            
            actor_loss = actor_loss + (log_policy * gae)  
            critic_loss = critic_loss + ((R - value) ** 2 / 2)
            entropy_loss = entropy_loss + entropy

        total_loss = 0.1 * (-actor_loss + critic_loss) + 0.8 * reverse_loss + 0.2 * forward_loss 

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac_net.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.forward_net.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.reverse.parameters(), 0.5)
        self.optimizer.step()
        
        return total_loss.item(), [reverse_loss.item(), actor_loss.item(), critic_loss.item(), forward_loss.item()]





'''
class ICM_Agent():
    def __init__(self, action_space, args, device):
        
        self.forward_net = ForwardModule(action_space).to(device)
        self.ac_net = AC_NET(4, action_space, latent=True).to(device)
        self.optimizer = optim.Adam(list(self.forward_net.parameters()) + 
                                    list(self.ac_net.parameters()), 
                                    lr=args.lr
                        )
        self.action_space = action_space
        self.gamma = args.gamma
        self.beta = args.beta
        self.log_policies = []
        self.values = []
        self.rewards = []
        self.entropies = []
        self.actions = []
        self.states = []
        self.last_state = None
        self.done = False
        self.cluster = args.cluster
        self.info = None
        self.counter = 0
        self.device = device
        
        
    
    def plot_stats(self, q, action, color, v=None, ax=None):
        plt.clf()
        bars = plt.bar(range(len(q)), q, width=0.4)
        bars[action].set_color(color)
        plt.xlabel('Actions')
        plt.ylabel('Policy-values')
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)
        plt.show(block=False)   
    
        
        
    def act(self, state, height, show_stats=True, ax=None):
        self.counter += 1
        color = 'r'
        
        with torch.no_grad():
            state = state.to(self.device)
            logits= self.ac_net(state, model=1)
            p = torch.softmax(logits, dim=1)
            m = Categorical(p)
            action = m.sample().item()
        
        if show_stats and self.counter % 2 == 0:
            self.plot_stats(p.squeeze().detach().cpu().numpy(), action, color)
            
        
        return action
    
    
    def get_experience(self, env, state, local_steps, device, show_stats=True, ax=None):
        self.log_policies = []
        self.values = []
        self.rewards = []
        self.entropies = []
        self.actions = []
        self.states = []
        self.last_state = None
        self.done = None
        
        for _ in range(local_steps):
            
            action = self.act(state, height=None, ax=ax, show_stats=show_stats)
            logits= self.ac_net(state, model=1)
            value = self.ac_net(state, model=2)
            policy = torch.softmax(logits, dim=1)
            log_policy = torch.log_softmax(logits, dim=1)
            entropy = -(policy * log_policy).sum(1, keepdim=True)
            
            state, reward, self.done, self.info = env.step(action)
            state = torch.tensor(np.asarray(state) / 255.0, dtype=torch.float32, device=device).unsqueeze(0).to(self.device)
            self.ac_net.reverse.get_latent_state(state)
            
            self.states.append(state)
            self.actions.append(action)
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
        self.ac_net.train()
        self.ac_net.reverse.train()
        self.forward_net.train()
        
        if len(self.states) == 1:
            return 0, 0
        
        forward_loss = 0.0
        reverse_loss = 0.0
        actor_loss = 0.0
        critic_loss = 0.0
        entropy_loss = 0.0
        gae = torch.zeros((1, 1), dtype=torch.float).to(self.device)

        if self.done:
            R = torch.zeros(1, 1, device=self.device)
        else:
            R = self.ac_net(self.last_state, model=2)
                
        R = R.to(self.device)
        next_value = R
        
        for i, (value, action, state, log_policy, reward, entropy) in enumerate(list(zip(self.values, self.actions, self.states, self.log_policies, self.rewards, self.entropies))[-2::-1]):
            gae = gae * self.gamma 
            gae = gae + reward + self.gamma * next_value.detach() - value.detach()
            next_value = value
            R = R * self.gamma + reward
            
            next_state = self.states[i+1]
            a = torch.zeros(1, self.action_space).to(self.device)
            a[0, action] = 1.0
            a_hat = self.ac_net.reverse(state, next_state)
            latent_state = self.ac_net.reverse.get_latent_state(state)
            latent_next_state = self.ac_net.reverse.get_latent_state(next_state)
            forward_out = self.forward_net(latent_state, a)
            
            reverse_loss = torch.abs(a - a_hat).sum()
            forward_loss = ((forward_out - latent_next_state) ** 2).sum()
            
            actor_loss = actor_loss + (log_policy * gae)  
            critic_loss = critic_loss + ((R - value) ** 2 / 2)
            entropy_loss = entropy_loss + entropy

        total_loss = -actor_loss + critic_loss + reverse_loss + forward_loss

        self.optimizer.zero_grad() 
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac_net.parameters(), 0.5)
        self.optimizer.step()
        
        return next_value.item(), total_loss.item()
'''   
        
