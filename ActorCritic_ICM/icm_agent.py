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
        
        self.reverse_net = Reverse_Dynamics_Module(4, action_space).to(device)
        self.forward_net = ForwardModule(action_space).to(device)
        self.ac_net = AC_NET(4, action_space).to(device)
        self.optimizer = optim.Adam(list(self.reverse_net.parameters()) + 
                                    list(self.forward_net.parameters()) + 
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
        state = self.reverse_net.get_latent_state(state).unsqueeze(0)
        
        for _ in range(local_steps):
            
            action = self.act(state, height=None, ax=ax, show_stats=show_stats)
            logits= self.ac_net(state, model=1)
            value = self.ac_net(state, model=2)
            policy = torch.softmax(logits, dim=1)
            log_policy = torch.log_softmax(logits, dim=1)
            entropy = -(policy * log_policy).sum(1, keepdim=True)
            
            state, reward, self.done, self.info = env.step(action)
            state = torch.tensor(np.asarray(state) / 255.0, dtype=torch.float32, device=device).unsqueeze(0).to(self.device)
            self.reverse_net.get_latent_state(state)
            
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
        self.reverse_net.train()
        self.forward_net.train()

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
            a_hat = self.reverse_net(state, next_state)
            latent_state = self.reverse_net.get_latent_state(state)
            latent_next_state = self.reverse_net.get_latent_state(next_state)
            forward_out = self.forward_net(latent_state, a)
            
            reverse_loss = torch.abs(a - a_hat).sum()
            forward_loss = ((forward_out - latent_next_state) ** 2).sum()
            
            actor_loss = actor_loss + (log_policy * gae)  
            critic_loss = critic_loss + ((R - value) ** 2 / 2)
            entropy_loss = entropy_loss + entropy

        total_loss = -actor_loss + critic_loss - self.beta * entropy_loss + reverse_loss + forward_loss
        self.optimizer.zero_grad() 
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac_net.parameters(), 0.5)
        self.optimizer.step()
        
        return next_value.item(), total_loss.item(), forward_loss.item()
            
        
        