import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Modules import *
from abc import abstractmethod
from torch.distributions import Categorical


class Agent():

    def __init__(self, action_space, gamma=0.9, batch_size=32, size=84, max_memory=int(1e5), 
                 device="cpu", learn_every=4, warmup=1000, lr=0.00025,
                 epsilon = 0.15, epsilon_min=0.01, epsilon_decay=0.999997, ckpt=None, learn_states=False):
        
        self.counter = 0        
        self.memory = []
        self.max_memory = max_memory   
        self.action_space = action_space
        self.learn_states = learn_states

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
    

    def mask_jumps(self, action):
        map = {0:0, 1:1, 2:1, 3:3, 4:3, 5:0, 6:6}
        return map[action]
    

    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def plot_stats(self, q, action, color, v=None, ax=None):
        pass



class FDQN_Agent(Agent):

    def __init__(self, action_space, gamma=0.9, batch_size=32, size=84, max_memory=int(1e4), 
                 device="cpu", learn_every=4, warmup=1000, lr=0.00025,
                 epsilon = 0.15, epsilon_min=0.01, epsilon_decay=0.999997, sync_every=1000, ckpt=None, learn_states=False):
        
        super().__init__(action_space, gamma, batch_size, size, max_memory, device, 
                         learn_every, warmup, lr, epsilon, epsilon_min, epsilon_decay, ckpt, learn_states)

        self.algorithm = "fdqn"
        self.sync_every = sync_every
        self.net = Net(4, action_space, size, "fdqn", learn_states).to(device)
        if ckpt is not None:
            self.net.load_state_dict(torch.load(ckpt, map_location=device))
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.h = 0
        self.cross_loss = nn.CrossEntropyLoss()


    def td_estimate(self, state, action):
        self.net.eval()
        q = self.net(state, 1)[np.arange(0, self.batch_size), action]
        return q
    

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        self.net.eval()
        q_target = self.net(next_state, 2)
        q_target = torch.max(q_target, dim=1).values
        return reward + self.gamma * q_target * ~done


    def update_q_online(self, td_estimate, td_target, state=None, next_state=None, action=None):
        self.net.train()
        if state is not None and next_state is not None and action is not None:
            ahat = self.net.RDM(state, next_state)
            reverse_loss = self.cross_loss(ahat, action.squeeze(1))
            loss = self.loss(td_estimate, td_target) + reverse_loss
        else:
            loss = self.loss(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    

    def update_target(self):
        self.net.fc2.load_state_dict(self.net.fc1.state_dict())
 
    @torch.no_grad()
    def act(self, state, height, show_stats=True, ax=None):
        self.counter += 1

        self.net.eval()
        q = self.net(state, 1)
        
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_space)
            color = 'g'
        else:
            action = torch.argmax(q).item()
            if height < self.h:
                action = self.mask_jumps(action)
            color = 'r' 

        self.h = height
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if show_stats and self.counter > self.warmup and self.counter % 2 == 0:
            self.plot_stats(q.squeeze().detach().cpu().numpy(), action, color)

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
        td_tgt = self.td_target(reward, next_state, done)
        if self.learn_states:
            loss = self.update_q_online(td_est, td_tgt, state, next_state, action)
        else:
            loss = self.update_q_online(td_est, td_tgt)

        return [td_est.mean().item(), None], [loss, None]
    

    def plot_stats(self, q, action, color, v=None, ax=None):
        plt.clf()
        bars = plt.bar(range(len(q)), q, width=0.4)
        bars[action].set_color(color)
        plt.xlabel('Actions')
        plt.ylabel('Q-values')
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)
        plt.show(block=False)        
    


class DDQN_Agent(Agent):

    def __init__(self, action_space, gamma=0.9, batch_size=32, size=84, max_memory=int(1e4), 
                device="cpu", learn_every=4, warmup=1000, lr=0.00025,
                epsilon = 0.15, epsilon_min=0.01, epsilon_decay=0.999997, ckpt=None, learn_states=False):
    
        super().__init__(action_space, gamma, batch_size, size, max_memory, device, 
                         learn_every, warmup, lr, epsilon, epsilon_min, epsilon_decay, ckpt, learn_states)
        

        self.algorithm = "ddqn"
        self.net = Net(4, action_space, size, "ddqn").to(device)
        if ckpt is not None:
            self.net.load_state_dict(torch.load(ckpt, map_location=device))
        self.optimizer1 = optim.Adam([{"params":self.net.blocks.parameters()}, {"params":self.net.fc1.parameters()}], lr=lr)
        self.optimizer2 = optim.Adam([{"params":self.net.blocks.parameters()}, {"params":self.net.fc2.parameters()}], lr=lr)
        self.h = 0
        

    def update_q(self, state, next_state, action, reward, done):
        self.net.train()

        # Select best action from both networks
        q1 = self.net(next_state, 1)
        action1 = torch.argmax(q1, dim=1)
        q2 = self.net(next_state, 2)
        action2 = torch.argmax(q2, dim=1)

        target1 = self.gamma * self.net(state, 2)[np.arange(0, self.batch_size), action1] * ~done + reward
        target2 = self.gamma * self.net(state, 1)[np.arange(0, self.batch_size), action2] * ~done + reward

        q1t = self.net(state, 1)[np.arange(0, self.batch_size), action]
        q2t = self.net(state, 2)[np.arange(0, self.batch_size), action]

        loss1 = self.loss(target2, q1t)
        loss2 = self.loss(target1, q2t) 

        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()

        loss1.backward(retain_graph=True)
        loss2.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)

        self.optimizer1.step()
        self.optimizer2.step()

        return [q1t.mean().item(), q2t.mean().item()], [loss1.item(), loss2.item()]
    

    @torch.no_grad()    
    def act(self, state, height, show_stats=True, ax=None):
        self.counter += 1
        
        self.net.eval()
        
        q1 = self.net(state, 1)
        q2 = self.net(state, 2)
        q = q1 if np.random.rand() < 0.5 else q2
                
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_space)
            color = 'g'
        else: 
            action = torch.argmax(q).item()
            if height < self.h:
                action = self.mask_jumps(action)
            color = 'r' 

        self.h = height
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if show_stats and self.counter > self.warmup and self.counter % 2 == 0:
            self.plot_stats([q1.squeeze().detach().cpu().numpy(), q2.squeeze().detach().cpu().numpy()], action, color, ax=ax)
            
        return action
        

    def learn(self):
        if self.counter < self.warmup:
            return None, None
        
        if self.counter % self.learn_every != 0:
            return None, None

        state, next_state, action, reward, done = self.sample_from_memory()
        
        mean_q, loss = self.update_q(state, next_state, action, reward, done)

        return mean_q, loss
    

    def plot_stats(self, q, action, color, v=None, ax=None):
        assert ax is not None
        q1, q2 = q
        ax1, ax2 = ax
        ax1.clear()
        ax2.clear()
        bars1 = ax1.bar(range(len(q1)), q1, width=0.4)
        bars1[action].set_color(color)
        ax1.set_xlabel('Actions')
        ax1.set_ylabel('Q-values')
        bars2 = ax2.bar(range(len(q2)), q2, width=0.4)
        bars2[action].set_color(color)
        ax2.set_xlabel('Actions')
        plt.draw()
        plt.pause(0.001)
        plt.show(block=False)
    
    
        
class AC_Agent(Agent):
    def __init__(self, action_space, gamma=0.9, batch_size=32, size=84, max_memory=int(1e4), 
                 device="cpu", learn_every=4, warmup=1000, lr=0.00025, beta=0.01,
                 epsilon = 0.15, epsilon_min=0.01, epsilon_decay=0.999997, temperature=1.0, min_temperature=0.1, temperature_decay=0.999997, ckpt=None):
        
        super().__init__(action_space, gamma, batch_size, size, max_memory, device, 
                         learn_every, warmup, lr, epsilon, epsilon_min, epsilon_decay, ckpt)
        
        self.net = Net(4, action_space, size, "ac").to(device)
        self.optimizer1 = optim.Adam(self.net.parameters(), lr=lr)
        self.loss = torch.nn.SmoothL1Loss(reduction="mean")
        # self.optimizer2 = optim.Adam([{"params":self.net.blocks.parameters()}, {"params":self.net.fc2.parameters()}], lr=lr)

        self.temperature = temperature
        self.min_temperature = min_temperature
        self.temperature_decay = temperature_decay
        self.beta = beta
        self.h = 0
    
    
    def act(self, state, height, show_stats=True):
        self.counter += 1
        color = 'r'
        self.net.eval()
        state = state.to(self.device)
        q = self.net(state, 1)
        q_ = None
        if self.h > height:
            q_ = q * torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]).to(self.device)
        self.h = height
        
        if q_ is not None:
            p = torch.softmax(q_, dim=1)
        else:
            p = torch.softmax(q, dim=1)
             
        action = torch.multinomial(p, 1).item()
        color = 'r'

        if show_stats and self.counter > self.warmup and self.counter % 2 == 0:
            plot_stats(q.squeeze().detach().cpu().numpy(), action, color)

        return action
    
    
    def act2(self, state, height, show_stats=True):
        self.counter += 1
        color = 'r'
        self.net.eval()
        state = state.to(self.device)
        q = self.net(state, 1)
        q_ = None
        if self.h > height:
            q_ = q * torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]).to(self.device)
        self.h = height
        
        if q_ is not None:
            p = torch.softmax(q_, dim=1)
        else:
            p = torch.softmax(q, dim=1)
             
        action = torch.multinomial(p, 1).item()
        color = 'r'

        if show_stats and self.counter > self.warmup and self.counter % 2 == 0:
            plot_stats(p.squeeze().detach().cpu().numpy(), action, color)

        return action, q
    
    

    def update(self, values, log_policies, rewards, entropies):
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        next_value = values[-1]

        for value, log_policy, reward, entropy in list(zip(values, log_policies, rewards, entropies))[-2::-1]:
            advantage = reward + self.gamma * next_value.detach() - value.detach()
            next_value = value
            actor_loss = actor_loss + log_policy * advantage
            critic_loss = critic_loss + (advantage) ** 2 
            entropy_loss = entropy_loss + entropy

        total_loss = -actor_loss + critic_loss - self.beta * entropy_loss
        self.optimizer1.zero_grad()
        total_loss.backward()
        
        return next_value, total_loss
            
 
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
        epsilon = 1e-8
        p = torch.clamp(p, epsilon, 1.0 - epsilon)

        vt = self.net(state, 2)
        vtnext = self.net(next_state, 2)
        value_loss = self.loss(vt, reward + self.gamma * vtnext)

        log_policy = torch.log_softmax(p, dim=1)
        entropy = -(p * log_policy).sum(1, keepdim=True).mean()
            
        advantage = (reward + self.gamma * vtnext - vt).detach()
        log_p = torch.log(p[np.arange(0, self.batch_size), action.squeeze()])
        policy_loss = -log_p.unsqueeze(1) * advantage - self.beta * entropy
        
        #total_loss = policy_loss.mean() + value_loss #- entropy * self.beta

        #self.optimizer2.zero_grad()
        self.optimizer1.zero_grad()

        #value_loss.mean().backward(retain_graph=True)
        #policy_loss.mean().backward()
        value_loss.backward(retain_graph=True)
        policy_loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)

        #self.optimizer2.step()
        self.optimizer1.step()

        return vt.mean().item(), [policy_loss.mean().item(), value_loss.mean().item()]
    
    def plot_stats(self, q, action, color, v=None, ax=None):
        pass


class DUELING_Agent(Agent):
    def __init__(self, action_space, gamma=0.9, batch_size=32, size=84, max_memory=int(1e4), 
            device="cpu", learn_every=4, warmup=1000, lr=0.00025,
            epsilon = 0.15, epsilon_min=0.01, epsilon_decay=0.9, ckpt=None, learn_states=False):

        super().__init__(action_space, gamma, batch_size, size, max_memory, device, 
            learn_every, warmup, lr, epsilon, epsilon_min, epsilon_decay, ckpt, learn_states)
        
        self.algorithm = "dueling"
        self.h = 0
        self.values = np.empty((0, 2))
        self.net = Net(4, action_space, size, "dueling").to(device)
        if ckpt is not None:
            self.net.load_state_dict(torch.load(ckpt, map_location=device))
        self.optimizer1 = optim.Adam([{"params":self.net.blocks.parameters()}, {"params":self.net.fc1.parameters()}, {"params":self.net.fc2.parameters()}], lr=lr)
        self.optimizer2 = optim.Adam([{"params":self.net.blocks.parameters()}, {"params":self.net.fc1_2.parameters()}, {"params":self.net.fc2_2.parameters()}], lr=lr)
        self.scheduler1 = torch.optim.lr_scheduler.CyclicLR(self.optimizer1, base_lr=1e-5, max_lr=1e-3, step_size_up=2000, cycle_momentum=False)
        self.scheduler2 = torch.optim.lr_scheduler.CyclicLR(self.optimizer2, base_lr=1e-5, max_lr=1e-3, step_size_up=2000, cycle_momentum=False)
            

    def update_q(self, state, next_state, action, reward, done):

        self.net.train()

        q1 = self.net(next_state, 1)
        action1 = torch.argmax(q1, dim=1)
        q2 = self.net(next_state, 2)
        action2 = torch.argmax(q2, dim=1)

        target1 = self.gamma * self.net(state, 2)[np.arange(0, self.batch_size), action1] * ~done + reward
        target2 = self.gamma * self.net(state, 1)[np.arange(0, self.batch_size), action2] * ~done + reward

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

        self.scheduler1.step()
        self.scheduler2.step()

        return [q1t.mean().item(), q2t.mean().item()], [loss1.item(), loss2.item()]


    @torch.no_grad()
    def act(self, state, height, show_stats=True, ax=None):
        self.counter += 1
        self.net.eval()
        
        q1 = self.net(state, 1)
        q2 = self.net(state, 2)
        q = q1 if np.random.rand() < 0.5 else q2
                
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_space)
            color = 'g'
        else: 
            action = torch.argmax(q).item()
            if height < self.h:
                action = self.mask_jumps(action)
            color = 'r' 

        self.h = height
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if show_stats and self.counter > self.warmup and self.counter % 2 == 0:
            v1, v2 = self.net.get_value(state)
            ad1, ad2 = self.net.get_adv(state)
            value = np.array([v1.squeeze().detach().cpu().numpy(), v2.squeeze().detach().cpu().numpy()])
            ad = np.array([ad1.squeeze().detach().cpu().numpy(), ad2.squeeze().detach().cpu().numpy()])
            self.plot_stats(ad, action, color, value, ax=ax)
            
        return action
        

    def learn(self):
        if self.counter < self.warmup:
            return None, None
        
        if self.counter % self.learn_every != 0:
            return None, None

        state, next_state, action, reward, done = self.sample_from_memory()
        mean_q, loss = self.update_q(state, next_state, action, reward, done)

        return mean_q, loss 
    

    def plot_stats(self, q, action, color, v, ax):
        ax1, ax2, ax3 = ax
        q1, q2 = q
        self.values = np.append(self.values, v.reshape(1,2), axis=0)
        if len(self.values) > 100:
            self.values = self.values[1:]
        v1 = self.values[:,0]
        v2 = self.values[:,1]   
        ax1.clear()
        ax2.clear()
        ax3.clear()
        bars1 = ax1.bar(range(len(q1)), q1, width=0.4)
        bars1[action].set_color(color)
        ax1.set_xlabel('Actions')
        ax1.set_ylabel('Q-values')
        bars2 = ax2.bar(range(len(q2)), q2, width=0.4)
        bars2[action].set_color(color)
        ax2.set_xlabel('Actions')
        ax3.plot(v1, label="Value1")
        ax3.plot(v2, label="Value2")
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