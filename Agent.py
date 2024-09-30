import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Modules import *
from abc import abstractmethod
from torch.distributions import Categorical

class Agent():
    '''
    def __init__(self, action_space, gamma=0.9, batch_size=32, size=84, max_memory=int(1e5), 
                 device="cpu", learn_every=4, warmup=1000, lr=0.00025,
                 epsilon = 0.15, epsilon_min=0.01, epsilon_decay=0.999997, ckpt=None, learn_states=False):
    '''
    def __init__(self, action_space, args, device):
        self.counter = 0        
        self.memory = []
        self.max_memory = args.max_memory   
        self.action_space = action_space
        #self.learn_states = learn_states

        self.gamma = args.gamma # Reward Discount
        self.epsilon = args.epsilon # Exploration Rate
        self.epsilon_min = args.epsilon_min
        self.epsilon_decay = args.epsilon_decay

        self.warmup = args.warmup
        self.lr = args.lr
        self.learn_every = args.learn_every
        self.batch_size = args.batch_size
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
    '''
    def __init__(self, action_space, gamma=0.9, batch_size=32, size=84, max_memory=int(1e4), 
                 device="cpu", learn_every=4, warmup=1000, lr=0.00025,
                 epsilon = 0.15, epsilon_min=0.01, epsilon_decay=0.999997, sync_every=1000, ckpt=None, learn_states=False):
        
        super().__init__(action_space, gamma, batch_size, size, max_memory, device, 
                         learn_every, warmup, lr, epsilon, epsilon_min, epsilon_decay, ckpt, learn_states)
    '''
    def __init__(self, action_space, args, device):
        super().__init__(action_space, args, device)
        
        self.sync_every = args.sync_every
        self.net = FDQN_NET(4, action_space).to(device)
        
        if args.no_ckpt:
            self.net.load_state_dict(torch.load(args.load_param, map_location=device))
            
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr)
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

    '''
    def __init__(self, action_space, gamma=0.9, batch_size=32, size=84, max_memory=int(1e4), 
                device="cpu", learn_every=4, warmup=1000, lr=0.00025,
                epsilon = 0.15, epsilon_min=0.01, epsilon_decay=0.999997, ckpt=None, learn_states=False):
    
        super().__init__(action_space, gamma, batch_size, size, max_memory, device, 
                         learn_every, warmup, lr, epsilon, epsilon_min, epsilon_decay, ckpt, learn_states)
    ''' 
    def __init__(self, action_space, args, device):
        super().__init__(action_space, args, device)

        self.net = DDQN_NET(4, action_space).to(device)
        
        if args.no_ckpt:
            self.net.load_state_dict(torch.load(args.load_param, map_location=device))
            
        self.optimizer = optim.Adam(self.net.parameters(), lr=args.lr)
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
        loss = loss1 + loss2

        self.optimizer.zero_grad()

        loss1.backward(retain_graph=True)
        loss2.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)

        self.optimizer.step()
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
    def __init__(self, action_space, args, device="cpu"):
        super().__init__(action_space, args, device)
        
        self.net = AC_NET(4, action_space).to(device)
        self.optimizer1 = optim.Adam(self.net.parameters(), lr=args.lr)

        self.temperature = args.temperature
        self.min_temperature = args.temperature_min
        self.temperature_decay = args.temperature_decay
        self.beta = args.beta
        self.log_policies = []
        self.values = []
        self.rewards = []
        self.entropies = []
        self.last_state = None
        
        if args.no_ckpt:
            self.net.load_state_dict(torch.load(args.load_param, map_location=device))
        
    
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
    
    
    def update_temperature(self):
        self.temperature = max(self.min_temperature, self.temperature * self.temperature_decay)
        
        
    def act(self, state, height, show_stats=True, return_both=False, ax=None):
        self.counter += 1
        color = 'r'
        #self.update_temperature()
        print(type(state))
        
        with torch.no_grad():
            state = state.to(self.device)
            logits= self.net(state, model=1)
            #logits = logits / self.temperature
            p = torch.softmax(logits, dim=1)
            
            m = Categorical(p)
            action = m.sample().item()
        
        if show_stats and self.counter > self.warmup and self.counter % 2 == 0:
            self.plot_stats(p.squeeze().detach().cpu().numpy(), action, color)
            
        if not return_both:
            return action
        else:
            return action, logits
    
    
    def get_experience(self, env, state, local_steps, device, show_stats=True, ax=None):
        self.log_policies = []
        self.values = []
        self.rewards = []
        self.entropies = []
        self.last_state = None
        for _ in range(local_steps):
            
            action, logits = self.act(state, height=None, ax=ax, show_stats=show_stats, return_both=True)
            value = self.net(state, model=2)
            policy = torch.softmax(logits, dim=1)
            log_policy = torch.log_softmax(logits, dim=1)
            entropy = -(policy * log_policy).sum(1, keepdim=True)
            
            state, reward, done, info = env.step(action)
            #height = info['y_pos']
            state = torch.tensor(np.asarray(state) / 255.0, dtype=torch.float32, device=device).unsqueeze(0)

            if done:
                break
            
            env.render()
            
            self.values.append(value)
            self.log_policies.append(log_policy[0, action])
            self.rewards.append(reward)
            self.entropies.append(entropy)
            
        self.last_state = state
        return done, self.last_state
        
    def learn(self):
        self.net.train()
        
        actor_loss = torch.zeros((1, 1), dtype=torch.float).to(self.device)
        critic_loss = torch.zeros((1, 1), dtype=torch.float).to(self.device)
        entropy_loss = torch.zeros((1, 1), dtype=torch.float).to(self.device)
        gae = torch.zeros((1, 1), dtype=torch.float).to(self.device)
        
        if self.done:
            R = torch.zeros(1, 1, device=self.device)
        else:
            with torch.no_grad():
                R = self.net(self.last_state, model=2)
                
        R = R.to(self.device)
        next_value = R
        
        for value, log_policy, reward, entropy in list(zip(self.values, self.log_policies, self.rewards, self.entropies))[::-1]:
            gae = gae * self.gamma 
            gae = gae + reward + self.gamma * next_value.detach() - value.detach()
            next_value = value
            actor_loss = actor_loss + log_policy * gae  
            R = R * self.gamma + reward
            critic_loss += (R - value) ** 2 / 2
            entropy_loss += entropy

        total_loss = -actor_loss + critic_loss - self.beta * entropy_loss
        self.optimizer1.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.optimizer1.step()
        
        return next_value.item(), total_loss.item()
            
    


class DUELING_Agent(Agent):
    '''
    def __init__(self, action_space, gamma=0.9, batch_size=32, size=84, max_memory=int(1e4), 
            device="cpu", learn_every=4, warmup=1000, lr=0.00025,
            epsilon = 0.15, epsilon_min=0.01, epsilon_decay=0.9, ckpt=None, learn_states=False):

        super().__init__(action_space, gamma, batch_size, size, max_memory, device, 
            learn_every, warmup, lr, epsilon, epsilon_min, epsilon_decay, ckpt, learn_states)
    '''
    def __init__(self, action_space, args, device="cpu"):
        super().__init__(action_space, args, device)
        self.h = 0
        self.values = np.empty((0, 2))
        self.net = DUELING_NET(4, action_space).to(device)
        
        if args.no_ckpt:
            self.net.load_state_dict(torch.load(args.load_param, map_location=device))
            
        self.optimizer1 = optim.Adam([{"params":self.net.blocks.parameters()}, {"params":self.net.fc1.parameters()}, {"params":self.net.fc2.parameters()}], lr=args.lr)
        self.optimizer2 = optim.Adam([{"params":self.net.blocks.parameters()}, {"params":self.net.fc1_2.parameters()}, {"params":self.net.fc2_2.parameters()}], lr=args.lr)
            

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



def visualize_state(state):

    state_np = state.squeeze(0).cpu().numpy()  # Remove batch dimension and move to CPU
    fig, axs = plt.subplots(1, 4, figsize=(12, 6))

    for i in range(4):
        axs[i].imshow(state_np[i])
        axs[i].set_title(f'Frame {i+1}')
        axs[i].axis('off')
    
    plt.tight_layout()
    plt.show()