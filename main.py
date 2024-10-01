from audioop import avg
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
from Agent import *
from Logger import *
from Environment import *
from arg_parse import *

args = get_args()

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using device CUDA')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using device MPS')
else:
    print('Using device CPU')


env = new_env(args.movement, args.pixels)

'''
cluster = False
learn = True
if cluster:
    learn = True
savefile = "ddqn.pth"
save = False

action_space = 7

# Hyperparameters
batch_size = 32
warmup = 1000
epsilon = 1.0
mem = int(1e5)
if learn is False:
    epsilon = 0.0
epsilon_decay = 0.999
lr = 0.000025
num_steps = 50

# Start from pretrained, learn state representations, algorithm
ckpt = "./weights/ddqn.pth"
learn_states = False
# algo = 'fdqn'
#algo = 'ddqn'
# algo = 'dueling'
algo = 'ddqn'
'''

if args.movement == "simple":
    action_space = 7
else:
    action_space= 12
    
if args.algo == 'fdqn':
    #player = FDQN_Agent(action_space, batch_size=batch_size, device=device, warmup=warmup, epsilon=epsilon, epsilon_decay=epsilon_decay, lr=lr, ckpt=ckpt, learn_states=learn_states, max_memory=mem)
    player = FDQN_Agent(action_space, args, device=device )
elif args.algo == 'ddqn':
    #player = DDQN_Agent(action_space, batch_size=batch_size, device=device, warmup=warmup, epsilon=epsilon, epsilon_decay=epsilon_decay, lr=lr, ckpt=ckpt, learn_states=learn_states, max_memory=mem)
    player = DDQN_Agent(action_space, args,  device=device)
elif args.algo =='dueling':
    #player = DUELING_Agent(action_space, batch_size=batch_size, device=device, warmup=warmup, epsilon=epsilon, epsilon_decay=epsilon_decay, lr=lr, ckpt=ckpt, learn_states=learn_states, max_memory=mem)
    player = DUELING_Agent(action_space, args, device=device)
elif args.algo == 'ac':
    player = AC_Agent(action_space, args, device=device)

# filename = "dueling.txt"
filename = None
logger = Logger(5, filename)

ax = None
# Plot Setups
plt.ion()
if args.algo == "ddqn":
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax = [ax1, ax2]
elif args.algo == "fdqn": 
    plt.show()
    ax = None
elif args.algo == "dueling":
    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2) 
    ax = [ax1, ax2, ax3]


for episode in range(1, int(args.episodes)):

    distance = 0
    height = 0
    state = env.reset()
    state = torch.tensor(np.asarray(state) / 255.0, dtype=torch.float32, device=device).unsqueeze(0).to(device)
    
    while True:
        
        action = player.act(state, height, show_stats=not args.cluster, ax=ax)
        
        next_state, reward, done, info = env.step(action)
        next_state = torch.tensor(np.asarray(next_state) / 255.0, dtype=torch.float32).unsqueeze(0).to(device)
        distance = info['x_pos']
        height = info['y_pos']

        if args.algo != "ac":
            player.cache(state.squeeze(0), next_state.squeeze(0), action, reward, done)
        else:
            done, next_state = player.get_experience(env, state, args.local_steps, device, show_stats=not args.cluster)

            q, loss = player.learn()

        if done:
            break
        
        if not args.cluster:
            env.render()

        state = next_state

    
    if episode % 5 == 0 and args.save_param != "":
        torch.save(player.net.state_dict(), args.save_param)

    if player.counter > player.warmup and args.learn:
        logger.log_episode()
        logger.print_last_episode()
    
        
assert False
player.net.load_state_dict(torch.load("./AC.pth", map_location=device))
step = 0
num_steps = 50


for episode in range(1, episodes + 1):
    
    state = env.reset()
    state = torch.tensor(np.asarray(state) / 255.0, dtype=torch.float32, device=device).unsqueeze(0).to(device)
    height = 0
    
    done = False
    while not done:
        log_policies = []
        values = []
        rewards = []
        entropies = []
        curr_step = 0
        step += 1
        
        for _ in range(num_steps):
            curr_step += 1
            
            action, logits = player.act2(state, height, show_stats=True)
            value = player.net(state, model=2)
            policy = torch.softmax(logits, dim=1)
            log_policy = torch.log_softmax(logits, dim=1)
            entropy = -(policy * log_policy).sum(1, keepdim=True)
            
            state, reward, done, info = env.step(action)
            #distance = info['x_pos']
            height = info['y_pos']
            state = torch.tensor(np.asarray(state) / 255.0, dtype=torch.float32, device=device).unsqueeze(0).to(device)

            if done:
                break
            
            env.render()
            
            values.append(value)
            log_policies.append(log_policy[0, action])
            rewards.append(reward)
            entropies.append(entropy)
            
        v, loss = player.update(log_policies=log_policies, rewards=rewards, values=values, entropies=entropies, done=done, last_state=state)
        print(f"value {v.item()}", {info["score"]})
        
        if step % 500 == 0 and save:
            torch.save(player.net.state_dict(), "AC.pth")
            print(f"saved model parameters step {step}")

env.close()

assert False
for _ in range(30):
    player.net.eval()
    
    state = env.reset()
    state = torch.tensor(np.asarray(state) / 255.0, dtype=torch.float32, device=device).unsqueeze(0).to(device)
    height = 0
    
    done = False
    while not done:
        action, logits = player.act2(state, height, show_stats=True)
        value = player.net(state, model=2)
        state, reward, done, info = env.step(action)
        state = torch.tensor(np.asarray(state) / 255.0, dtype=torch.float32, device=device).unsqueeze(0).to(device)
        print(value.item())
        
        env.render()
        
env.close()

env.close()

