from audioop import avg
import gym
import gym_super_mario_bros
from gym.wrappers import GrayScaleObservation, FrameStack
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import nes_py
import numpy as np
import matplotlib.pyplot as plt
import torch
from Agent import *
from Logger import *

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using device CUDA')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using device MPS')
else:
    print('Using device CPU')

w = 84
movement = SIMPLE_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = gym.wrappers.ResizeObservation(env, (w, w)) 
env = GrayScaleObservation(env)     
env = FrameStack(env, 4) 
env = JoypadSpace(env, movement)

cluster = False
save = False

if movement == COMPLEX_MOVEMENT:
    action_space = 12
else:
    action_space = 7

# Hyperparameters
batch_size = 32
warmup = 100
epsilon = 0.2
epsilon_decay = 0.999
lr = 0.00025

# Start from pretrained, learn state representations, algorithm
ckpt = None
learn_states = False
# algo = 'fdqn'
# algo = 'ddqn'
algo = 'dueling'

if algo == 'fdqn':
    player = FDQN_Agent(action_space, batch_size=batch_size, device=device, warmup=warmup, epsilon=epsilon, epsilon_decay=epsilon_decay, lr=lr, ckpt=ckpt, learn_states=learn_states)
elif algo == 'ddqn':
    player = DDQN_Agent(action_space, batch_size=batch_size, device=device, warmup=warmup, epsilon=epsilon, epsilon_decay=epsilon_decay, lr=lr, ckpt=ckpt, learn_states=learn_states)
elif algo =='dueling':
    player = DUELING_Agent(action_space, batch_size=batch_size, device=device, warmup=warmup, epsilon=epsilon, epsilon_decay=epsilon_decay, lr=lr, ckpt=ckpt, learn_states=learn_states)


episodes = 10000
# filename = "fdqn_learn_states.txt"
filename = None
logger = Logger(5, filename)


plt.ion()
two = True # Set True if function returns two q values
if algo == "ddqn":
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax = [ax1, ax2]
elif algo == "fdqn": 
    plt.show()
    ax = None
elif algo == "dueling":
    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2) 
    ax = [ax1, ax2, ax3]


for episode in range(1, episodes+1):

    distance = 0
    height = 0
    state = env.reset()
    state = torch.tensor(np.asarray(state) / 255.0, dtype=torch.float32, device=device).unsqueeze(0).to(device)
    
    while True:
        
        action = player.act(state, height, show_stats=not cluster, ax=ax)
        
        next_state, reward, done, info = env.step(action)
        next_state = torch.tensor(np.asarray(next_state) / 255.0, dtype=torch.float32).unsqueeze(0).to(device)
        distance = info['x_pos']
        height = info['y_pos']

        player.cache(state.squeeze(0), next_state.squeeze(0), action, reward, done)

        q, loss = player.learn()
        logger.log_step(reward, loss, q, distance)

        if done:
            break
        
        if not cluster:
            env.render()

        state = next_state

    if episode % 5 == 0 and save:
        torch.save(player.net.state_dict(), "fdql_learn_states.pth")

    if player.counter > player.warmup:
        logger.log_episode()
        logger.print_last_episode()

env.close()
