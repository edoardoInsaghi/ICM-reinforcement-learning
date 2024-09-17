from audioop import avg
import gym
import gym_super_mario_bros
from gym.wrappers import GrayScaleObservation, FrameStack
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import numpy as np
import matplotlib.pyplot as plt
import torch
from modules import *

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using device CUDA')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using device MPS')

w = 84
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = gym.wrappers.ResizeObservation(env, (w, w)) 
env = GrayScaleObservation(env)     
env = FrameStack(env, 4) 
env = JoypadSpace(env, COMPLEX_MOVEMENT)

obs = env.reset()

player = Agent(12, batch_size=32, size=w, device=device)
reward = 0
warmup = 250
skip = 4
avg_reward = 0
epsilon = 0.1
count = 0

for iter in range(100000):

    state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    action = player.act(state, epsilon)
    next_state, reward, done, info = env.step(action)
    player.cache(state.squeeze(0), next_state, action, reward, done)
    count += 1

    if done:
        avg_reward = 0
        count = 0
        state = env.reset()

    if iter < warmup:
        avg_reward += reward / (count + 1)
        print(f'Iter: {iter}, Reward: {reward}, Avg Reward: {avg_reward}')
        env.render()
        continue

    if iter == warmup:
        print('Warmup done')
        avg_reward = 0
        count = 0

    avg_reward += reward / (count + 1)
    states, next_states, actions, rewards, dones = player.sample_from_memory()

    td_estimate = player.td_estimate(states, actions)
    td_target = player.td_target(rewards, next_states)
    loss = player.update_q_online(td_estimate, td_target)

    env.render()

    if iter % 100 == 0:
        player.update_target()
        print(f'Iter: {iter}, Loss: {loss}, Avg Reward: {avg_reward}')

    if done:
        state = env.reset()

env.close()