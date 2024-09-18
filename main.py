from audioop import avg
import gym
import gym_super_mario_bros
from gym.wrappers import GrayScaleObservation, FrameStack
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
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

w = 84
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = gym.wrappers.ResizeObservation(env, (w, w)) 
env = GrayScaleObservation(env)     
env = FrameStack(env, 4) 
env = JoypadSpace(env, SIMPLE_MOVEMENT)

player = DDQN_Agent(7, batch_size=32, size=w, device=device, algo="DDQN")
episodes = 50
logger = Logger()

for episode in range(1, episodes+1):

    state = env.reset()

    while True:

        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = player.act(state)
        next_state, reward, done, info = env.step(action)
        player.cache(state.squeeze(0), next_state, action, reward, done)

        q, loss = player.learn()

        if q is not None and loss is not None and reward is not None:
            logger.log_step(reward, loss, q)

        if done:
            break

        env.render()
        state = state.squeeze(0)

    logger.log_episode()
    logger.print_last_episode()

env.close()