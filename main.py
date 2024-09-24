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


w = 84
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = gym.wrappers.ResizeObservation(env, (w, w)) 
env = GrayScaleObservation(env)     
env = FrameStack(env, 4) 
env = JoypadSpace(env, SIMPLE_MOVEMENT)


player = FDQN_Agent(7, batch_size=32, device=device, warmup=500, epsilon=1, epsilon_decay=0.999, lr=0.00025, ckpt='fdqn.pth')
episodes = 1000000
logger = Logger()
# rdm = Reverse_Dynamics_Module(action_space=7, device=device).to(device)

plt.ion()
plt.show()

y1 = 0
y2 = 0
height = 0
for episode in range(1, episodes+1):

    state = env.reset()
    state = torch.tensor(np.asarray(state) / 255.0, dtype=torch.float32, device=device).unsqueeze(0).to(device)
    
    while True:
        
        action = player.act(state, height, show_stats=True)
        
        next_state, reward, done, info = env.step(action)
        next_state = torch.tensor(np.asarray(next_state) / 255.0, dtype=torch.float32).unsqueeze(0).to(device)
        next_state = next_state / 255.0 
        distance = info['x_pos']
        height = info['y_pos']

        player.cache(state.squeeze(0), next_state.squeeze(0), action, reward, done)

        q, loss = player.learn()
        logger.log_step(reward, loss, q, distance)

        if done:
            break

        env.render()
        state = next_state

    if episode % 5 == 0:
        torch.save(player.net.state_dict(), "fdqnlocal.pth")

    if player.counter > player.warmup:
        logger.log_episode()
        logger.print_last_episode()

env.close()