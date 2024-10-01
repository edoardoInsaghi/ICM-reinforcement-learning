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
logger = Logger(5, args.save_file)


if args.movement == "simple":
    action_space = 7
else:
    action_space= 12

    
if args.algo == 'fdqn':
    player = FDQN_Agent(action_space, args, device=device )

elif args.algo == 'ddqn':
    player = DDQN_Agent(action_space, args,  device=device)

elif args.algo =='dueling':
    player = DUELING_Agent(action_space, args, device=device)

elif args.algo == 'ac':
    player = AC_Agent(action_space, args, device=device)


logger = Logger(5, args.save_file)

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
        logger.log_step(reward, loss, q, distance)

        if done:
            break
        
        if not args.cluster:
            env.render()

        state = next_state

    
    if episode % 5 == 0:
        torch.save(player.net.state_dict(), args.save_param)

    if player.counter > player.warmup:
        logger.log_episode()
        logger.print_last_episode()


env.close()

