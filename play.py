from audioop import avg
import numpy as np
import argparse
import torch
from ActorCritic_ICM.icm_agent import ICM_Agent
from ActorCritic_ICM.ac_agent import AC_Agent
from Environment import *
from arg_parse import *
import torch
import matplotlib.pyplot as plt

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

plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax = [ax1, ax2]


if args.movement == "simple":
    action_space = 7
else:
    action_space= 12
        
if args.agent != "ac":
    env = new_env(args.movement, args.pixels, args.world, args.stage, reward=0)
    player = ICM_Agent(action_space, args, device)
    if args.load_param != "":
        ac_save = torch.load(f"{args.load_param}1", map_location=torch.device(device))
        rev_save = torch.load(f"{args.load_param}2", map_location=torch.device(device))
        forward_save = torch.load(f"{args.load_param}3", map_location=torch.device(device))
        player.reverse.load_state_dict(rev_save)
        player.ac_net.load_state_dict(ac_save)
        player.forward_net.load_state_dict(forward_save)
        print("Loaded parameters")
    
else:
    env = new_env(args.movement, args.pixels, args.world, args.stage, reward=False)
    player = AC_Agent(action_space, args, device)
    if args.load_param != "":
        player.net.load_state_dict(torch.load(args.load_param, map_location=device))
    
training_step = 0

for episode in range(0, int(args.episodes)):

    state = env.reset()
    state = torch.tensor(np.asarray(state) / 255.0, dtype=torch.float32, device=device).unsqueeze(0).to(device)

    steps = 0
    r = 0
    
    while True:
        
        done, last_state = player.get_experience(env, state, args.local_steps, device, show_stats=not args.cluster, ax=ax)
        steps += len(player.values)

        if done:
            env.reset()
            break
        
        state = last_state
