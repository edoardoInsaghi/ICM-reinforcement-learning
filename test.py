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

env = new_env(args.movement, args.pixels, args.world, args.stage)

if args.movement == "simple":
    action_space = 7
else:
    action_space= 12
    
player = AC_Agent(action_space, args, device=device)
    
if args.load_param != "":
    print(f"Loading weights from {args.load_param}")
    player.net.load_state_dict(torch.load(args.load_param, map_location=device))

player.net.eval()
for episode in range(0, int(args.episodes)):

    state = env.reset()
    state = torch.tensor(np.asarray(state) / 255.0, dtype=torch.float32, device=device).unsqueeze(0).to(device)
    
    while True:
    
        action = player.act(state, height=None, ax=None, show_stats=not args.cluster)
        state, reward, done, info = env.step(action)
        state = torch.tensor(np.asarray(state) / 255.0, dtype=torch.float32, device=device).unsqueeze(0).to(device)

        env.render()
        if done:
            break
    
