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

if args.movement == "simple":
    action_space = 7
else:
    action_space= 12
    
player = AC_Agent(action_space, args, device=device)
training_step = 0

for episode in range(0, int(args.episodes)):

    state = env.reset()
    state = torch.tensor(np.asarray(state) / 255.0, dtype=torch.float32, device=device).unsqueeze(0).to(device)
    
    while True:
    
        done, last_state = player.get_experience(env, state, args.local_steps, device, show_stats=not args.cluster)
        v, loss = player.learn()
        training_step += 1
        if training_step % 250 == 0 and args.save_param != "":
            print(f"saved model pramaters step {training_step}")
            torch.save(player.net.state_dict(), args.save_param)
        if done:
            break
            
        state = last_state
        
