import numpy as np
import torch
import torch.multiprocessing as mp
from Agent import AC_Agent
from Environment import new_env
from shared_adam import SharedAdam
from arg_parse import get_args
from Modules import AC_NET
from worker import worker
from multiprocessing import Lock


if __name__ == '__main__':

    mp.set_start_method('spawn')
    
    args = get_args()
    if args.movement == "simple":
        action_space = 7
    else:
        action_space= 12
        
    device = torch.device('cpu')
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using device CUDA')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print('Using device MPS')
    else:
        print('Using device CPU')
        
    
    update_lock = Lock()
    
    global_model = AC_NET(4, action_space)
    
    if args.load_param != "":
        print(f"Loading weights from {args.load_param}")
        global_model.load_state_dict(torch.load(args.load_param, map_location=device))
        
    global_model = global_model.to(device)
    global_model.share_memory()  
    #global_model = global_model.cpu()
    
    optimizer = SharedAdam(global_model.parameters(), lr=args.lr)
    
    processes = []
    for rank in range(args.num_workers):
        p = mp.Process(target=worker, args=(rank, args, global_model, optimizer, device, action_space, update_lock))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
