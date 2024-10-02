import torch
import numpy as np
from Agent import AC_Agent
from Environment import *

def worker(rank, args, global_model, optimizer, device, action_space, update_lock):
    torch.manual_seed(123 + rank)
    env = new_env(args.movement, args.pixels, args.world, args.stage)

    agent = AC_Agent(action_space, args, device=device)  

    state = env.reset()
    state = torch.tensor(np.asarray(state) / 255.0, dtype=torch.float32).unsqueeze(0).to(device)
    step = 0
    num_action = 0
    r = 0
    episode = 0
    
    while True:
        agent.net.load_state_dict(global_model.state_dict())
        episode += 1
        
        if rank == 0:
            if step % 250 == 0 and args.save_param != "":
                print(f"process {rank} speaking, {step} steps, saving model paramaters")
                torch.save(global_model.state_dict(), "{}/a3c_super_mario_bros_{}_{}".format(args.save_param, args.world, args.stage))
            
            done, state = agent.get_experience(env, state, args.local_steps, device, show_stats=not args.cluster)
            num_action += len(agent.values)
            r += np.sum(agent.rewards)
        else:
            done, state = agent.get_experience(env, state, args.local_steps, device, show_stats=False)
        

        loss = agent.learn2()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.net.parameters(), 0.5)
        
        with update_lock:
            for local_param, global_param in zip(agent.net.parameters(), global_model.parameters()):
                if global_param.grad is None:
                    global_param._grad = local_param.grad.clone()
                else:
                    global_param._grad += local_param.grad.clone()
            optimizer.step()
        step += 1
        
        print(f"process {rank}, step number {step}")

        if done:
            state = env.reset()
            state = torch.tensor(np.asarray(state) / 255.0, dtype=torch.float32).unsqueeze(0)
            
            if rank == 0:
                with open(args.save_file, 'a') as f:
                    f.write(f"{episode},{r},{agent.info['x_pos']},{agent.info['flag_get']},{num_action}\n")
                    if agent.info["flag_get"] == True:
                        break         
