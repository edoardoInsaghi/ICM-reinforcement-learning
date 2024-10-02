import argparse

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--movement", type=str, default="simple")
    
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--beta', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument("--local_steps", type=int, default=50)
    parser.add_argument('--world', type=int, default=1, help='World number (1-8)')
    parser.add_argument('--stage', type=int, default=1, help='Stage number (1-4)')
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=1e6)
    parser.add_argument("--max_memory", type=int, default=1e5)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--epsilon_min", type=float, default=0.1)
    parser.add_argument("--epsilon_decay", type=float, default=0.999997)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--temperature_min", type=float, default=0.1)
    parser.add_argument("--temperature_decay", type=float, default=0.99)
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--sync_every", type=int, default=1000)
    parser.add_argument("--pixels", type=int, default=84)
    parser.add_argument("--learn_every", type=int, default=4)
    parser.add_argument("--save_file", type=str, default="data/fdqn.txt")
    parser.add_argument("--save_param", type=str, default="weights")
    parser.add_argument("--load_param", type=str, default="")
    parser.add_argument("--algo", type=str, default='fdqn')
    parser.add_argument("--warmup", type=int, default=1000)
    args = parser.parse_args()
    return args
