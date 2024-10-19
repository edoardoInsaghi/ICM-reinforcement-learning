import numpy as np

class Logger():
    def __init__(self, round=5, filename=None):
        
        self.round = round
        self.filename = filename
        if filename is not None:
            with open(filename, 'w') as f:
                f.write(f"Rewards,Lengths,Loss1,Loss2,Q1,Q2,Best_Distances\n")

        self.rewards = []
        self.avg_rewards = []
        self.lengths = []
        self.losses = []
        self.q_values = []
        self.best_distances = []

        self.best_distance = 0.0
        self.curr_ep_reward = 0.0
        self.curr_ep_avg_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss_length = 0
        self.curr_ep_loss = np.array([0.0, 0.0])
        self.curr_ep_q = np.array([0.0, 0.0])


    def log_step(self, reward, loss, q, distance):
        self.best_distance = max(self.best_distance, distance)
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss is not None and q is not None:
            self.curr_ep_loss += np.array(loss, dtype=np.float32) if isinstance(loss, list) else np.array([loss, None], dtype=np.float32)
            self.curr_ep_q += np.array(q, dtype=np.float32) if isinstance(q, list) else np.array([q, None], dtype=np.float32)
            self.curr_ep_loss_length += 1


    def log_episode(self):
        self.best_distances.append(np.round(self.best_distance, self.round))
        self.rewards.append(np.round(self.curr_ep_reward, self.round))
        self.curr_ep_avg_reward = self.curr_ep_reward / self.curr_ep_length
        self.avg_rewards.append(np.round(self.curr_ep_avg_reward, self.round))
        self.lengths.append(np.round(self.curr_ep_length, self.round))
        assert self.curr_ep_loss_length > 0 
        avg_loss = self.curr_ep_loss / self.curr_ep_loss_length
        avg_q = self.curr_ep_q / self.curr_ep_loss_length
        self.losses.append(np.round(avg_loss, self.round))
        self.q_values.append(np.round(avg_q, self.round))
        self.new_episode()
    

    def print_last_episode(self):
        n = len(self.rewards)
        assert n > 0
        print(f"Episode {n}: Total Rewards: {self.rewards[-1]}, Length of the episode: {self.lengths[-1]}, Avg Rewards: {self.avg_rewards[-1]}")  
        print(f"             Avg Loss: {self.losses[-1]}, Avg Q: {self.q_values[-1]}, Best Distance: {self.best_distances[-1]}")

        if self.filename is not None:
            with open(self.filename, 'a') as f:
                f.write(f"{self.rewards[-1]},{self.lengths[-1]},{self.losses[-1][0]},{self.losses[-1][1]},{self.q_values[-1][0]},{self.q_values[-1][1]},{self.best_distances[-1]}\n")


    def reset(self):
        self.rewards = []
        self.lengths = []
        self.losses = []
        self.q_values = []
        self.best_distances = []

        self.best_distance = 0.0
        self.curr_ep_reward = 0.0
        self.curr_ep_avg_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0 


    def new_episode(self):
        self.best_distance = 0.0
        self.curr_ep_reward = 0.0
        self.curr_ep_avg_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0 