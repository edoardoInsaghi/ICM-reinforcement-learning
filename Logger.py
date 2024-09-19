import numpy as np

class Logger():
    def __init__(self, round=5):
        
        self.round = round

        self.rewards = []
        self.lengths = []
        self.losses = []
        self.q_values = []

        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0


    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss is not None and q is not None:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1


    def log_episode(self):
        self.rewards.append(np.round(self.curr_ep_reward / self.curr_ep_length, self.round))
        self.lengths.append(np.round(self.curr_ep_length, self.round))
        assert self.curr_ep_loss_length > 0 
        avg_loss = self.curr_ep_loss / self.curr_ep_loss_length
        avg_q = self.curr_ep_q / self.curr_ep_loss_length
        self.losses.append(np.round(avg_loss, self.round))
        self.q_values.append(np.round(avg_q, self.round))
        self.new_episode()


    def compute_avg(self, terms=100):
        assert len(self.rewards) > terms

        mean_reward = np.round(np.mean(self.rewards[-terms:]), self.round)
        mean_length = np.round(np.mean(self.lengths[-terms:]), self.round)
        mean_loss = np.round(np.mean(self.losses[-terms:]), self.round)
        mean_q = np.round(np.mean(self.q_values[-terms:]), self.round)

        self.moving_avg_ep_rewards.append(mean_reward)
        self.moving_avg_ep_lengths.append(mean_length)
        self.moving_avg_ep_avg_losses.append(mean_loss)
        self.moving_avg_ep_avg_qs.append(mean_q)

        return mean_reward, mean_length, mean_loss, mean_q
    

    def print_last_episode(self):
        n = len(self.rewards)
        assert n > 0
        print(f"Episode {n}: Avg Rewards: {self.rewards[-1]}, Length of the episode: {self.lengths[-1]}, Avg Loss: {self.losses[-1]}, Avg Q: {self.q_values[-1]}")


    def reset(self):
        self.rewards = []
        self.lengths = []
        self.losses = []
        self.q_values = []

        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0 


    def new_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0 