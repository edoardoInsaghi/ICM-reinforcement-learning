import gym
import gym_super_mario_bros
from gym.wrappers import GrayScaleObservation, FrameStack
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace


class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip=4):
        super(FrameSkip, self).__init__(env)
        self._skip = skip
        self.current_score = 0
        self.life = 3

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            lifes = info["life"]
            if self.life > lifes:
                reward -= 50
                self.life = lifes
            total_reward += (info["score"] - self.current_score) / 40.0
            self.current_score = info["score"]
            if done:
                self.life = 3
                if info["flag_get"]:
                    reward += 50
                else:
                    reward -= 50
                break
        
        return obs, total_reward / 10.0, done, info

    
class FrameSkipNoReward(gym.Wrapper):
    def __init__(self, env, skip=4, sparse=False):
        super(FrameSkipNoReward, self).__init__(env)
        self._skip = skip
        self.sparse = sparse
    

    def step(self, action):
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            reward = 0
            
            if self.sparse:
                if done:
                    if info["flag_get"]:
                        reward += 5
                    else:
                        reward -= 5 
                    break
            else:
                if done:
                    break    
            
        
        return obs, reward, done, info


def new_env(movement_type, w, world, stage, reward):
    env = gym_super_mario_bros.make('SuperMarioBros-{}-{}-v0'.format(world, stage))
    
    if movement_type == "simple":
        movement = SIMPLE_MOVEMENT
    else:
        movement = COMPLEX_MOVEMENT
        
    if reward == 0:
        env = FrameSkipNoReward(env, 4)
    elif reward == 1: 
        env = FrameSkipNoReward(env, 4, sparse=True)
    else:
        env = FrameSkip(env, 4)
        
    env = gym.wrappers.ResizeObservation(env, (w, w)) 
    env = GrayScaleObservation(env)     
    env = FrameStack(env, 4) 
    env = JoypadSpace(env, movement)
    return env
