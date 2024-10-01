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

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            total_reward += (info["score"] - self.current_score) / 40.0
            self.current_score = info["score"]
            if done:
                if info["flag_get"]:
                    reward += 50
                else:
                    reward -= 50
                break

        return obs, total_reward / 10.0, done, info
    
'''
class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip=4):
        super(FrameSkip, self).__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        obs = None

        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return obs, total_reward, done, info
'''

def new_env(movement_type, w):
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    
    if movement_type == "simple":
        movement = SIMPLE_MOVEMENT
    else:
        movement = COMPLEX_MOVEMENT
        
    env = FrameSkip(env, 4)
    env = gym.wrappers.ResizeObservation(env, (w, w)) 
    env = GrayScaleObservation(env)     
    env = FrameStack(env, 4) 
    env = JoypadSpace(env, movement)
    return env