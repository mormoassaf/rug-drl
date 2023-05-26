from tmrl import get_environment
from time import sleep
import numpy as np
import gym

# env wrapper that only gets the LIDAR observations
class LidarWrapper(gym.Env):
    def __init__(self, env=get_environment(), df=6):
        self.trac_env = env
        self.df = df
        # shape 4, 19
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4, 19), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3 * df) 
        self.max_distance = 2048

    def __transform_observation(self, obs):
        obs = obs[1]
        assert obs.max() <= self.max_distance, "max distance is too small got {}, expected {}".format(obs.max(), self.max_distance)
        obs = obs / self.max_distance
        return obs

    def reset(self):
        obs, _ = self.trac_env.reset()
        return self.__transform_observation(obs)

    def interpret_action(self, action):
        action_taken = action // self.df
        strength = action % self.df
        action = np.zeros(3)
        action[action_taken] = strength / (self.df - 1.0) * 2.0 - 1.0
        return action

    def step(self, action):
        action = self.interpret_action(action)
        obs, rew, terminated, truncated, info = self.trac_env.step(action)
        obs = self.__transform_observation(obs)
        return obs, rew, terminated, info
    
    def wait(self):
        self.trac_env.wait()