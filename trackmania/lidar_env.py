from tmrl import get_environment
from time import sleep
import numpy as np
import gym
from settings import MAX_SPEED_REWARD, MAX_SPEED

# env wrapper that only gets the LIDAR observations
class LidarWrapper(gym.Env):
    def __init__(self, env=get_environment(), df=4):
        self.trac_env = env
        self.df = df
        # shape 4, 19
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4, 19), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3 * df) 
        self.max_distance = 600
        self.time = 0

    def __transform_observation(self, obs):
        obs = obs[1]
        print(obs.max(), obs.min(), obs.shape)
        obs = obs / self.max_distance
        return obs

    def reset(self):
        obs, _ = self.trac_env.reset()
        return self.__transform_observation(obs)

    def interpret_action(self, action):
        if self.time <= 16:
            return np.array([1, 0, 0])
        action_taken = action // self.df
        strength = action % self.df
        action = np.zeros(3)
        action[action_taken] = strength / (self.df - 1.0) * 2.0 - 1.0
        return action

    def step(self, action):
        self.time += 1
        action = self.interpret_action(action)
        obs, rew, terminated, truncated, info = self.trac_env.step(action)
        speed = obs[0]
        reward_speed = 0.00001 * (speed / MAX_SPEED) * MAX_SPEED_REWARD

        obs = self.__transform_observation(obs)
        final_reward = rew + reward_speed
        if terminated or truncated:
            final_reward = -0.5
        print("\033[92m{}\033[0m, rew_speed={}, rew={}, speed={}, done={}".format(final_reward, reward_speed, rew, speed, terminated))
        return obs, final_reward, terminated or truncated, info
    
    def wait(self):
        self.trac_env.wait()