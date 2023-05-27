from tmrl import get_environment
from time import sleep
import numpy as np
import gym
from settings import MAX_SPEED_REWARD, MAX_SPEED

def dummy_model(lidar):
    """
    simplistic policy for LIDAR observations
    """
    if np.random.random() < 0.3:
        act = np.random.randint(0, 3)
        action = np.zeros(3)
        action[act] = 2*np.random.random() - 1
        return action
    deviation = lidar.mean(0)
    deviation /= (deviation.sum() + 0.001)
    steer = 0
    for i in range(19):
        steer += (i - 9) * deviation[i]
    steer = - np.tanh(steer * 4)
    steer = min(max(steer, -1.0), 1.0)
    return np.array([0.0, 0.0, steer])

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
        obs = obs / self.max_distance
        return obs

    def reset(self):
        self.time = 0
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

        # Compute rewards
        speed = obs[0]
        reward_speed = 0.000 * (speed / MAX_SPEED) * MAX_SPEED_REWARD
        dummy_action = dummy_model(obs[1])
        action_cosine = np.dot(action, dummy_action) / (np.linalg.norm(action) * np.linalg.norm(dummy_action) + 1e-8)
        reward_dummy = 0.01 * (action_cosine + 1.0) / 2.0

        final_reward = rew + reward_speed + reward_dummy
        if terminated or truncated:
            final_reward = -1
        

        print("\033[92m{}\033[0m, rew_speed={}, rew_dummy={}, rew={}, speed={}, done={}".format(final_reward, reward_speed, reward_dummy, rew, speed, terminated))
        obs = self.__transform_observation(obs)
        return obs, final_reward, terminated or truncated, info
    
    def wait(self):
        self.trac_env.wait()