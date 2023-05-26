from tmrl import get_environment
from time import sleep
import numpy as np
import gym

# default LIDAR observations are of shape: ((1,), (4, 19), (3,), (3,))
# representing: (speed, 4 last LIDARs, 2 previous actions)
# actions are [gas, break, steer], analog between -1.0 and +1.0
def model(lidar):
    """
    simplistic policy for LIDAR observations
    """
    deviation = lidar.mean(0)
    deviation /= (deviation.sum() + 0.001)
    steer = 0
    for i in range(19):
        steer += (i - 9) * deviation[i]
    steer = - np.tanh(steer * 4)
    steer = min(max(steer, -1.0), 1.0)
    return np.array([1.0, 0.0, steer])

# env wrapper that only gets the LIDAR observations
class LidarWrapper(gym.Env):
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space[1]
        self.action_space = env.action_space

    def reset(self):
        obs, info = self.env.reset()
        return obs[1], info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        return obs[1], rew, terminated, truncated, info

    def wait(self):
        self.env.wait()

env = LidarWrapper(get_environment())

sleep(1.0)  # just so we have time to focus the TM20 window after starting the script

obs, info = env.reset()  # reset environment
for _ in range(100000):  # rtgym ensures this runs at 20Hz by default
    act = model(obs)  # compute action
    obs, rew, terminated, truncated, info = env.step(act)  # step (rtgym ensures healthy time-steps)
    if terminated or truncated:
        print("terminated")
        break
env.wait()  # rtgym-specific method to artificially 'pause' the environment when needed