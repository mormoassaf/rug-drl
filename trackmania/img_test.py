from tmrl import get_environment
from time import sleep
import numpy as np
import matplotlib.pyplot as plt

env = get_environment()
print(env.observation_space)

sleep(1.0)  # just so we have time to focus the TM20 window after starting the script

obs, info = env.reset()  # reset environment
for _ in range(100000):  # rtgym ensures this runs at 20Hz by default

    speed, t, d, lidar, action_1, action_2 = obs
    print(f"d: {d}, speed: {speed}m/s, time: {t}s, lidar: {lidar.shape}, action: {action_1}, {action_2}")

    vizualization = lidar.mean(axis=0)
    plt.imsave("temp/viz.png", vizualization, cmap="gray")

    act = np.array([1.0, 0.0, 0.0])  # gas, break, steer
    obs, rew, terminated, truncated, info = env.step(act)  # step (rtgym ensures healthy time-steps)
    # termintted means the episode is over, truncated means the episode is not over but the time limit is reached
    if terminated or truncated:
        break
env.wait()  # rtgym-specific method to artificially 'pause' the environment when needed