# ready to run example: PythonClient/multirotor/hello_drone.py
from settings import NUM_EPISODES
import airsim
import os
import numpy as np
from PIL import Image
import logging
from car_env import CarEnv
from stable_baselines3 import A2C
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
from cnn_policy import CustomPolicy

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)


# connect to the AirSim simulator
client = airsim.CarClient()
client.reset()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

env = CarEnv(client)

# create custom model and learn using conv
model = A2C(
    CustomPolicy, 
    env, 
    verbose=1, 
    device='cuda', 
    tensorboard_log="./tensorboard/",
    policy_kwargs=dict(net_arch=[64, 64], activation_fn=th.nn.ReLU, ortho_init=False),
)

model.learn(total_timesteps=1000)

for eps_i in range(NUM_EPISODES):

    i = 0
    done = False
    logging.info(f"Episode {eps_i}")
    env.reset()

    while not done:
        i += 1
        # obtain state and image and relevant info to compute reward
        action = np.random.randint(1, 6)
        observation, reward, done, info = env.step(action)
        scene = observation[:, :, :3].astype("uint8")
        planner = observation[:, :, 3].astype("uint8")

   
        state = env.state["car_state"]
        logging.info(f"speed={state.speed}, gear={state.gear}, collision={state.collision.has_collided}, timestamp={state.timestamp}, reward={reward}")

   