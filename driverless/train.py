# ready to run example: PythonClient/multirotor/hello_drone.py
import airsim
import os
import numpy as np
from PIL import Image
import logging
from car_env import CarEnv

from settings import NUM_EPISODES

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

# connect to the AirSim simulator
client = airsim.CarClient()
client.reset()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

env = CarEnv(client)

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

        # tranform and save images in temp enumerated by i
        Image.fromarray(scene).save(os.path.normpath('./temp/scene_' + str(i) + '.png'))
        Image.fromarray(planner.astype("uint8")).save(os.path.normpath('./temp/planner_' + str(i) + '.png'))
        state = client.getCarState()
        
        # print state details
        logging.info(f"speed={state.speed}, gear={state.gear}, collision={state.collision.has_collided}, timestamp={state.timestamp}, reward={reward}")

   