# ready to run example: PythonClient/multirotor/hello_drone.py
import airsim
import os
import numpy as np
from PIL import Image
import logging
import gym

NUM_EPISODES = 5
MAX_SPEED = 300
MIN_SPEED = 10
MAX_DEPTH = 65536

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

# connect to the AirSim simulator
client = airsim.CarClient()
client.reset()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)


class CarEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, client):
        self.client = client
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 4), dtype=np.float32)
        self.car_controls = airsim.CarControls()
        self.time = 0
        self.state = self._refresh_state()

    def reset(self):
        self.client.reset()
        self.time = 0

    def step(self, action):        
        state = self._refresh_state()
        a = self._interpret_action(action)
        self.client.setCarControls(a)
        reward = self._compute_reward(self.state)
        observation = self._get_obvervation()

        # Check if done
        done = False
        if reward < -1:
            done = self.time >= 10
        if self.car_controls.brake == 0:
            if state["car_state"].speed <= 4:
                done = self.time >= 10
        self.time += 1
        
        return observation, reward, done, {}

    def render(self):
        pass

    def close(self):
        self.client.reset()
        self.client.enableApiControl(False)
        self.client.armDisarm(False)

    def _refresh_state(self):
        car_state = self.client.getCarState()
        state = {
            "car_state": car_state,
        }
        self.state = state 
        return state

    def _get_obvervation(self):
        res_depth_plannar, res_scene = client.simGetImages([
            airsim.ImageRequest("1", airsim.ImageType.DepthPlanar, True),
            airsim.ImageRequest("2", airsim.ImageType.Scene, False, False),
        ])
        scene = np.frombuffer(res_scene.image_data_uint8, dtype=np.uint8)
        scene = scene.reshape(res_scene.height, res_scene.width, 3).astype(np.float32)
        planner = np.array(res_depth_plannar.image_data_float, dtype=np.float32)
        planner = planner.reshape(res_depth_plannar.height, res_depth_plannar.width, 1)
        planner /= MAX_DEPTH / 255.0 
        observation = np.concatenate((scene, planner), axis=-1)
        return observation

    def _interpret_action(self, action):
        self.car_controls.brake = 0
        self.car_controls.throttle = 1
        if action == 0:
            self.car_controls.throttle = 0
            self.car_controls.brake = 1
        elif action == 1:
            self.car_controls.steering = 0
        elif action == 2:
            self.car_controls.steering = 0.5
        elif action == 3:
            self.car_controls.steering = -0.5
        elif action == 4:
            self.car_controls.steering = 0.25
        else:
            self.car_controls.steering = -0.25
        return self.car_controls

    def _compute_reward(self, state):
        
        car_state = state["car_state"]

        thresh_dist = 3.5
        beta = 3

        z = 0
        pts = [np.array([0, -1, z]), np.array([130, -1, z]), np.array([130, 125, z]), np.array([0, 125, z]), np.array([0, -1, z]), np.array([130, -1, z]), np.array([130, -128, z]), np.array([0, -128, z]), np.array([0, -1, z])]
        pd: airsim.Vector3r = car_state.kinematics_estimated.position
        car_pt = np.array([pd.x_val, pd.y_val, pd.z_val])

        dist = 10000000
        for i in range(0, len(pts)-1):
            dist = min(dist, np.linalg.norm(np.cross((car_pt - pts[i]), (car_pt - pts[i+1])))/np.linalg.norm(pts[i]-pts[i+1]))

        #print(dist)
        if dist > thresh_dist:
            reward = -3
        else:
            reward_dist = (np.exp(-beta*dist) - 0.5)
            reward_speed = (((car_state.speed - MIN_SPEED)/(MAX_SPEED - MIN_SPEED)) - 0.5)
            reward = reward_dist + reward_speed

        return reward


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

   