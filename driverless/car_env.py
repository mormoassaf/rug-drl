import airsim
import numpy as np
import gym

from settings import MAX_DEPTH, MAX_SPEED, MIN_SPEED

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
            done = self.time >= 20
        if self.car_controls.brake == 0:
            if state["car_state"].speed <= 4:
                done = self.time >= 20
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
        res_depth_plannar, res_scene = self.client.simGetImages([
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
