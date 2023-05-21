import airsim
import numpy as np
import gym

from settings import MAX_DEPTH, MAX_SPEED, MIN_SPEED, MAX_SPEED_REWARD

class CarEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, client, frame_rate=32):
        self.client = client
        self.frame_rate = frame_rate
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(4*frame_rate, 144, 256), dtype=np.float32)
        self.car_controls = airsim.CarControls()
        self.time = 0
        self.state = None
        self._refresh_state()
        self.observation_buffer = None

    def reset(self):
        self.client.reset()
        self.time = 0
        self.observation_buffer = None
        return self._get_obvervation()

    def step(self, action):        
        state = self._refresh_state()
        a = self._interpret_action(action)
        self.client.setCarControls(a)
        reward = self._compute_reward(self.state)
        observation = self._get_obvervation()

        # Check if done
        done = False
        if reward < -1:
            done = True
        if state["speed_moving_avg"] <= MIN_SPEED:
            done = self.time >= 10
            if done:
                reward = -10
        print(f"time={self.time}, reward={reward}, speed={state['speed_moving_avg']}, action={action}, done={done}")
        self.time += 1
        
        return observation, reward, done, {}

    def render(self):
        pass

    def close(self):
        self.client.reset()
        self.client.enableApiControl(False)
        self.client.armDisarm(False)

    def _refresh_state(self, speed_moving_avg_decay=0.1):
        prev_state = self.state
        car_state = self.client.getCarState()
        state = {
            "car_state": car_state,
            "speed_moving_avg": MIN_SPEED if prev_state is None else speed_moving_avg_decay * prev_state["speed_moving_avg"] + (1 - speed_moving_avg_decay) * car_state.speed,
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
        if self.observation_buffer is None:
            self.observation_buffer = np.repeat(observation, self.frame_rate, axis=-1)
        else:
            self.observation_buffer = np.concatenate((self.observation_buffer[:, :, 4:], observation), axis=-1)
        observation = np.moveaxis(self.observation_buffer, -1, 0)

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
            reward_speed = min(reward_speed, MAX_SPEED_REWARD)
            reward = reward_speed + reward_dist

        return reward
