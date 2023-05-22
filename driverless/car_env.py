import airsim
import numpy as np
import gym
import PIL
from settings import MAX_DEPTH, MAX_SPEED, MIN_SPEED, MAX_SPEED_REWARD, CAUTON_PROXIMITY, MAX_DIST_REWARD, KILL_PENALTY, CHANNELS_PER_FRAME, FRAME_RATE

# computes reward, if metric is above threshold, reward grows linearly, otherwise it decays exponentially
def metric2reward(metric, threshold, max_reward=None, min_reward=None, m=2):

    if metric > threshold:
        reward = (metric - threshold) / m
        if max_reward is not None:
            reward = min(reward, max_reward)
    else:
        reward = (2.5/threshold) * np.power(metric - threshold, 3)
        if min_reward is not None:
            reward = max(reward, min_reward)
    return reward

class CarEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, client, frame_rate=FRAME_RATE):
        self.client = client
        self.frame_rate = frame_rate
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(CHANNELS_PER_FRAME*FRAME_RATE, 144, 256), dtype=np.float32)
        self.car_controls = airsim.CarControls()
        self.time = 0
        self.state = None
        self._refresh_state()
        self.observation_buffer = None

    def reset(self):
        self.client.reset()
        # place car at random location
        xs = np.random.randint(-10, 10)
        ys = np.random.randint(-10, 10)
        # rotate the car quaternion to face random directions
        dirs = np.random.randint(0, 360)
        self.client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(xs, 1, -1), airsim.to_quaternion(0, 0, dirs)), True)
        # move car forward
        actions = self._interpret_action(1)
        for _ in range(10):
            self.client.setCarControls(actions)
        self.time = 0
        self.observation_buffer = None
        self.state = self._refresh_state()

        return self._get_obvervation()[0]

    def step(self, action):        
        state = self._refresh_state()
        a = self._interpret_action(action)
        self.client.setCarControls(a)
        
        observation, _, lider = self._get_obvervation()
        reward = self._compute_reward(self.state, lider)
        done = False
        done = reward == -KILL_PENALTY
        if done:
            print("\tcar stopped moving. killing episode")
        self.time += 1
        
        return observation, reward, done, {
            "time": self.time,
            "speed": state["car_state"].speed,
            "speed_moving_avg": state["speed_moving_avg"],
            "throttle": a.throttle,
            "steering": a.steering,
            "brake": a.brake,
            "reward": reward,
        }

    def render(self):
        pass

    def close(self):
        self.client.reset()
        self.client.enableApiControl(False)
        self.client.armDisarm(False)

    def _refresh_state(self, speed_moving_avg_decay=0.01):
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
        scene = np.dot(scene, np.array([0.2989, 0.5870, 0.1140]))
        scene = scene.reshape(res_scene.height, res_scene.width, 1)
        
        planner = np.array(res_depth_plannar.image_data_float, dtype=np.float32)
        planner = planner.reshape(res_depth_plannar.height, res_depth_plannar.width, 1)
        # clip depth values
        planner = np.clip(planner, 0, MAX_DEPTH) / MAX_DEPTH
        planner = 1 - planner
        planner *= 255

        planner_img = PIL.Image.fromarray(planner.reshape(res_depth_plannar.height, res_depth_plannar.width).astype(np.uint8))
        planner_img.save("./temp/1.png")
        scene_img = PIL.Image.fromarray(scene.reshape(res_scene.height, res_scene.width).astype(np.uint8))
        scene_img.save("./temp/2.png")

        observation = np.concatenate((scene, 255-planner), axis=-1)
        observation = observation / 255

        # add frame to buffer
        if self.observation_buffer is None:
            self.observation_buffer = np.repeat(observation, self.frame_rate, axis=-1)
        else:
            self.observation_buffer = np.concatenate((self.observation_buffer[:, :, CHANNELS_PER_FRAME:], observation), axis=-1)
        
        observation = np.moveaxis(self.observation_buffer, -1, 0)
        
        observation = 2*observation - 1
        assert observation.min() >= -1 and observation.max() <= 1
        return observation, scene, 255-planner

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

    def _compute_reward(self, state, lidar):
        
        car_state = state["car_state"]
        if self.time <= 10 and car_state.speed < MIN_SPEED:
            print("waiting for car to move")
            return 0
        
        # check if car is looking upwards
        quaternionr = car_state.kinematics_estimated.orientation
        pitch, roll, yaw = airsim.to_eularian_angles(quaternionr)
        if pitch > 0.2:
            return -KILL_PENALTY
        
        # compute average distance of the closest 65% of the lidar points
        k = int(0.65 * lidar.size)
        avg_distance = np.average(np.sort(lidar.flatten())[:k])

        # compute caution distance by interpolating between min_speed and max_speed
        alpha = (car_state.speed / (MAX_SPEED - MIN_SPEED))
        caution_prox = (1 - alpha) * CAUTON_PROXIMITY + alpha * (CAUTON_PROXIMITY * 2)

        # compute reward
        reward_dist = metric2reward(avg_distance, caution_prox, MAX_DIST_REWARD, min_reward=-4)
        reward_speed = metric2reward(car_state.speed, MIN_SPEED, MAX_SPEED_REWARD, min_reward=-1) 

        # if car is stuck, return -KILL_PENALTY, if any 
        if state["speed_moving_avg"] < MIN_SPEED:
            reward = -KILL_PENALTY
        elif reward_dist < 0 or reward_speed < 0:
            reward = min(reward_dist, reward_speed)
        else:
            reward = reward_speed + reward_dist

        print("\033[1;34;40m d: ", "{:.2f}".format(avg_distance), "\033[0m", "d_c: ", "{:.2f}".format(caution_prox))
        print("\033[1;32;40m reward: ", "{:.2f}".format(reward), "\033[0m", "r_d: ", "{:.2f}".format(reward_dist), "r_s: ", "{:.2f}".format(reward_speed))
        
        return reward
    
