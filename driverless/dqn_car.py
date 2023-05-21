# ready to run example: PythonClient/multirotor/hello_drone.py
import airsim
import os
import numpy as np
from PIL import Image
import logging

NUM_EPISODES = 5
MAX_SPEED = 300
MIN_SPEED = 10

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

# connect to the AirSim simulator
client = airsim.CarClient()
client.reset()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

car_controls = airsim.CarControls()

def interpret_action(action):
    car_controls.brake = 0
    car_controls.throttle = 1
    if action == 0:
        car_controls.throttle = 0
        car_controls.brake = 1
    elif action == 1:
        car_controls.steering = 0
    elif action == 2:
        car_controls.steering = 0.5
    elif action == 3:
        car_controls.steering = -0.5
    elif action == 4:
        car_controls.steering = 0.25
    else:
        car_controls.steering = -0.25
    return car_controls

def compute_reward(car_state):

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

for eps_i in range(NUM_EPISODES):
    client.reset()

    i = 0
    done = False
    logging.info(f"Episode {eps_i}")

    while not done:
        i += 1
        # obtain state and image and relevant info to compute reward
        action = np.random.randint(0, 6)
        car_controls = interpret_action(action)
        client.setCarControls(car_controls)
        
        # get camera images from the car
        res_depth_vis, res_depth_plannar, res_scene = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthVis, True, False),
            airsim.ImageRequest("1", airsim.ImageType.DepthPlanar, True),
            airsim.ImageRequest("2", airsim.ImageType.Scene, False, False),
        ])

        depth_vis = np.array(res_depth_vis.image_data_float, dtype=np.float32)
        depth_vis = depth_vis.reshape(res_depth_vis.height, res_depth_vis.width)
        scene = np.frombuffer(res_scene.image_data_uint8, dtype=np.uint8)
        scene = scene.reshape(res_scene.height, res_scene.width, 3)
        planner = np.array(res_depth_plannar.image_data_float, dtype=np.float32)
        planner = planner.reshape(res_depth_plannar.height, res_depth_plannar.width)

        # tranform and save images in temp enumerated by i
        Image.fromarray(scene).save(os.path.normpath('./temp/scene_' + str(i) + '.png'))
        Image.fromarray(depth_vis.astype("uint8")).save(os.path.normpath('./temp/depth_vis_' + str(i) + '.png'))
        Image.fromarray(planner.astype("uint8")).save(os.path.normpath('./temp/planner_' + str(i) + '.png'))
        state = client.getCarState()
        
        # print state details
        reward = compute_reward(state)
        logging.info(f"speed={state.speed}, gear={state.gear}, collision={state.collision.has_collided}, timestamp={state.timestamp}, reward={reward}")

        # Check 
        if reward < -1:
            done = i >= 10
        if car_controls.brake == 0:
            if state.speed <= 5:
                done = i >= 10