# ready to run example: PythonClient/multirotor/hello_drone.py
import airsim
import os
import numpy as np
from PIL import Image

# connect to the AirSim simulator
client = airsim.CarClient()
client.reset()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# take images
# responses = client.simGetImages([
#     airsim.ImageRequest("0", airsim.ImageType.DepthVis),
#     airsim.ImageRequest("1", airsim.ImageType.DepthPlanar, True)])

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

done = False
while not done:
    # obtain state and image and relevant info to compute reward
    action = np.random.randint(0, 6)
    state = client.getCarState()
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

    # tranform and save images in temp enumerated by i
    Image.fromarray(scene).save(os.path.normpath('./temp/scene' + str(i) + '.png'))
    
    # print state details
    print(f"speed={state.speed}, gear={state.gear}, collision={state.collision.has_collided}, timestamp={state.timestamp}")