
from settings import NUM_EPISODES, MODELS_DIR
import airsim
import os
import numpy as np
from PIL import Image
import logging
from car_env import CarEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
from cnn_policy import CNNFeatureExtractor, ResNetFeatureExtractor, SemanticSegFormerFeatureExtractor, LightCNNFeatureExtractor, MobileNetV2FeatureExtractor
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from monitoring import init_callback, init_experiment

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

TIMESTEPS = 1000

# connect to the AirSim simulator
client = airsim.CarClient()
client.reset()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

env = CarEnv(client)
env.reset()
pretrained_model = "./models/145000.zip"

# create custom model and learn using conv
model = A2C(
    policy=ActorCriticCnnPolicy,
    policy_kwargs={
        "net_arch": [16, 16],
        "activation_fn": th.nn.ReLU,
        "ortho_init": True,
        "normalize_images": False,
        "features_extractor_class": LightCNNFeatureExtractor,
        "features_extractor_kwargs": dict(features_dim=64),
    },
    env=env, 
    verbose=1, 
    device='cuda', 
    ent_coef=0.01,
    tensorboard_log="./tensorboard",
)

# load pretrained model
if pretrained_model:
    model = A2C.load(pretrained_model, env=env, device='cuda')
    logging.info(f"Loaded model from {pretrained_model}")


init_experiment({
    "model": "A2C",
    "env": "CarEnv",
    "timesteps": TIMESTEPS,
    "n_steps": 16,  
})

iters = 0
while True:
    iters += 1
    model.learn(
        total_timesteps=TIMESTEPS, 
        reset_num_timesteps=True, 
        log_interval=50, 
        callback=init_callback(),
    )
    model.save(f"{MODELS_DIR}/{TIMESTEPS*iters}")
    logging.info(f"Saved model at {MODELS_DIR}/{TIMESTEPS*iters}")

# for eps_i in range(NUM_EPISODES):

#     i = 0
#     done = False
#     logging.info(f"Episode {eps_i}")
#     env.reset()

#     while not done:
#         i += 1
#         # obtain state and image and relevant info to compute reward
#         action = np.random.randint(1, 6)
#         observation, reward, done, info = env.step(action)
#         scene = observation[:, :, :3].astype("uint8")
#         planner = observation[:, :, 3].astype("uint8")

   
#         state = env.state["car_state"]
#         logging.info(f"speed={state.speed}, gear={state.gear}, collision={state.collision.has_collided}, timestamp={state.timestamp}, reward={reward}")

   