
from settings import MODELS_DIR
import os
import numpy as np
from PIL import Image
import logging
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
from cnn_policy import LightCNNFeatureExtractor
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from monitoring import init_callback, init_experiment
from tmrl import get_environment

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

TIMESTEPS = 1000000

env = get_environment()
env.reset()

pretrained_model = None 

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


# init_experiment({
#     "model": "A2C",
#     "env": "CarEnv",
#     "timesteps": TIMESTEPS,
#     "n_steps": 16,  
# })

iters = 0
while True:
    iters += 1
    model.learn(
        total_timesteps=TIMESTEPS, 
        reset_num_timesteps=True, 
        log_interval=50, 
        # callback=init_callback(),
    )
    model.save(f"{MODELS_DIR}/a2c-{TIMESTEPS*iters}")
    logging.info(f"Saved model at {MODELS_DIR}/{TIMESTEPS*iters}")