
from settings import MODELS_DIR
import logging
from stable_baselines3 import PPO
from lidar_env import LidarWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os
import wandb    
from wandb.integration.sb3 import WandbCallback

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

TIMESTEPS = 1000000

env = LidarWrapper()
env = Monitor(env)
env = DummyVecEnv([lambda: env])
env.reset()

pretrained_model = None #"./artifacts/PPO-10000.zip" 
config = {
    "policy": "MlpPolicy",
    "env": "LidarWrapper",
    "total_timesteps": TIMESTEPS,
}

run = wandb.init(
    group="rug-ai-group",
    project="rug-drl-trackmania",
    config=config,
    sync_tensorboard=True,
    monitor_gym=True,
)


### CONSTRUCT MODEL
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=f"runs/{run.id}",
    device='cuda',
    batch_size=512,
    policy_kwargs=dict(net_arch=dict(pi=[64, 64, 64], vf=[64, 64, 64])),
)

# load pretrained model
if pretrained_model:
    loaded_model = model.__class__.load(pretrained_model, env=env, device='cuda')
    model.set_parameters(loaded_model.get_parameters())
    logging.info(f"Loaded model from {pretrained_model}")


model.learn(
    total_timesteps=TIMESTEPS, 
    callback=WandbCallback(
        verbose=2,
        gradient_save_freq=100,
        model_save_path=os.path.join(MODELS_DIR, f"PPO-{run.id}"),
    ),
    progress_bar=True,  
)
