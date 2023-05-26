
from settings import MODELS_DIR
import logging
from stable_baselines3 import A2C, PPO
from monitoring import init_callback, init_experiment
from lidar_env import LidarWrapper
from stable_baselines3.common.vec_env import DummyVecEnv

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

TIMESTEPS = 1000000

env = LidarWrapper()
env = DummyVecEnv([lambda: env])
env.reset()

pretrained_model = None 

# tracmania model
model = A2C(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=f"tensorboard/",
    device='cuda',
    policy_kwargs=dict(net_arch=[dict(pi=[128, 128, 128], vf=[128, 128, 128])]),
)

# load pretrained model
if pretrained_model:
    model = model.__class__.load(pretrained_model, env=env, device='cuda')
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
    model.save(f"{MODELS_DIR}/{model.__class__}-{TIMESTEPS*iters}")
    logging.info(f"Saved model at {MODELS_DIR}/{TIMESTEPS*iters}")