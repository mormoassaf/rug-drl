import wandb 
from datetime import datetime
import pandas as pd
import numpy as np
import os
from wandb.integration.sb3 import WandbCallback

experiment_log = []

# function version
def save_experiment(outfolder="./runs"):
    run_name = "run-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(outfolder, exist_ok=True)
    os.makedirs(os.path.join(outfolder, run_name), exist_ok=True)
    keys = experiment_log[0].keys()
    for key in keys:
        # save if numerical data
        if isinstance(experiment_log[0][key], (int, float, np.float32, np.float64, np.int32, np.int64)):
            np.save(os.path.join(outfolder, run_name, f"{key}.npy"), np.array([log.get(key, None) for log in experiment_log]))

    df = pd.DataFrame(experiment_log)
    df.to_csv(os.path.join(outfolder, run_name, "log.csv"), index=False)

    print(f"Experiment saved to {os.path.join(outfolder, run_name)}")

def init_experiment(config):
    wandb.init(
        project="rug-drl-driverless",
        config=config,
    )

def terminate_experiment(save=True):
    wandb.finish()
    if save:
        save_experiment()

def log(data):
    experiment_log.append(data)
    save_experiment()
    print("; ".join([f"{k}: {v}" for k, v in data.items()]))
    wandb.log(data)


init_callback = lambda: WandbCallback(
    verbose=1,
)