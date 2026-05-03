# pyright: reportAttributeAccessIssue=false
import os
import json
import numpy as np
import torch
import random

import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from env import ReachEnv   


# -------------------------
# CONFIG
# -------------------------

CONFIG = {
    "run_name": "reach_goal/obs26/r020_thr005_finetune",

    "training": {
        "learning_rate": 5e-5,      # fine-tuning
        "ent_coef": 0.0005,         # reduced exploration
        "clip_range": 0.1,
        "target_kl": 0.015,
        "n_steps": 1280,
        "batch_size": 128,
        "total_timesteps": 2000_000 },

    "env": {
        "max_steps": 500,
        "success_threshold": 0.05,
        "goal_radius": 1,
        "min_goal_distance": 0.08,
        "min_lateral_distance" : 0.04
    },

    "seed": 42, # ****Most important parameter******

    # Load checkpoint if exists
    "load_model": True,
    "model_path": "reach_goal/obs26/r020_thr005_finetune/best_model/best_model.zip"
}


# -------------------------
# SEED - Never forget seed!!
# -------------------------

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def main():
    config = CONFIG
    seed = config["seed"]

    # Set seed
    set_seed(seed)

    # Create run folder
    run_dir = config["run_name"]
    os.makedirs(run_dir, exist_ok=True)

    # Save config
    with open(f"{run_dir}/config.json", "w") as f:
        json.dump(config, f, indent=4)

    # -------------------------
    # ENV
    # -------------------------

    model = mujoco.MjModel.from_xml_path("scenes/scene.xml")

    env = ReachEnv(model)
    env.max_steps = config["env"]["max_steps"]
    env.success_threshold = config["env"]["success_threshold"]
    env.min_goal_distance = config["env"]["min_goal_distance"]
    env.min_lateral_distance = config["env"]["min_goal_distance"]
    env = Monitor(env)

    eval_env = ReachEnv(model)
    eval_env.max_steps = config["env"]["max_steps"]
    eval_env.success_threshold = config["env"]["success_threshold"]
    eval_env.min_goal_distance = config["env"]["min_goal_distance"]
    eval_env.min_lateral_distance = config["env"]["min_goal_distance"]
    eval_env = Monitor(eval_env)

    # -------------------------
    # SAVE BEST
    # -------------------------

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{run_dir}/best_model/",
        log_path=f"{run_dir}/logs/",
        eval_freq=100000,
        deterministic=True,
        render=False,
    )

    # -------------------------
    # LOAD OR CREATE MODEL
    # -------------------------
    if config["load_model"] and os.path.exists(config["model_path"]):
        from stable_baselines3.common.utils import LinearSchedule
        from stable_baselines3.common.logger import configure
        print("Loading existing model...")
        ppo = PPO.load(config["model_path"], env=env, device='cpu')

        # Override params for fine-tuning
        ppo.learning_rate = config["training"]["learning_rate"]
        ppo.ent_coef = config["training"]["ent_coef"]
        ppo.target_kl = config["training"]["target_kl"]
        ppo.clip_range = LinearSchedule(config["training"]["clip_range"], config["training"]["clip_range"],0)

        # Re-enable logging
        new_logger = configure(f"{run_dir}/tb/", ["stdout", "tensorboard"])
        ppo.set_logger(new_logger)

    else:
        print("Training from scratch...")
        ppo = PPO(
            "MlpPolicy",
            env,
            seed=seed,
            verbose=1,
            learning_rate=config["training"]["learning_rate"],
            ent_coef=config["training"]["ent_coef"],
            target_kl=config["training"]["target_kl"],
            clip_range=config["training"]["clip_range"],
            n_steps=config["training"]["n_steps"],
            batch_size=config["training"]["batch_size"],
            tensorboard_log=f"{run_dir}/tb/",
            device="cpu"
        )

    # -------------------------
    # TRAIN
    # -------------------------

    ppo.learn(
        total_timesteps=config["training"]["total_timesteps"],
        callback=eval_callback
    )

    # -------------------------
    # SAVE FINAL MODEL
    # -------------------------

    ppo.save(f"{run_dir}/final_model")

    print("Training complete!")


if __name__ == "__main__":
    main()
