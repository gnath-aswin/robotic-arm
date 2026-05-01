import os
import json
import numpy as np
import torch
import random

import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import LinearSchedule

from grasp_env import GraspEnv   


# -------------------------
# CONFIG
# -------------------------

CONFIG = {
    "run_name": "ppo_grasp",

    "training": {
        "learning_rate": 1e-4,      # fine-tuning
        "ent_coef": 0.01,       # exploration
        "clip_range": 0.1,
        "vf_coef":0.3,
        "n_steps": 512,
        "batch_size": 64,
        "total_timesteps": 800_000
    },

    "env": {
        "max_steps": 1000,
        "success_threshold": 0.05
    },

    "seed": 42, # ****Most important parameter******

    # Load checkpoint if exists
    "load_model": True,
    "model_path": "best_model/best_model.zip"
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
    # ------------------------
    model = mujoco.MjModel.from_xml_path("scene_with_objects.xml")

    env = GraspEnv(model)
    env.max_steps = config["env"]["max_steps"]
    env = Monitor(env)

    eval_env = GraspEnv(model)
    eval_env.max_steps = config["env"]["max_steps"]
    eval_env = Monitor(eval_env)

    # Train Environment
    def make_env():
        model = mujoco.MjModel.from_xml_path("scene_with_objects.xml")
        env = GraspEnv(model)
        env.max_steps = config["env"]["max_steps"]
        return Monitor(env)
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    

    # Evaluation Environment
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    

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
    if config["load_model"] and os.path.exists(os.path.join(run_dir,config["model_path"])):
        from stable_baselines3.common.logger import configure
        print("Loading existing model...")
        ppo = PPO.load(os.path.join(run_dir,config["model_path"]), env=env)

        # Override params for fine-tuning
        ppo.learning_rate = config["training"]["learning_rate"]
        ppo.ent_coef = config["training"]["ent_coef"]
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
            vf_coef=config["training"]["vf_coef"],
            clip_range=config["training"]["clip_range"],
            n_steps=config["training"]["n_steps"],
            batch_size=config["training"]["batch_size"],
            tensorboard_log=f"{run_dir}/tb/",
            device="cpu",
            policy_kwargs = dict(log_std_init=-2,)
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