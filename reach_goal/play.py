
# visualize_policy.py

# pyright: reportAttributeAccessIssue=false

import time
import numpy as np
import mujoco
import mujoco.viewer

from stable_baselines3 import PPO

from env import ReachEnv


MODEL_XML_PATH = "scene.xml"
POLICY_PATH = (
    "/home/void/custom_robot/reach_goal/reach_1/best_model/"
    "best_model"
)

MAX_STEPS = 10000
SUCCESS_THRESHOLD = 0.05


def add_goal_marker(viewer, goal_pos, radius=0.02):
    """
    Add a red sphere marker at the goal position.

    This uses viewer.user_scn, which is meant for custom visualization geoms.
    """

    scene = viewer.user_scn

    if scene.ngeom >= scene.maxgeom:
        print("Warning: user scene geom limit reached. Cannot add goal marker.")
        return

    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, 0.0, 0.0]),
        pos=np.asarray(goal_pos, dtype=np.float64),
        mat=np.eye(3).flatten(),
        rgba=np.array([1.0, 0.0, 0.0, 1.0]),  # red
    )

    scene.ngeom += 1


def reset_episode(env: ReachEnv):
    """
    Reset the RL environment and return the first observation.
    """

    obs, info = env.reset()

    print("\nRESET")
    print(f"Goal: {info['goal']}")
    print(f"Initial EE pos: {info['ee_pos']}")
    print(f"Initial distance: {info['distance']:.4f}")

    return obs


def main():
    # -------------------------
    # Load MuJoCo model + env
    # -------------------------

    model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)

    env = ReachEnv(model)
    env.max_steps = MAX_STEPS
    env.success_threshold = SUCCESS_THRESHOLD

    # -------------------------
    # Load trained PPO policy
    # -------------------------

    ppo = PPO.load(POLICY_PATH, device="cpu")

    # -------------------------
    # Start viewer
    # -------------------------

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        obs = reset_episode(env)

        while viewer.is_running():
            # Clear previous custom markers.
            viewer.user_scn.ngeom = 0

            # Add goal marker.
            if env.goal is not None:
                add_goal_marker(viewer, env.goal)

            # Predict action from trained policy.
            action, _ = ppo.predict(obs, deterministic=True)

            # Step environment.
            obs, reward, done, truncated, info = env.step(action)

            print(
                f"step: {env.step_count:04d} | "
                f"distance: {info['distance']:.4f} | "
                f"reward: {reward:.4f} | "
                f"success: {info['is_success']}"
            )

            # Sync viewer.
            viewer.sync()

            # Slow down visualization if needed.
            time.sleep(0.05)

            # Reset if episode ends.
            if done or truncated:
                obs = reset_episode(env)


if __name__ == "__main__":
    main()
