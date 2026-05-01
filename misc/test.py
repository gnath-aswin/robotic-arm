from grasp_env import GraspEnv
import mujoco
import time

model = mujoco.MjModel.from_xml_path("scene_with_objects.xml")
env = GraspEnv(model, render=True)
obs, _ = env.reset()

for _ in range(5000):
    time.sleep(0.005)
    action = env.action_space.sample()  # random
    obs, reward, done, truncated, info = env.step(action)

    if done or truncated:
        obs, _ = env.reset()