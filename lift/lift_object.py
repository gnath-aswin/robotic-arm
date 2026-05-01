import time
import numpy as np
import mujoco
import mujoco.viewer
from grasp import PickAndLiftController, Phase
from stable_baselines3 import PPO

model = mujoco.MjModel.from_xml_path("scene_with_objects.xml")
data = mujoco.MjData(model)


def randomize_object_position(model, data, object_name: str):
    body_id = model.body(object_name).id
    joint_id = model.body(object_name).jntadr[0]  # free joint index

    qpos_adr = model.jnt_qposadr[joint_id]

    x = np.random.uniform(0.3, 0.5)
    y = np.random.uniform(-0.2, 0.2)
    z = 0.0

    # Free joint
    data.qpos[qpos_adr:qpos_adr+7] = np.array([x, y, z, 1, 0, 0, 0])

def reset(model, data, controller):
    mujoco.mj_resetData(model, data)
    randomize_object_position(model, data, "cube")

    # Open gripper at start
    data.ctrl[6:8] = 0.0

    # Reset phase
    controller.phase = Phase.PRE_GRASP
    controller.grasp_counter = 0

    # Let simulation settle
    for _ in range(2000):
        mujoco.mj_step(model, data)
    print("RESET")

# load PPO if needed
ppo = PPO.load("/home/void/custom_robot/ppo_run_finetune/final_model.zip")

controller = PickAndLiftController(
    model,
    object="cube"   
)

# reset
data.ctrl[6:8] = 0.0

reset(model, data, controller)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        time.sleep(0.005)

        controller.step(model,data)
        mujoco.mj_step(model, data)
        
        if controller.phase == Phase.DONE:
            reset(model, data, controller)
        viewer.sync()
