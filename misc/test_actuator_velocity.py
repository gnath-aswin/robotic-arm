# test_velocity_actuators.py

import time
import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("scenes/scene.xml")
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)

print("nu:", model.nu)
print("nq:", model.nq)
print("nv:", model.nv)

with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0

    while viewer.is_running():
        data.ctrl[:] = 0.0

        # Move only joint1 slowly
        data.ctrl[0] = -1.5

        # Keep gripper open/frozen
        data.ctrl[6:8] = 0.0
        data.qpos[6:8] = 0.0
        data.qvel[6:8] = 0.0

        mujoco.mj_step(model, data)

        if step % 100 == 0:
            print(
                "qpos arm:", data.qpos[:6],
                "qvel arm:", data.qvel[:6],
                "ctrl:", data.ctrl[:8],
            )

        viewer.sync()
        time.sleep(0.0001)
        step += 1
