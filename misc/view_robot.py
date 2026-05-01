import cv2
import mujoco
import mujoco.viewer
import numpy as np
from interpolation import interpolate_cubic, circle_trajectory
import matplotlib.pyplot as plt
from controller import inverse_kinematics_step, set_gripper
import time

model = mujoco.MjModel.from_xml_path("scene_with_objects.xml")
data = mujoco.MjData(model)


def render_camera(renderer, data):
    renderer.update_scene(data, camera="camera_joint")
    image = renderer.render()
    return image


def show_image(image):

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Camera", image_bgr)
    cv2.waitKey(1)

waypoints = [
    (0.0,  [0, 0, 0, 0, 0, 0]),
    (2.0,  [0.5, 0.2, 0.1, 0, 0, 0.5]),
    (4.0,  [1.0, 0.5, 0.3, 0.2, 0, 0]),
    (6.0,  [0.0, 0.0, 0.0, 0.0, 0, 0]),
]

renderer = mujoco.Renderer(model)
target_pos = np.array([0.3, 0.0, 0.3])

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        t = data.time

        # trajectory
        # target = interpolate_cubic(t, waypoints)
        # data.ctrl[:6] = target

        # End effector pos
        ee_id = model.body("tool0").id
        ee_pos = data.xpos[ee_id]
        target_pos = circle_trajectory(t)
        dq, error = inverse_kinematics_step(model, data, target_pos)
        set_gripper(model, data, "close")

        if round(t, 3) % 1 == 0:
            print(f"end effector pos: {ee_pos}")
            print("Error:", np.linalg.norm(error))
            print(f"target pos: {target_pos}")

            
        mujoco.mj_step(model, data)

        # render camera
        img = render_camera(renderer, data)
        show_image(img)
        time.sleep(0.01)
        viewer.sync()

