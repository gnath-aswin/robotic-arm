import numpy as np
import mujoco
from config import CONFIG

# -------------------------
# INVERSE KINEMATICS
# -------------------------
def damped_least_squares(J, error, damping=0.02):
    J_T = J.T
    JJ_T = J @ J_T
    lambda_I = (damping ** 2) * np.eye(JJ_T.shape[0])
    inv = np.linalg.inv(JJ_T + lambda_I)
    dq = J_T @ inv @ error
    return dq


# Inverse Kinematics Controller
def inverse_kinematics_step(model, data, target_pos, body_name="tool0",
                           joint_indices=None,
                           joint_mins=CONFIG["joint_mins"],
                           joint_maxs=CONFIG["joint_maxs"],
                           gain=0.01,
                           max_step=0.002,
                           damping=0.02):
    """
    One step of Damped Least Squares IK (position only)

    Args:
        model, data : MuJoCo model/data
        target_pos  : np.array (3,) in world frame
        body_name   : end-effector body (default = tool0)
        joint_indices : indices of controlled joints
        joint_mins/maxs : joint limits
        gain        : scaling factor for stability
        max_step    : max joint step (rad)
        damping     : DLS damping factor

    Returns:
        dq : joint update
        error : position error
    """

    # Default: first 6 joints
    if joint_indices is None:
        joint_indices = np.arange(6)

    # End-effector ID
    ee_id = model.body(body_name).id

    # Current EE position
    ee_pos = data.xpos[ee_id]

    # Position error
    error = target_pos - ee_pos

    # Jacobian
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, ee_id)

    # Select relevant joints
    J = jacp[:, joint_indices]

    # Damped Least Squares IK
    dq = damped_least_squares(J, error, damping)

    # Stabilization
    dq = gain * dq
    dq = np.clip(dq, -max_step, max_step)

    # Set control (position control)
    data.ctrl[joint_indices] += dq

    # Clip to limits SAFE JOINT UPDATE (IMPORTANT)
    if joint_mins is not None and joint_maxs is not None:
        data.ctrl[joint_indices] = np.clip(
            data.ctrl[joint_indices],
            joint_mins,
            joint_maxs
        )

    return dq, error


# Action layer
def step_cartesian_action(model, data, action):

    # Limits
    max_delta = 0.005
    workspace_min = CONFIG["workspace_min"]
    workspace_max = CONFIG["workspace_max"]

    # clip action
    action = np.clip(action, -max_delta, max_delta)

    # current EE position
    ee_id = model.body("tool0").id
    ee_pos = data.xpos[ee_id]

    # compute target
    target_pos = ee_pos + action

    # clip workspace
    target_pos = np.clip(target_pos, workspace_min, workspace_max)

    # IK step
    dq, error = inverse_kinematics_step(model, data, target_pos)

    return target_pos, error



def set_gripper(model, data, command, max_step=0.001):
    """
    High-level gripper control

    Args:
        command: str -> "open" or "close"
    """
    # Joint indices
    gripper_left_id = model.joint("gripper_left").id
    gripper_right_id = model.joint("gripper_right").id
    gripper_indices = CONFIG.get("gripper_indices", [gripper_left_id, gripper_right_id])

    grip_min = CONFIG["gripper"]["min"]  # -0.04 (closed)
    grip_max = CONFIG["gripper"]["max"]  #  0.00 (open)

    if command == "close":
        target = grip_min
    elif command == "open":
        target = grip_max
    else:
        raise ValueError("Command must be 'open' or 'close'")

    # APPLY CONTROL
    for idx in gripper_indices:
            current = data.ctrl[idx]
            delta = target - current
            step = np.clip(delta, -max_step, max_step)
            data.ctrl[idx] += step


# State check
def is_gripper_closed(data, tol=1e-3):
    gripper_indices = CONFIG.get("gripper_indices", [6, 7])
    grip_min = CONFIG["gripper"]["min"]

    vals = data.ctrl[gripper_indices]
    return np.all(np.abs(vals - grip_min) < tol)


def is_gripper_open(data, tol=1e-3):
    gripper_indices = CONFIG.get("gripper_indices", [6, 7])
    grip_max = CONFIG["gripper"]["max"]

    vals = data.ctrl[gripper_indices]
    return np.all(np.abs(vals - grip_max) < tol)

# -------------------------
# GRIPPER CONTROLLER
# -------------------------
def move_with_ppo(model, data, ppo, target):

    ee_id = model.body("tool0").id
    ee_pos = data.xpos[ee_id]

    rel_goal = target - ee_pos
    distance = np.linalg.norm(rel_goal)

    obs = np.concatenate([ee_pos, rel_goal, [distance]]).astype(np.float32)

    action, _ = ppo.predict(obs, deterministic=True)

    target, error = step_cartesian_action(model, data, action)

    return target, error