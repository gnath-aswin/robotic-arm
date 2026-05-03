import json
from pathlib import Path
import numpy as np

CONFIG_PATH = Path(__file__).parent / "trlc_dk1_constraints.json"

with open(CONFIG_PATH, "r") as f:
    _cfg = json.load(f)


# ------------------------
# Joint limits
# ------------------------

joint_names = list(_cfg["joints"].keys())

joint_mins = np.array(
    [_cfg["joints"][j]["min"] for j in joint_names],
    dtype=np.float64,
)

joint_maxs = np.array(
    [_cfg["joints"][j]["max"] for j in joint_names],
    dtype=np.float64,
)

joint_max_vels = np.array(
    [_cfg["joints"][j]["max_vel"] for j in joint_names],
    dtype=np.float64,
)

num_joints = len(joint_names)


# ------------------------
# Workspace
# ------------------------

workspace_min = np.array(
    [
        _cfg["workspace"]["x"][0],
        _cfg["workspace"]["y"][0],
        _cfg["workspace"]["z"][0],
    ],
    dtype=np.float64,
)

workspace_max = np.array(
    [
        _cfg["workspace"]["x"][1],
        _cfg["workspace"]["y"][1],
        _cfg["workspace"]["z"][1],
    ],
    dtype=np.float64,
)


# ------------------------
# Action
# ------------------------

raw_action_cfg = _cfg.get("action", {})

action_type = raw_action_cfg.get("type", "joint_position_delta")
action_normalized = raw_action_cfg.get("normalized", True)

delta_scale = np.array(
    raw_action_cfg.get(
        "delta_scale",
        [0.03, 0.03, 0.03, 0.05, 0.05, 0.05],
    ),
    dtype=np.float64,
)

if delta_scale.shape != (num_joints,):
    raise ValueError(
        f"delta_scale must have shape ({num_joints},), "
        f"but got {delta_scale.shape}"
    )


# ------------------------
# Gripper
# ------------------------

gripper = _cfg.get("gripper", {})


# ------------------------
# Reward
# ------------------------

reward = _cfg.get("reward", {})


# ------------------------
# Export CONFIG
# ------------------------

CONFIG = {
    "joint_names": joint_names,
    "num_joints": num_joints,

    "joint_mins": joint_mins,
    "joint_maxs": joint_maxs,
    "joint_max_vels": joint_max_vels,

    "workspace_min": workspace_min,
    "workspace_max": workspace_max,

    "action_type": action_type,
    "action_normalized": action_normalized,
    "delta_scale": delta_scale,

    "gripper": gripper,
    "reward": reward,
}
