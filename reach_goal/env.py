# pyright: reportUnboundVariable=false
# pyright: reportAttributeAccessIssue=false
# env.py
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from config import CONFIG


class ReachEnv(gym.Env):
    """
    MuJoCo reaching environment using joint position delta control.

    Action:
        The policy outputs normalized joint position deltas.

        action[i] in [-1, 1]

        Internally:
            q_target[i] = q_current[i] + action[i] * delta_scale[i]
            ctrl[i] = q_target[i]

        MuJoCo position actuators track q_target.

    Observation:
        ee_pos      : 3
        ee_vel      : 3
        abs_goal    : 3
        rel_goal    : 3
        distance    : 1
        qpos        : num_joints
        qvel        : num_joints

        For 6 joints:
            obs_dim = 3 + 3 + 3 + 3 + 1 + 6 + 6 = 25
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        model: mujoco.MjModel,
        seed: int | None = None,
        ee_body_name: str = "tool0",
    ):
        super().__init__()

        self.model = model
        self.data = mujoco.MjData(model)

        self.seed_value = seed
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.ee_body_name = ee_body_name
        self.ee_body_id = self.model.body(self.ee_body_name).id

        self.joint_names = CONFIG["joint_names"]
        self.num_joints = len(self.joint_names)

        self.joint_mins = CONFIG["joint_mins"].astype(np.float64)
        self.joint_maxs = CONFIG["joint_maxs"].astype(np.float64)
        self.delta_scale = CONFIG["delta_scale"].astype(np.float64)

        self.workspace_min = CONFIG["workspace_min"].astype(np.float64)
        self.workspace_max = CONFIG["workspace_max"].astype(np.float64)

        self.reward_cfg = CONFIG["reward"]

        # These can be overwritten from your training config:
        # env.success_threshold = config["env"]["success_threshold"]
        self.max_steps = 500
        self.success_threshold = self.reward_cfg.get("success_threshold", 0.02)

        self.goal: np.ndarray | None = None
        self.prev_distance: float | None = None
        self.step_count = 0 

        # --------------------------------------------------
        # Joint indexing
        # --------------------------------------------------
        # This uses joint names from your JSON/config.
        # Much safer than assuming np.arange(6).
        self.qpos_indices = self._get_qpos_indices_from_joint_names()
        self.qvel_indices = self._get_qvel_indices_from_joint_names()

        # --------------------------------------------------
        # Action space
        # --------------------------------------------------
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_joints,),
            dtype=np.float32,
        )

        # --------------------------------------------------
        # Observation space
        # --------------------------------------------------
        obs_dim = 3 + 3 + 3 + 3 + 1 + self.num_joints + self.num_joints

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # --------------------------------------------------
        # Sampling space
        self.goal_radius = 0.20
        self.min_goal_distance = 0.08
        # --------------------------------------------------


    def reset(self, *, seed: int | None = None): 
        super().reset(seed=seed)

        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        mujoco.mj_resetData(self.model, self.data)

        self.step_count = 0
        self.prev_distance = None

        # Optional custom reset
        # self._reset_robot_to_mid_joint_configuration()

        # For position control, initialize actuator targets
        # to the current joint positions.
        self.data.ctrl[0:6] = self.data.qpos[0:6].copy()

        self._freeze_gripper_open()

        mujoco.mj_forward(self.model, self.data)

        self.goal = self._sample_goal()

        ee_pos = self._get_ee_pos()
        distance = self._compute_distance(ee_pos)

        self.prev_distance = distance

        obs = self._get_obs()

        info = {
            "goal": self.goal.copy(),
            "ee_pos": ee_pos.copy(),
            "distance": distance,
        }

        return obs, info
 
    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, -1.0, 1.0)

        # ------------------------
        # Joint position delta control
        # ------------------------
        q_current = self.data.qpos[0:6].copy()

        q_target = q_current + action * self.delta_scale
        q_target = np.clip(
            q_target,
            self.joint_mins,
            self.joint_maxs,
        )

        # For MuJoCo position actuators:
        # ctrl[0:6] are target joint positions, not velocities.
        self.data.ctrl[0:6] = q_target

        # Keep gripper fixed/open.
        self._freeze_gripper_open()

        mujoco.mj_step(self.model, self.data)

        # Keep gripper frozen after physics step too.
        self._freeze_gripper_open()

        self.step_count += 1

        # Optional safety clamp.
        # With position actuators + clipped q_target, this should rarely be needed,
        # but it is okay to keep during debugging.
        self._enforce_joint_limits()

        ee_pos = self._get_ee_pos()
        distance = self._compute_distance(ee_pos)

        reward, reward_info = self._compute_reward(
            distance=distance,
            action=action,
        )

        terminated = distance < self.success_threshold
        truncated = self.step_count >= self.max_steps

        obs = self._get_obs()

        info = {
            "distance": distance,
            "is_success": terminated,
            "goal": self.goal.copy(),
            "ee_pos": ee_pos.copy(),
            "q_current": q_current.copy(),
            "q_target": q_target.copy(),
            "q_actual": self.data.qpos[0:6].copy(),
            "step_count": self.step_count,
            **reward_info,
        }

        return obs, reward, terminated, truncated, info

    # ======================================================
    # Observation
    # ======================================================

    def _get_obs(self) -> np.ndarray:
        assert self.goal is not None, "Goal is None. Call reset() before step()."

        ee_pos = self._get_ee_pos()
        ee_vel = self._get_ee_linear_velocity()

        goal = self.goal
        rel_goal = goal - ee_pos
        distance = np.linalg.norm(rel_goal)

        qpos = self.data.qpos[self.qpos_indices]
        qvel = self.data.qvel[self.qvel_indices]

        obs = np.concatenate(
            [
                ee_pos,
                ee_vel,
                goal,
                rel_goal,
                np.array([distance], dtype=np.float64),
                qpos,
                qvel,
            ]
        )

        return obs.astype(np.float32)

    # ======================================================
    # Reward
    # ======================================================

    def _compute_reward(
        self,
        distance: float,
        action: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        cfg = self.reward_cfg

        if self.prev_distance is None:
            self.prev_distance = distance

        progress = self.prev_distance - distance

        reward_distance = cfg["distance_weight"] * distance
        reward_progress = cfg["progress_weight"] * progress
        reward_action = cfg["action_penalty_weight"] * np.linalg.norm(action)

        reward_precision = (
            cfg.get("precision_weight", 1.0)
            * np.exp(-cfg["precision_exp_scale"] * distance)
        )

        reward_success = 0.0
        if distance < self.success_threshold:
            reward_success = cfg["success_bonus"]

        reward = (
            reward_distance
            + reward_progress
            + reward_action
            + reward_precision
            + reward_success
        )

        self.prev_distance = distance

        reward_info = {
            "reward_distance": float(reward_distance),
            "reward_progress": float(reward_progress),
            "reward_action": float(reward_action),
            "reward_precision": float(reward_precision),
            "reward_success": float(reward_success),
            "progress": float(progress),
        }

        return float(reward), reward_info

    # ======================================================
    # Goal
    # ======================================================

    def _sample_goal(self) -> np.ndarray:
        ee_pos = self._get_ee_pos()

        radius = self.goal_radius
        min_goal_distance = self.min_goal_distance

        for _ in range(100):
            offset = self.np_random.uniform(
                low=np.array([-radius, -radius, -radius]),
                high=np.array([radius, radius, radius]),
            )

            goal = ee_pos + offset
            goal = np.clip(goal, self.workspace_min, self.workspace_max)

            if np.linalg.norm(goal - ee_pos) > min_goal_distance:
                return goal.astype(np.float64)

        for _ in range(100):
            goal = self.np_random.uniform(
                low=self.workspace_min,
                high=self.workspace_max,
            )

            if np.linalg.norm(goal - ee_pos) > min_goal_distance:
                return goal.astype(np.float64)

        return self.workspace_max.astype(np.float64)

    # ======================================================
    # Robot control
    # ======================================================

    def _enforce_joint_limits(self):
        qpos = self.data.qpos[self.qpos_indices].copy()
        qvel = self.data.qvel[self.qvel_indices].copy()

        clipped_qpos = np.clip(
            qpos,
            self.joint_mins,
            self.joint_maxs,
        )

        lower_hit = clipped_qpos <= self.joint_mins
        upper_hit = clipped_qpos >= self.joint_maxs

        # Stop velocity that pushes beyond limits.
        qvel[lower_hit & (qvel < 0.0)] = 0.0
        qvel[upper_hit & (qvel > 0.0)] = 0.0

        self.data.qpos[self.qpos_indices] = clipped_qpos
        self.data.qvel[self.qvel_indices] = qvel

        mujoco.mj_forward(self.model, self.data)

    def _reset_robot_to_mid_joint_configuration(self):
        mid_qpos = 0.5 * (self.joint_mins + self.joint_maxs)

        self.data.qpos[self.qpos_indices] = mid_qpos
        self.data.qvel[self.qvel_indices] = 0.0


    def _freeze_gripper_open(self):
        """
        Keep gripper open and immobile.

        From MuJoCo debug:
            gripper_left  -> qpos[6], qvel[6], ctrl[6]
            gripper_right -> qpos[7], qvel[7], ctrl[7]
        """

        gripper_qpos_indices = np.array([6, 7], dtype=np.int64)
        gripper_qvel_indices = np.array([6, 7], dtype=np.int64)
        gripper_ctrl_indices = np.array([6, 7], dtype=np.int64)

        gripper_open_qpos = np.array([0.0, 0.0], dtype=np.float64)
        gripper_open_ctrl = np.array([0.0, 0.0], dtype=np.float64)

        self.data.qpos[gripper_qpos_indices] = gripper_open_qpos
        self.data.qvel[gripper_qvel_indices] = 0.0
        self.data.ctrl[gripper_ctrl_indices] = gripper_open_ctrl   


    # ======================================================
    # MuJoCo helpers
    # ======================================================

    def _get_qpos_indices_from_joint_names(self) -> np.ndarray:
        qpos_indices = []

        for joint_name in self.joint_names:
            joint_id = self.model.joint(joint_name).id
            qpos_adr = self.model.jnt_qposadr[joint_id]
            qpos_indices.append(qpos_adr)

        return np.array(qpos_indices, dtype=np.int64)

    def _get_qvel_indices_from_joint_names(self) -> np.ndarray:
        qvel_indices = []

        for joint_name in self.joint_names:
            joint_id = self.model.joint(joint_name).id
            qvel_adr = self.model.jnt_dofadr[joint_id]
            qvel_indices.append(qvel_adr)

        return np.array(qvel_indices, dtype=np.int64)

    def _get_ee_pos(self) -> np.ndarray:
        return self.data.xpos[self.ee_body_id].copy()

    def _get_ee_linear_velocity(self) -> np.ndarray:
        """
        MuJoCo cvel format:
            cvel[body_id, 0:3] = angular velocity
            cvel[body_id, 3:6] = linear velocity
        """
        return self.data.cvel[self.ee_body_id, 3:6].copy()

    def _compute_distance(self, ee_pos: np.ndarray) -> float:
        assert self.goal is not None, "Goal is None. Call reset() first."
        return float(np.linalg.norm(ee_pos - self.goal))


def main():
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    import numpy as np
    import torch
    import random


    seed = 42

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    model = mujoco.MjModel.from_xml_path("scenes/scene.xml")
    env = ReachEnv(model, seed=seed)
    env = Monitor(env)

    ppo = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        verbose=1,
        learning_rate=4e-4,
        ent_coef=0.00,
        clip_range=0.3,
        n_steps=1024,
        batch_size=128,
        tensorboard_log="./logs/", 
        device="cpu"
    )

    ppo.learn(total_timesteps=2000)
    ppo.save("ppo_reach")

if __name__ == "__main__":
    main()
