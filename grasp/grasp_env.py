import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
from controller import inverse_kinematics_step
from config import CONFIG


class GraspEnv(gym.Env):
    def __init__(self, model, render=False, seed=None):
        super().__init__()

        self.model = model
        self.data = mujoco.MjData(model)

        self.n_joints = 6
        self.gripper_idx = 6  # adjust if needed

        # ACTION: joint velocities + gripper
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_joints + 1,), dtype=np.float32
        )

        # OBS: qpos, qvel, ee_pos, obj_pos, rel_pos, gripper
        obs_dim = self.n_joints * 2 + 3 + 3 + 3 + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.grasp_counter = 0
        self.prev_distance = None

        self.max_steps = 5000
        self.step_count = 0

        # IDs
        self.ee_id = self.model.body("tool0").id
        self.obj_body_id = self.model.body("cube").id

        self.obj_geom = self.model.geom("cube").id
        self.left_geom = self.model.geom("finger_left").id
        self.right_geom = self.model.geom("finger_right").id

        # Joint Constraints 
        self.joint_mins = CONFIG["joint_mins"]
        self.joint_maxs = CONFIG["joint_maxs"]
        self.joint_max_vels = CONFIG["joint_max_vels"]

        self.grip_min = CONFIG["gripper"]["min"]
        self.grip_max = CONFIG["gripper"]["max"]
        self.grip_vel_max = CONFIG["gripper"]["max_vel"]

        # Scale on top of the constraint
        self.joint_vel_scale = 0.05
        self.gripper_vel_scale = 0.05

        # Visualization
        self.render = render
        if self.render:
            self.viewer = mujoco.viewer.launch_passive(model, self.data)

        # Seed for initialization
        self.seed = seed
        if self.seed is not None:
            np.random.seed(seed)
  
    # -------------------------
    # RESET
    # -------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0

        # randomize object
        # obj_pos = np.array([
        #     np.random.uniform(0.3, 0.5),
        #     np.random.uniform(-0.2, 0.2),
        #     0.0
        # ])
        obj_pos = np.array([0.4, 0.0, 0.0])
        self.prev_distance = None
        self.grasp_counter = 0

        # set free joint position 
        qpos_addr = self.model.body_jntadr[self.obj_body_id]
        self.data.qpos[qpos_addr:qpos_addr+3] = obj_pos

        # open gripper
        self.data.ctrl[self.gripper_idx:self.gripper_idx+2] = 0.0

        # move EE above object (With slight noise)
        target = obj_pos + np.array([
            np.random.uniform(-0.01, 0.01),
            np.random.uniform(-0.01, 0.01),
            0.08
        ])  # pre-grasp height

        for _ in range(500):
            inverse_kinematics_step(self.model, self.data, target)
            mujoco.mj_step(self.model, self.data)

        return self._get_obs(), {}

    # -------------------------
    # STEP
    # -------------------------
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # Split action
        qdot = action[:self.n_joints] * self.joint_max_vels
        grip_action = action[-1] * self.grip_vel_max

        # Apply gripper limits
        current_grip = self.data.ctrl[self.gripper_idx]
        new_grip = current_grip + grip_action
        new_grip = np.clip(new_grip, self.grip_min, self.grip_max)
        # Apply gripper
        self.data.ctrl[self.gripper_idx:self.gripper_idx+2] = new_grip * self.gripper_vel_scale

        # Apply the joint limits
        qpos = self.data.qpos[:self.n_joints]
        for i in range(self.n_joints):
            if qpos[i] <= self.joint_mins[i] and qdot[i] < 0:
                qdot[i] = 0.0
            if qpos[i] >= self.joint_maxs[i] and qdot[i] > 0:
                qdot[i] = 0.0
        # Apply joint velocities as
        target_q = self.data.qpos[:self.n_joints] + qdot * self.joint_vel_scale
        target_q = np.clip(target_q, self.joint_mins, self.joint_maxs)

        self.data.ctrl[:self.n_joints] = target_q

        mujoco.mj_step(self.model, self.data)
        if self.render:
            self.viewer.sync()
 
        # Clamp positions (POST-STEP SAFETY)
        self.data.qpos[:self.n_joints] = np.clip(
            self.data.qpos[:self.n_joints],
            self.joint_mins,
            self.joint_maxs
        ) 

        self.step_count += 1

        obs = self._get_obs()
        reward, info = self._compute_reward()

        done = info["success"]
        truncated = self.step_count >= self.max_steps

        return obs, reward, done, truncated, {"is_success":done, }

    # -------------------------
    # OBSERVATION
    # -------------------------
    def _get_obs(self):
        qpos = self.data.qpos[:self.n_joints]
        qvel = self.data.qvel[:self.n_joints]

        ee_pos = self.data.xpos[self.ee_id]
        obj_pos = self.data.xpos[self.obj_body_id]

        rel_pos = obj_pos - ee_pos

        gripper = np.array([self.data.ctrl[self.gripper_idx]])

        return np.concatenate([
            qpos, qvel,
            ee_pos,
            obj_pos,
            rel_pos,
            gripper
        ]).astype(np.float32)

    # -------------------------
    # CONTACT DETECTION
    # -------------------------
    def _detect_contacts(self):
        left, right = False, False

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2

            if (g1 == self.left_geom and g2 == self.obj_geom) or \
               (g2 == self.left_geom and g1 == self.obj_geom):
                left = True

            if (g1 == self.right_geom and g2 == self.obj_geom) or \
               (g2 == self.right_geom and g1 == self.obj_geom):
                right = True

        return left, right

    # -------------------------
    # REWARD
    # -------------------------
    # def _compute_reward(self, action):
        # ee_pos = self.data.xpos[self.ee_id]
        # obj_pos = self.data.xpos[self.obj_body_id]

        # distance = np.linalg.norm(ee_pos - obj_pos)

        # left, right = self._detect_contacts()

        # reward = 0.0
        # # -------------------------
        # # REACH
        # # -------------------------
        # reward += -5.0 * distance

        # if self.prev_distance is None:
        #     self.prev_distance = distance
        # progress = self.prev_distance - distance
        # reward += 10.0 * progress
        # self.prev_distance = distance

        # reward += 2.0 * np.exp(-10 * distance)

        # # -------------------------
        # # CONTACT
        # # -------------------------
        # if left or right:
        #     reward += 2.0

        # # -------------------------
        # # STABLE GRASP
        # # -------------------------
        # if left and right:
        #     reward += 20.0
        #     self.grasp_counter += 1
        # else:
        #     self.grasp_counter = 0

        # # stability bonus
        # if self.grasp_counter > 50:
        #     reward += 100.0

        # # -------------------------
        # # SMOOTHNESS
        # # -------------------------
        # reward -= 0.05 * np.sum(action**2)

       

        # qpos = self.data.qpos[:self.n_joints]
        # limit_violation = np.sum(
        #     (qpos <= self.joint_mins + 1e-3) |
        #     (qpos >= self.joint_maxs - 1e-3)
        # )
        # reward -= 0.5 * limit_violation

        # # -------------------------
        # # SUCCESS
        # # -------------------------
        # success = self.grasp_counter > 10
        # if success:
        #     reward += 100.0

        # return reward, {
        #     "success": success,
        #     "distance": distance,
        #     "left_contact": left,
        #     "right_contact": right
        # }
    

    def _compute_reward(self):
        ee_pos = self.data.xpos[self.ee_id]
        obj_pos = self.data.xpos[self.obj_body_id]

        distance = np.linalg.norm(ee_pos - obj_pos)

        left, right = self._detect_contacts()

        reward = 0.0

        # -------------------------
        # 1. REACHING (dense, stable)
        # -------------------------
        reward += -3.0 * distance

        if self.prev_distance is None:
            self.prev_distance = distance

        progress = self.prev_distance - distance
        reward += 5.0 * progress
        self.prev_distance = distance

        reward += 2.0 * np.exp(-10 * distance)

        # -------------------------
        # 2. CONTACT (smooth, not binary)
        # -------------------------
        contact_count = float(left) + float(right)   # 0, 1, 2
        reward += 5.0 * contact_count

        # -------------------------
        # 3. GRIP ENCOURAGEMENT (VERY IMPORTANT)
        # -------------------------
        # encourage closing gripper when near object
        if distance < 0.05:
            grip_action = self.data.ctrl[self.gripper_idx]
            reward += 3.0 * (1.0 - grip_action)  # encourage closing

        # -------------------------
        # 4. STABLE GRASP (gradual, not jump)
        # -------------------------
        if left and right:
            self.grasp_counter += 1
            reward += 2.0 * self.grasp_counter   # increases over time
        else:
            self.grasp_counter = 0

        # -------------------------
        # 5. LIFT OBJECT (next stage) -> lets learn to grasp the object first
        # -------------------------
        # obj_height = obj_pos[2]

        # if left and right:
        #     reward += 20.0 * max(0.0, obj_height - 0.02)

        # -------------------------
        # 6. SUCCESS
        # -------------------------
        success = self.grasp_counter > 2000

        if success:
            reward += 100.0

        # -------------------------
        # 7. ACTION PENALTY (stability)
        # -------------------------
        reward -= 0.02 * np.sum(self.data.ctrl[:self.n_joints] ** 2)

        # -------------------------
        # 8. VELOCITY SMOOTHNESS
        # -------------------------
        reward -= 0.001 * np.sum(self.data.qvel[:self.n_joints] ** 2)

        # -------------------------
        # 9. JOINT LIMIT PENALTY
        # -------------------------
        qpos = self.data.qpos[:self.n_joints]
        limit_violation = np.sum(
            (qpos <= self.joint_mins + 1e-3) |
            (qpos >= self.joint_maxs - 1e-3)
        )
        reward -= 0.5 * limit_violation

        return reward, {
            "success": success,
            "distance": distance,
            "left_contact": left,
            "right_contact": right,
            # "height": obj_height,
        }