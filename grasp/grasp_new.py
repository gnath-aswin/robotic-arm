import numpy as np
from enum import Enum, auto
from controller import step_cartesian_action, set_gripper, inverse_kinematics_step

class Phase(Enum):
    PRE_GRASP = auto()
    DESCEND   = auto()
    GRASP     = auto()
    LIFT      = auto()
    DONE      = auto()


class PickAndLiftController:
    def __init__(self, model, object="cube", use_ppo=False, ppo=None):

        self.phase = Phase.PRE_GRASP
        self.grasp_counter = 0

        self.use_ppo = use_ppo
        self.ppo = ppo

        # Body ids
        self.ee_id = model.body("tool0").id
        self.obj_body_id = model.body("cube").id

        self.obj_geom = model.geom(object).id
        self.left_geom = model.geom("finger_right").id
        self.right_geom = model.geom("finger_left").id


        # tuning
        self.target = None
        self.pre_grasp_height = 0.1
        self.lift_height = 0.2


    # Motion primitive
    def move_to_target(self, model, data, target):

        ee_pos = data.xpos[self.ee_id]
        dist = np.linalg.norm(target - ee_pos)

        # Hybrid control
        if self.use_ppo and self.ppo is not None and dist > 0.08:

            rel_goal = target - ee_pos
            obs = np.concatenate([ee_pos, rel_goal, [dist]]).astype(np.float32)

            action, _ = self.ppo.predict(obs, deterministic=True)
            target_pos, error = step_cartesian_action(model, data, 0.2*action)


        else:
            inverse_kinematics_step(model, data, target)
            print("inverse kinematics")
    # -------------------------
    # CONTACT DETECTION
    # -------------------------
    def detect_grasp_contacts(self, model, data):

        left_contact = False
        right_contact = False

        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = c.geom1, c.geom2

            if (g1 == self.left_geom and g2 == self.obj_geom) or \
               (g2 == self.left_geom and g1 == self.obj_geom):
                left_contact = True

            if (g1 == self.right_geom and g2 == self.obj_geom) or \
               (g2 == self.right_geom and g1 == self.obj_geom):
                right_contact = True

        return left_contact, right_contact


    def step(self, model, data):

        ee_pos = data.xpos[self.ee_id]
        obj_pos = data.xpos[self.obj_body_id]

        error = obj_pos - ee_pos
        xy_error = np.linalg.norm(error[:2])
        z_error = abs(error[2])

        # PRE_GRASP
        if self.phase == Phase.PRE_GRASP:
        
            if self.target is None:
                self.target = obj_pos.copy() + np.array([0, 0, self.pre_grasp_height])
            self.move_to_target(model, data, self.target)

            if np.linalg.norm(self.target - ee_pos) < 0.01:
                print("PRE_GRASP -> DESCEND")
                self.phase = Phase.DESCEND

        # DESCEND
        elif self.phase == Phase.DESCEND:

            self.target = obj_pos.copy()
            self.move_to_target(model, data, self.target)

            if xy_error < 0.005 and z_error < 0.01:
                print("DESCEND -> GRASP")
                self.phase = Phase.GRASP

        # GRASP
        elif self.phase == Phase.GRASP:

            set_gripper(model, data, "close")

            left, right = self.detect_grasp_contacts(model, data)

            if left and right:
                self.grasp_counter += 1
            else:
                self.grasp_counter = 0

            if self.grasp_counter > 5:
                print("GRASP STABLE -> LIFT")
                self.phase = Phase.LIFT

        # LIFT
        elif self.phase == Phase.LIFT:

            self.target = obj_pos.copy() + np.array([0, 0, self.lift_height])
            self.move_to_target(model, data, self.target)

            if obj_pos[2] > 0.15:
                print("OBJECT LIFTED")
                self.phase = Phase.DONE

        # DONE
        elif self.phase == Phase.DONE:
            print("DONE")