from enum import Enum, auto
import numpy as np
from controller import inverse_kinematics_step, set_gripper

class Phase(Enum):
    PRE_GRASP = auto()
    DESCEND   = auto()
    GRASP     = auto()
    LIFT      = auto()
    HOLD      = auto()
    DONE      = auto()



class PickAndLiftController:

    def __init__(self, model, object="cube"):
        self.phase = Phase.PRE_GRASP
        self.grasp_counter = 0
        self.hold_counter = 0

        # Body ids
        self.ee_id = model.body("tool0").id
        self.obj_body_id = model.body("cube").id

        self.obj_geom = model.geom(object).id
        self.left_geom = model.geom("finger_right").id
        self.right_geom = model.geom("finger_left").id

        self.pre_grasp_height = 0.1
        self.lift_height = 0.4
    # Pos error
    def pos_error(self, a, b):
        return np.linalg.norm(a - b)

    # -------------------------
    # Contact detection
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


    # -------------------------
    # MAIN STEP
    # -------------------------
    def step(self, model, data):

        ee_pos = data.xpos[self.ee_id]
        obj_pos = data.xpos[self.obj_body_id]

        error = obj_pos - ee_pos
        xy_error = np.linalg.norm(error[:2])
        z_error = abs(error[2])


        # PRE_GRASP (go above)
        if self.phase == Phase.PRE_GRASP:

            target = obj_pos + np.array([0, 0, 0.1])
            inverse_kinematics_step(model, data, target)

            if self.pos_error(ee_pos, target) < 0.01:
                print("PRE_GRASP → DESCEND")
                self.phase = Phase.DESCEND

        # DESCEND (align + go down)
        elif self.phase == Phase.DESCEND:

            target = obj_pos
            inverse_kinematics_step(model, data, target)

            if xy_error < 0.003 and z_error < 0.003:
                print("DESCEND → GRASP")
                self.phase = Phase.GRASP

        # GRASP
        elif self.phase == Phase.GRASP:

            set_gripper(model, data, "close")

            left, right = self.detect_grasp_contacts(model, data)

            if left and right:
                self.grasp_counter += 1
            else:
                self.grasp_counter = 0

            if self.grasp_counter > 500:
                print("GRASP STABLE → LIFT")
                self.phase = Phase.LIFT

        # LIFT
        elif self.phase == Phase.LIFT:
            set_gripper(model, data, "close")


            target = np.array([0.4, 0, self.lift_height]) # Dont add object position , it can cause drift
            inverse_kinematics_step(model, data, target)

            if obj_pos[2] > (self.lift_height - 0.01) :
                print("OBJECT LIFTED → HOLD")
                self.phase = Phase.HOLD

        # HOLD
        elif self.phase == Phase.HOLD:

            set_gripper(model, data, "close")  

            ee_pos = data.xpos[self.ee_id]
            target = ee_pos.copy()

            inverse_kinematics_step(model, data, target)

            self.hold_counter += 1

            print("Holding...", self.hold_counter)

            if self.hold_counter > 2000:   
                print("HOLD COMPLETE → DONE")
                self.phase = Phase.DONE

        # DONE
        elif self.phase == Phase.DONE:
            pass