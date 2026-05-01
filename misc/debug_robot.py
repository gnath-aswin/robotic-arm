# debug_mujoco_indices.# debug_mujoco_indices.py

# pyright: reportAttributeAccessIssue=false

import mujoco
import numpy as np

from config import CONFIG


MODEL_XML_PATH = "scene_with_objects.xml"


def safe_name(model, obj_type, obj_id):
    name = mujoco.mj_id2name(model, obj_type, obj_id)
    return name if name is not None else f"<unnamed_{obj_id}>"


def joint_type_name(joint_type):
    names = {
        mujoco.mjtJoint.mjJNT_FREE: "FREE",
        mujoco.mjtJoint.mjJNT_BALL: "BALL",
        mujoco.mjtJoint.mjJNT_SLIDE: "SLIDE",
        mujoco.mjtJoint.mjJNT_HINGE: "HINGE",
    }
    return names.get(joint_type, str(joint_type))


def actuator_trn_type_name(trn_type):
    names = {
        mujoco.mjtTrn.mjTRN_JOINT: "JOINT",
        mujoco.mjtTrn.mjTRN_JOINTINPARENT: "JOINTINPARENT",
        mujoco.mjtTrn.mjTRN_SLIDERCRANK: "SLIDERCRANK",
        mujoco.mjtTrn.mjTRN_TENDON: "TENDON",
        mujoco.mjtTrn.mjTRN_SITE: "SITE",
        mujoco.mjtTrn.mjTRN_BODY: "BODY",
    }
    return names.get(trn_type, str(trn_type))


def actuator_target_name(model, aid):
    trn_type = model.actuator_trntype[aid]
    trnid = model.actuator_trnid[aid]

    target_id = int(trnid[0])

    if target_id < 0:
        return None

    if trn_type in (
        mujoco.mjtTrn.mjTRN_JOINT,
        mujoco.mjtTrn.mjTRN_JOINTINPARENT,
    ):
        return safe_name(model, mujoco.mjtObj.mjOBJ_JOINT, target_id)

    if trn_type == mujoco.mjtTrn.mjTRN_TENDON:
        return safe_name(model, mujoco.mjtObj.mjOBJ_TENDON, target_id)

    if trn_type == mujoco.mjtTrn.mjTRN_SITE:
        return safe_name(model, mujoco.mjtObj.mjOBJ_SITE, target_id)

    if trn_type == mujoco.mjtTrn.mjTRN_BODY:
        return safe_name(model, mujoco.mjtObj.mjOBJ_BODY, target_id)

    return f"id={target_id}"


def print_joints(model: mujoco.MjModel):
    print("\n" + "=" * 100)
    print("JOINTS")
    print("=" * 100)

    for jid in range(model.njnt):
        name = safe_name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)

        jtype = model.jnt_type[jid]
        qpos_adr = int(model.jnt_qposadr[jid])
        qvel_adr = int(model.jnt_dofadr[jid])

        limited = bool(model.jnt_limited[jid])
        joint_range = model.jnt_range[jid]

        maybe_gripper = any(
            key in name.lower()
            for key in ["gripper", "finger", "left", "right", "jaw"]
        )

        tag = "  <-- POSSIBLE GRIPPER" if maybe_gripper else ""

        print(
            f"{jid:02d} | "
            f"name={name:30s} | "
            f"type={joint_type_name(jtype):6s} | "
            f"qpos={qpos_adr:02d} | "
            f"qvel={qvel_adr:02d} | "
            f"limited={limited} | "
            f"range={joint_range}"
            f"{tag}"
        )


def print_actuators(model: mujoco.MjModel):
    print("\n" + "=" * 100)
    print("ACTUATORS / ctrl indices")
    print("=" * 100)

    for aid in range(model.nu):
        name = safe_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)

        trn_type = model.actuator_trntype[aid]
        target_name = actuator_target_name(model, aid)

        ctrlrange = model.actuator_ctrlrange[aid]
        ctrllimited = bool(model.actuator_ctrllimited[aid])
        gear = model.actuator_gear[aid]

        check_text = f"{name} {target_name}".lower()
        maybe_gripper = any(
            key in check_text
            for key in ["gripper", "finger", "left", "right", "jaw"]
        )

        tag = "  <-- POSSIBLE GRIPPER CTRL" if maybe_gripper else ""

        print(
            f"ctrl[{aid:02d}] | "
            f"actuator={name:30s} | "
            f"trntype={actuator_trn_type_name(trn_type):14s} | "
            f"target={str(target_name):30s} | "
            f"limited={ctrllimited} | "
            f"ctrlrange={ctrlrange} | "
            f"gear={gear}"
            f"{tag}"
        )


def print_config_joint_mapping(model: mujoco.MjModel):
    print("\n" + "=" * 100)
    print("CONFIG ARM JOINT MAPPING")
    print("=" * 100)

    qpos_indices = []
    qvel_indices = []

    for name in CONFIG["joint_names"]:
        jid = model.joint(name).id
        qpos_adr = int(model.jnt_qposadr[jid])
        qvel_adr = int(model.jnt_dofadr[jid])

        qpos_indices.append(qpos_adr)
        qvel_indices.append(qvel_adr)

        print(
            f"CONFIG joint {name:30s} -> "
            f"joint_id={jid:02d}, qpos={qpos_adr:02d}, qvel={qvel_adr:02d}"
        )

    print("\nArm qpos indices used by env:", np.array(qpos_indices, dtype=int))
    print("Arm qvel indices used by env:", np.array(qvel_indices, dtype=int))


def print_gripper_guess_summary(model: mujoco.MjModel):
    print("\n" + "=" * 100)
    print("GRIPPER GUESS SUMMARY")
    print("=" * 100)

    keywords = ["gripper", "finger", "left", "right", "jaw"]

    gripper_qpos = []
    gripper_qvel = []
    gripper_ctrl = []

    for jid in range(model.njnt):
        name = safe_name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if any(k in name.lower() for k in keywords):
            gripper_qpos.append(int(model.jnt_qposadr[jid]))
            gripper_qvel.append(int(model.jnt_dofadr[jid]))

    for aid in range(model.nu):
        name = safe_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        target = actuator_target_name(model, aid)
        text = f"{name} {target}".lower()
        if any(k in text for k in keywords):
            gripper_ctrl.append(aid)

    print("Guessed gripper qpos indices:", np.array(gripper_qpos, dtype=int))
    print("Guessed gripper qvel indices:", np.array(gripper_qvel, dtype=int))
    print("Guessed gripper ctrl indices:", np.array(gripper_ctrl, dtype=int))

    print("\nIf these arrays are empty, your XML uses different names.")
    print("Then paste the JOINTS and ACTUATORS output and we can identify them manually.")


def print_bodies(model: mujoco.MjModel):
    print("\n" + "=" * 100)
    print("BODIES")
    print("=" * 100)

    for bid in range(model.nbody):
        name = safe_name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        print(f"{bid:02d} | {name}")


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)

    print("\nMODEL SUMMARY")
    print("=" * 100)
    print(f"nq    = {model.nq}")
    print(f"nv    = {model.nv}")
    print(f"nu    = {model.nu}")
    print(f"njnt  = {model.njnt}")
    print(f"nbody = {model.nbody}")

    print_joints(model)
    print_actuators(model)
    print_config_joint_mapping(model)
    print_gripper_guess_summary(model)
    print_bodies(model)


if __name__ == "__main__":
    main()
