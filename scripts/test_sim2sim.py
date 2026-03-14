"""Standalone sim2sim test: run ONNX policy in MuJoCo with viewer.

Bypasses mujoco_sim_ros2 entirely. Sets correct initial pose, applies
PD torques at every physics step, and runs policy inference at 50 Hz.
This replicates the legged_rl_controllers PD logic directly.

Usage:
    python scripts/test_sim2sim.py [--duration 30]
"""

import argparse
import math
import os
import time

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort


# -- Constants ---------------------------------------------------------------

POLICY_HZ = 50.0
MJCF_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "src", "xhand_manip_controller", "mjcf", "g1_xhand.xml",
)
POLICY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "src", "xhand_manip_controller", "config", "policies", "policy.onnx",
)


def parse_csv_float(s: str) -> list[float]:
    """Parse comma-separated float string from ONNX metadata."""
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_csv_str(s: str) -> list[str]:
    """Parse comma-separated string list from ONNX metadata."""
    return [x.strip() for x in s.split(",") if x.strip()]


def load_onnx_with_metadata(path: str) -> tuple:
    """Load ONNX model and extract deployment metadata.

    Args:
        path: Path to policy ONNX file.

    Returns:
        Tuple of (session, metadata_dict).
    """
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    model = __import__("onnx").load(path)
    meta = {p.key: p.value for p in model.metadata_props}
    return session, meta


def projected_gravity(quat: np.ndarray) -> np.ndarray:
    """Compute gravity vector in body frame from quaternion [w, x, y, z].

    Args:
        quat: Unit quaternion [w, x, y, z].

    Returns:
        np.ndarray: Gravity direction in body frame (3,).
    """
    w, x, y, z = quat
    gx = 2.0 * (x * z - w * y)
    gy = 2.0 * (y * z + w * x)
    gz = 1.0 - 2.0 * (x * x + y * y)
    return np.array([gx, gy, gz]) * (-1.0)


def main():
    """Run policy in MuJoCo with interactive viewer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Sim duration in seconds (0 = infinite)")
    parser.add_argument("--mjcf", default=MJCF_PATH)
    parser.add_argument("--policy", default=POLICY_PATH)
    args = parser.parse_args()

    mjcf = os.path.abspath(args.mjcf)
    policy_path = os.path.abspath(args.policy)
    print(f"MJCF:   {mjcf}")
    print(f"Policy: {policy_path}")

    session, meta = load_onnx_with_metadata(policy_path)

    joint_names = parse_csv_str(meta["joint_names"])
    default_pos = np.array(parse_csv_float(meta["default_joint_pos"]))
    stiffness = np.array(parse_csv_float(meta["joint_stiffness"]))
    damping = np.array(parse_csv_float(meta["joint_damping"]))
    action_scale = np.array(parse_csv_float(meta["action_scale"]))
    obs_names = parse_csv_str(meta["observation_names"])
    n_joints = len(joint_names)
    print(f"Joints: {n_joints}, obs terms: {obs_names}")

    model = mujoco.MjModel.from_xml_path(mjcf)
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    steps_per_policy = max(1, int(round(1.0 / (POLICY_HZ * dt))))
    print(f"Physics dt={dt:.4f}s, steps_per_policy={steps_per_policy}")

    joint_qpos_ids = []
    joint_qvel_ids = []
    for jn in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            raise ValueError(f"Joint '{jn}' not found in MJCF")
        joint_qpos_ids.append(model.jnt_qposadr[jid])
        joint_qvel_ids.append(model.jnt_dofadr[jid])

    # Set initial qpos to training defaults
    data.qpos[2] = 0.78
    data.qpos[3] = 1.0
    for i, idx in enumerate(joint_qpos_ids):
        data.qpos[idx] = default_pos[i]
    mujoco.mj_forward(model, data)

    print("\nInitial joint positions (first 15):")
    for i in range(min(15, n_joints)):
        print(f"  {joint_names[i]:35s}: {data.qpos[joint_qpos_ids[i]]:+.4f} "
              f"(default={default_pos[i]:+.4f})")

    last_action = np.zeros(n_joints, dtype=np.float32)
    velocity_cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    policy_step = 0
    start_wall = time.time()

    def apply_pd_and_step():
        """Apply PD torques based on current targets and step physics."""
        pos_target = default_pos + last_action * action_scale
        for i in range(n_joints):
            qi = joint_qpos_ids[i]
            vi = joint_qvel_ids[i]
            pos_err = pos_target[i] - data.qpos[qi]
            vel = data.qvel[vi]
            tau = stiffness[i] * pos_err - damping[i] * vel
            data.qfrc_applied[vi] = tau
        mujoco.mj_step(model, data)

    def build_obs() -> np.ndarray:
        """Construct observation vector matching controller conventions."""
        base_quat = data.qpos[3:7].copy()
        base_angvel_world = data.qvel[3:6].copy()
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, base_quat)
        rot = rot.reshape(3, 3)
        base_angvel_local = rot.T @ base_angvel_world
        proj_grav = projected_gravity(base_quat)

        joint_pos = np.array([data.qpos[qi] for qi in joint_qpos_ids])
        joint_vel = np.array([data.qvel[vi] for vi in joint_qvel_ids])
        joint_pos_rel = joint_pos - default_pos

        obs_parts = []
        for name in obs_names:
            if name == "base_ang_vel":
                obs_parts.append(base_angvel_local.astype(np.float32))
            elif name == "projected_gravity":
                obs_parts.append(proj_grav.astype(np.float32))
            elif name == "velocity_commands":
                obs_parts.append(velocity_cmd)
            elif name == "joint_pos":
                obs_parts.append(joint_pos_rel.astype(np.float32))
            elif name == "joint_vel":
                obs_parts.append(joint_vel.astype(np.float32))
            elif name == "actions":
                obs_parts.append(last_action)
        return np.concatenate(obs_parts).reshape(1, -1)

    print(f"\nLaunching viewer (duration={args.duration}s)...")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            sim_time = data.time

            if args.duration > 0 and sim_time > args.duration:
                break

            obs = build_obs()
            outputs = session.run(None, {"obs": obs})
            last_action = outputs[0][0].astype(np.float32)

            for _ in range(steps_per_policy):
                apply_pd_and_step()

            viewer.sync()
            policy_step += 1

            if policy_step % 250 == 0:
                hip = data.qpos[joint_qpos_ids[0]]
                knee = data.qpos[joint_qpos_ids[3]]
                height = data.qpos[2]
                elapsed_wall = time.time() - start_wall
                print(f"t={sim_time:6.1f}s | wall={elapsed_wall:5.1f}s | "
                      f"height={height:.3f} | hip_L={hip:+.3f} | knee_L={knee:+.3f}")

    print(f"\nDone. Sim time: {data.time:.1f}s")


if __name__ == "__main__":
    main()
