"""Export a 29-DOF standby controller checkpoint as ONNX for deployment.

PREFERRED: Use play.py which auto-exports with correct joint ordering
from the live IsaacLab env (no hardcoded BFS order needed):

    python scripts/rsl_rl/play.py --task Custom-Standby-G1-Play-v0 \\
        --load_run <run_name> --checkpoint <model.pt> --num_envs 1

This standalone script is a FALLBACK for exporting without Isaac Sim.
It uses hardcoded BFS joint ordering — if the URDF kinematic tree
changes, BODY_JOINT_NAMES must be updated manually.

Observations: base_ang_vel(3) + projected_gravity(3)
              + body_pose_commands(29)
              + joint_pos(29) + joint_vel(29) + actions(29)
              = 122

Usage:
    conda activate mimic
    cd ~/Desktop/xhand_manip_ws/locomotion/g1_xhand_standby
    python scripts/export_deploy.py --run <run_name> [--checkpoint model_XXXX.pt]

With baked default pose (93-dim input, no body_pose_commands):
    python scripts/export_deploy.py --run <run_name> --bake_defaults
"""

import argparse
import os
import shutil

import numpy as np
import onnx
import torch
import torch.nn as nn


# -- 29 G1 body joints in PhysX BFS articulation order -----------------------
BODY_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]

N_JOINTS = len(BODY_JOINT_NAMES)  # 29
N_BODY_CMD = 29  # full body pose command dimension
EXPECTED_OBS = 3 + 3 + N_BODY_CMD + N_JOINTS + N_JOINTS + N_JOINTS  # 122


# -- Physical constants from g1.py ------------------------------------------

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0

_KP_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ ** 2
_KP_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ ** 2
_KP_5020 = ARMATURE_5020 * NATURAL_FREQ ** 2
_KP_4010 = ARMATURE_4010 * NATURAL_FREQ ** 2

_KD_7520_14 = 2 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
_KD_7520_22 = 2 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
_KD_5020 = 2 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
_KD_4010 = 2 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

_JOINT_PROPS = {
    "hip_pitch": (_KP_7520_14, _KD_7520_14, 88.0),
    "hip_roll":  (_KP_7520_22, _KD_7520_22, 139.0),
    "hip_yaw":   (_KP_7520_14, _KD_7520_14, 88.0),
    "knee":      (_KP_7520_22, _KD_7520_22, 139.0),
    "ankle_pitch": (2 * _KP_5020, 2 * _KD_5020, 50.0),
    "ankle_roll":  (2 * _KP_5020, 2 * _KD_5020, 50.0),
    "waist_yaw":   (_KP_7520_14, _KD_7520_14, 88.0),
    "waist_roll":  (2 * _KP_5020, 2 * _KD_5020, 50.0),
    "waist_pitch": (2 * _KP_5020, 2 * _KD_5020, 50.0),
    "shoulder_pitch": (_KP_5020, _KD_5020, 25.0),
    "shoulder_roll":  (_KP_5020, _KD_5020, 25.0),
    "shoulder_yaw":   (_KP_5020, _KD_5020, 25.0),
    "elbow":          (_KP_5020, _KD_5020, 25.0),
    "wrist_roll":     (_KP_5020, _KD_5020, 25.0),
    "wrist_pitch":    (_KP_4010, _KD_4010, 5.0),
    "wrist_yaw":      (_KP_4010, _KD_4010, 5.0),
}

_DEFAULT_POS = {
    "left_hip_pitch_joint": -0.312, "right_hip_pitch_joint": -0.312,
    "left_knee_joint": 0.669, "right_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363, "right_ankle_pitch_joint": -0.363,
    "left_elbow_joint": 1.28, "right_elbow_joint": 1.28,
    "left_shoulder_roll_joint": 0.2, "right_shoulder_roll_joint": -0.2,
    "left_shoulder_pitch_joint": 0.2, "right_shoulder_pitch_joint": 0.2,
}


def _lookup_joint_prop(name: str) -> tuple[float, float, float]:
    """Return (stiffness, damping, effort) for a joint by name."""
    for key, val in _JOINT_PROPS.items():
        if key in name:
            return val
    raise ValueError(f"Unknown joint: {name}")


def _build_arrays() -> dict[str, list[float]]:
    """Build 29-element arrays for stiffness, damping, default_pos, action_scale."""
    kp, kd, default_pos, action_scale = [], [], [], []
    for name in BODY_JOINT_NAMES:
        s, d, effort = _lookup_joint_prop(name)
        kp.append(s)
        kd.append(d)
        default_pos.append(_DEFAULT_POS.get(name, 0.0))
        action_scale.append(round(0.25 * effort / s, 6))
    return {
        "stiffness": kp,
        "damping": kd,
        "default_pos": default_pos,
        "action_scale": action_scale,
    }


class EmpiricalNormalizer(nn.Module):
    """Running-mean normalizer matching rsl_rl.networks.EmpiricalNormalization."""

    def __init__(self, input_dim: int, eps: float = 1e-2):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(input_dim))
        self.register_buffer("var", torch.ones(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize observation to zero mean, unit variance."""
        return (x - self.mean) / (torch.sqrt(self.var) + self.eps)


class DeployablePolicy(nn.Module):
    """Direct 29-DOF standby policy for deployment (122-dim obs)."""

    def __init__(
        self,
        actor: nn.Sequential,
        normalizer: EmpiricalNormalizer | None = None,
    ):
        super().__init__()
        self.actor = actor
        self.normalizer = normalizer

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Run policy.

        Args:
            obs: Controller observation [batch, 122].

        Returns:
            torch.Tensor: Action [batch, 29].
        """
        if self.normalizer is not None:
            obs = self.normalizer(obs)
        return self.actor(obs)


class DeployablePolicyBakedPose(nn.Module):
    """Policy with baked default body pose commands.

    Takes 93-dim input (standard controller obs WITHOUT body_pose_commands),
    injects default joint positions as the command, and feeds the full
    122-dim vector to the trained actor.
    """

    def __init__(
        self,
        actor: nn.Sequential,
        normalizer: EmpiricalNormalizer | None,
        default_commands: list[float],
    ):
        super().__init__()
        self.actor = actor
        self.normalizer = normalizer
        self.register_buffer(
            "body_cmd",
            torch.tensor(default_commands, dtype=torch.float32).unsqueeze(0),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Run policy with baked body pose commands.

        Args:
            obs: Controller observation [batch, 93] (no body_pose_commands).

        Returns:
            torch.Tensor: Action [batch, 29].
        """
        prefix = obs[:, :6]    # base_ang_vel(3) + projected_gravity(3)
        suffix = obs[:, 6:]    # joint_pos(29) + joint_vel(29) + actions(29)
        cmd = self.body_cmd.expand(obs.shape[0], -1)
        full_obs = torch.cat([prefix, cmd, suffix], dim=1)
        if self.normalizer is not None:
            full_obs = self.normalizer(full_obs)
        return self.actor(full_obs)


def list_to_csv(arr, decimals=6):
    """Serialize a list to CSV string."""
    parts = []
    for x in arr:
        if isinstance(x, float):
            parts.append(f"{x:.{decimals}f}")
        else:
            parts.append(str(x))
    return ",".join(parts)


def main():
    """Export 29-DOF standby checkpoint as ONNX with full metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", required=True, help="Run name (subfolder under logs)"
    )
    parser.add_argument(
        "--checkpoint", default=None, help="Checkpoint file (default: latest)"
    )
    parser.add_argument(
        "--output", default=None, help="Output path (default: policies folder)"
    )
    parser.add_argument(
        "--bake_defaults",
        action="store_true",
        help="Bake default body pose commands into ONNX (93-dim input).",
    )
    args = parser.parse_args()

    log_dir = os.path.join(
        "logs", "rsl_rl", "custom_g1_standby", args.run
    )
    if not os.path.isdir(log_dir):
        raise FileNotFoundError(f"Run directory not found: {log_dir}")

    if args.checkpoint:
        ckpt_path = os.path.join(log_dir, args.checkpoint)
    else:
        pts = sorted(
            [
                f
                for f in os.listdir(log_dir)
                if f.startswith("model_") and f.endswith(".pt")
            ],
            key=lambda f: int(f.split("_")[1].split(".")[0]),
        )
        ckpt_path = os.path.join(log_dir, pts[-1])

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model_state_dict"]

    in_dim = sd["actor.0.weight"].shape[1]
    out_dim = sd["actor.6.weight"].shape[0]
    print(f"Actor: {in_dim} -> [512, 256, 128] -> {out_dim}")

    assert in_dim == EXPECTED_OBS, (
        f"Expected actor input dim {EXPECTED_OBS}, got {in_dim}."
    )
    assert out_dim == N_JOINTS, (
        f"Expected {N_JOINTS} action outputs, got {out_dim}."
    )

    actor = nn.Sequential(
        nn.Linear(in_dim, 512),
        nn.ELU(),
        nn.Linear(512, 256),
        nn.ELU(),
        nn.Linear(256, 128),
        nn.ELU(),
        nn.Linear(128, out_dim),
    )
    actor_sd = {
        k.replace("actor.", ""): v
        for k, v in sd.items()
        if k.startswith("actor.")
    }
    actor.load_state_dict(actor_sd)
    actor.eval()

    normalizer = None
    norm_key = "actor_obs_normalizer._mean"
    if norm_key in sd:
        norm_mean = sd["actor_obs_normalizer._mean"]
        norm_var = sd["actor_obs_normalizer._var"]
        normalizer = EmpiricalNormalizer(in_dim)
        normalizer.mean.copy_(norm_mean.squeeze())
        normalizer.var.copy_(norm_var.squeeze())
        print(
            f"Loaded empirical normalizer (mean range: "
            f"[{norm_mean.min():.3f}, {norm_mean.max():.3f}])"
        )

    arrays = _build_arrays()

    if args.bake_defaults:
        deploy_model = DeployablePolicyBakedPose(
            actor,
            normalizer=normalizer,
            default_commands=arrays["default_pos"],
        )
        onnx_in_dim = EXPECTED_OBS - N_BODY_CMD  # 93
        obs_names = [
            "base_ang_vel",
            "projected_gravity",
            "joint_pos",
            "joint_vel",
            "actions",
        ]
        obs_history = [1, 1, 1, 1, 1]
        cmd_names = []
        print(
            f"Baking default pose -> ONNX input: {onnx_in_dim} "
            f"(controller-compatible)"
        )
    else:
        deploy_model = DeployablePolicy(actor, normalizer=normalizer)
        onnx_in_dim = EXPECTED_OBS  # 122
        obs_names = [
            "base_ang_vel",
            "projected_gravity",
            "body_pose_commands",
            "joint_pos",
            "joint_vel",
            "actions",
        ]
        obs_history = [1, 1, 1, 1, 1, 1]
        cmd_names = ["body_pose"]

    deploy_model.eval()

    dummy_obs = torch.zeros(1, onnx_in_dim)
    onnx_dir = os.path.join(log_dir, "exported")
    os.makedirs(onnx_dir, exist_ok=True)
    onnx_path = os.path.join(onnx_dir, "policy.onnx")

    torch.onnx.export(
        deploy_model,
        dummy_obs,
        onnx_path,
        export_params=True,
        opset_version=11,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={},
    )
    print(f"Exported ONNX: {onnx_path}")

    model = onnx.load(onnx_path)

    metadata = {
        "run_path": ckpt_path,
        "joint_names": list_to_csv(BODY_JOINT_NAMES),
        "joint_stiffness": list_to_csv(arrays["stiffness"]),
        "joint_damping": list_to_csv(arrays["damping"]),
        "default_joint_pos": list_to_csv(arrays["default_pos"]),
        "action_scale": list_to_csv(arrays["action_scale"]),
        "command_names": list_to_csv(cmd_names) if cmd_names else "",
        "observation_names": list_to_csv(obs_names),
        "observation_history_lengths": list_to_csv(obs_history),
    }

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = v
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)
    print(f"Attached metadata to {onnx_path}")

    if args.output:
        dest = args.output
    else:
        ws_root = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "..",
                "..",
            )
        )
        dest = os.path.join(
            ws_root,
            "src",
            "g1_xhand_description",
            "config",
            "policies",
            "standby_policy.onnx",
        )

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copy2(onnx_path, dest)
    print(f"Copied to: {dest}")


if __name__ == "__main__":
    main()
