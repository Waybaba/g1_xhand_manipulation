"""IK-based pose command term for the standby controller.

Generates random end-effector (wrist) poses in the robot's workspace,
resolves them to arm joint positions via IsaacLab's
``DifferentialIKController``, and presents the full 29-DOF target
(legs + waist at default, arms from IK) as the command.

Target wrist positions are visualized as colored sphere markers.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING, field
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# ---------------------------------------------------------------------------
# Visualization marker configs
# ---------------------------------------------------------------------------
_LEFT_IK_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/LeftIKTarget",
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.05,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),
            ),
        ),
    },
)

_RIGHT_IK_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/RightIKTarget",
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.05,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 1.0),
            ),
        ),
    },
)


class IKPoseCommand(CommandTerm):
    """Command that generates random wrist EE poses and resolves via IK.

    For each wrist (left and right), random positions are sampled within
    a configurable workspace box relative to the robot's root frame.
    DifferentialIKController (damped least squares) is used to solve for
    the arm joint positions that reach those poses.

    The command output is always 29-DOF joint positions, with legs and
    waist set to their standing defaults.
    """

    cfg: "IKPoseCommandCfg"

    def __init__(self, cfg: "IKPoseCommandCfg", env: ManagerBasedEnv):
        """Initialize the IK pose command.

        Args:
            cfg: Command term configuration.
            env: The environment instance.
        """
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]

        # Resolve all 29 body joint indices
        self._all_ids, self._all_names = self.robot.find_joints(
            cfg.joint_names
        )
        self._num_cmd = len(self._all_ids)

        # Resolve arm joint indices (subset of all)
        self._left_arm_ids, _ = self.robot.find_joints(
            cfg.left_arm_joint_names
        )
        self._right_arm_ids, _ = self.robot.find_joints(
            cfg.right_arm_joint_names
        )

        # Map arm joints to local command indices
        all_list = list(self._all_ids)
        self._left_arm_local = [all_list.index(i) for i in self._left_arm_ids]
        self._right_arm_local = [
            all_list.index(i) for i in self._right_arm_ids
        ]

        # Resolve wrist body indices
        self._left_body_idx = self.robot.find_bodies(
            cfg.left_wrist_body_name
        )[0][0]
        self._right_body_idx = self.robot.find_bodies(
            cfg.right_wrist_body_name
        )[0][0]

        # Create IK controllers (one per wrist)
        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.1},
        )
        self._left_ik = DifferentialIKController(
            ik_cfg, self.num_envs, self.device
        )
        self._right_ik = DifferentialIKController(
            ik_cfg, self.num_envs, self.device
        )

        # Jacobian indices for floating-base
        # Reason: for floating-base robots the Jacobian includes 6 extra
        # columns for the unactuated base DOFs
        if self.robot.is_fixed_base:
            self._left_jac_body = self._left_body_idx - 1
            self._right_jac_body = self._right_body_idx - 1
            self._left_jac_joints = list(self._left_arm_ids)
            self._right_jac_joints = list(self._right_arm_ids)
        else:
            self._left_jac_body = self._left_body_idx
            self._right_jac_body = self._right_body_idx
            self._left_jac_joints = [i + 6 for i in self._left_arm_ids]
            self._right_jac_joints = [i + 6 for i in self._right_arm_ids]

        self._default_pos = (
            self.robot.data.default_joint_pos[0, self._all_ids].clone()
        )

        default_expanded = self._default_pos.unsqueeze(0).expand(
            self.num_envs, -1
        )
        self._body_command = default_expanded.clone()

        # Store desired EE poses for visualization
        self._left_ee_target = torch.zeros(
            self.num_envs, 7, device=self.device
        )
        self._left_ee_target[:, 3] = 1.0  # identity quaternion
        self._right_ee_target = torch.zeros(
            self.num_envs, 7, device=self.device
        )
        self._right_ee_target[:, 3] = 1.0

        self.metrics["error_body_pos"] = torch.zeros(
            self.num_envs, device=self.device
        )

    def __str__(self) -> str:
        """Return string representation."""
        return (
            f"IKPoseCommand:\n"
            f"\tJoints ({self._num_cmd}): {self._all_names}\n"
            f"\tLeft arm joints: {len(self._left_arm_ids)}\n"
            f"\tRight arm joints: {len(self._right_arm_ids)}\n"
            f"\tWorkspace pos range: {self.cfg.workspace_pos_range}"
        )

    @property
    def command(self) -> torch.Tensor:
        """The body target tensor. Shape: ``(num_envs, num_cmd)``."""
        return self._body_command

    def _update_metrics(self):
        """Track L1 error between actual and commanded positions."""
        error = torch.sum(
            torch.abs(
                self.robot.data.joint_pos[:, self._all_ids]
                - self._body_command
            ),
            dim=1,
        )
        self.metrics["error_body_pos"] = error

    def _resample_command(self, env_ids: Sequence[int]):
        """Sample random wrist poses and solve IK for joint positions."""
        n = len(env_ids)

        # Start from defaults for all joints
        targets = self._default_pos.unsqueeze(0).expand(n, -1).clone()

        # Sample random EE positions in robot root frame
        pos_range = self.cfg.workspace_pos_range
        for side, arm_local, body_idx, ik_ctrl, jac_body, jac_joints, ee_buf in [
            (
                "left",
                self._left_arm_local,
                self._left_body_idx,
                self._left_ik,
                self._left_jac_body,
                self._left_jac_joints,
                self._left_ee_target,
            ),
            (
                "right",
                self._right_arm_local,
                self._right_body_idx,
                self._right_ik,
                self._right_jac_body,
                self._right_jac_joints,
                self._right_ee_target,
            ),
        ]:
            # Sample target position relative to root
            target_pos_b = torch.zeros(n, 3, device=self.device)
            target_pos_b[:, 0].uniform_(*pos_range["x"])
            target_pos_b[:, 1].uniform_(*pos_range["y"])
            if side == "right":
                target_pos_b[:, 1] *= -1  # mirror y for right arm
            target_pos_b[:, 2].uniform_(*pos_range["z"])

            # Convert to world frame
            root_pos = self.robot.data.root_pos_w[env_ids]
            root_quat = self.robot.data.root_quat_w[env_ids]
            target_pos_w = root_pos + math_utils.quat_apply(
                root_quat, target_pos_b
            )

            # Current EE state
            ee_pos = self.robot.data.body_pos_w[env_ids, body_idx]
            ee_quat = self.robot.data.body_quat_w[env_ids, body_idx]

            # Target orientation = current orientation (keep wrist orientation)
            target_quat_w = ee_quat.clone()

            # Set IK command (absolute pose)
            # Reason: convert world-frame targets to root-frame for IK
            target_pos_root, target_quat_root = (
                math_utils.subtract_frame_transforms(
                    root_pos, root_quat, target_pos_w, target_quat_w
                )
            )
            ik_command = torch.cat(
                [target_pos_root, target_quat_root], dim=-1
            )
            ik_ctrl.reset(env_ids)
            ik_ctrl.set_command(ik_command, ee_pos_root, ee_quat_root)

            # Compute IK: iterate a few steps for convergence
            joint_pos_arm = self.robot.data.joint_pos[
                env_ids
            ][:, list(self._left_arm_ids if side == "left" else self._right_arm_ids)]

            ee_pos_root = target_pos_b.clone()
            ee_quat_root = math_utils.quat_mul(
                math_utils.quat_inv(root_quat), ee_quat
            )

            # Reason: set_command needs root-frame EE, recompute
            ik_ctrl.set_command(ik_command, ee_pos_root, ee_quat_root)

            # Get the Jacobian for this wrist
            jacobians = self.robot.root_physx_view.get_jacobians()
            jac = jacobians[env_ids][:, jac_body, :, jac_joints]

            # Solve IK (one step — DLS is robust enough)
            joint_pos_des = ik_ctrl.compute(
                ee_pos_root, ee_quat_root, jac, joint_pos_arm
            )
            targets[:, arm_local] = joint_pos_des

            # Store target for visualization
            ee_buf[env_ids, :3] = target_pos_w
            ee_buf[env_ids, 3:] = target_quat_w

        self._body_command[env_ids] = targets

    def _update_command(self):
        """No interpolation for IK commands — applied directly."""
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Create or toggle wrist target markers."""
        if debug_vis:
            if not hasattr(self, "_left_marker"):
                self._left_marker = VisualizationMarkers(_LEFT_IK_MARKER_CFG)
                self._right_marker = VisualizationMarkers(
                    _RIGHT_IK_MARKER_CFG
                )
            self._left_marker.set_visibility(True)
            self._right_marker.set_visibility(True)
        else:
            if hasattr(self, "_left_marker"):
                self._left_marker.set_visibility(False)
                self._right_marker.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Update marker positions to show IK target locations."""
        if not self.robot.is_initialized:
            return
        self._left_marker.visualize(
            self._left_ee_target[:, :3], self._left_ee_target[:, 3:]
        )
        self._right_marker.visualize(
            self._right_ee_target[:, :3], self._right_ee_target[:, 3:]
        )


@configclass
class IKPoseCommandCfg(CommandTermCfg):
    """Configuration for the IK pose command term.

    Attributes:
        asset_name: Name of the robot articulation in the scene.
        joint_names: Regex patterns for ALL 29 body joints.
        left_arm_joint_names: Joint name patterns for the left arm.
        right_arm_joint_names: Joint name patterns for the right arm.
        left_wrist_body_name: Body name for left wrist EE.
        right_wrist_body_name: Body name for right wrist EE.
        workspace_pos_range: Per-axis (x, y, z) sampling ranges in
            meters, relative to the robot root frame.
    """

    class_type: type = IKPoseCommand
    asset_name: str = "robot"

    joint_names: list[str] = MISSING
    left_arm_joint_names: list[str] = field(default_factory=lambda: [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ])
    right_arm_joint_names: list[str] = field(default_factory=lambda: [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ])

    left_wrist_body_name: str = "left_wrist_yaw_link"
    right_wrist_body_name: str = "right_wrist_yaw_link"

    workspace_pos_range: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (-0.1, 0.4),
            "y": (0.1, 0.5),
            "z": (-0.2, 0.4),
        }
    )

    debug_vis: bool = True
