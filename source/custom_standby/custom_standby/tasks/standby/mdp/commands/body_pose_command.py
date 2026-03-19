"""Full-body pose command term for the standby controller.

Generates random target joint positions for all 29 G1 body joints,
with per-group offset ranges and OpenHomie-style exponential curriculum
for the arm joints.  Legs are constrained to stay near the standing
default; arms ramp up progressively.

Visualization: two pairs of sphere markers per environment:
- Red / Blue (solid): *current* left / right wrist FK positions
  (updates every render step — cheap, just reads body_pos_w).
- Pink / Cyan (translucent): *target* left / right wrist FK positions
  (updates only when a command is resampled — avoids expensive
  write-joint-state-to-sim calls every step).
"""

from __future__ import annotations

import math

import torch
from collections.abc import Sequence
from dataclasses import MISSING, field
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _sample_exponential_ratio(
    ra: float,
    shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    """Sample ratios using OpenHomie's truncated exponential (Eq. 3).

    At ``ra ~ 0`` samples concentrate near 0; at ``ra ~ 1`` the
    distribution approaches ``U(0, 1)``.

    Args:
        ra: Curriculum ratio in ``[0, 1]``.
        shape: Output tensor shape.
        device: Torch device.

    Returns:
        torch.Tensor: Ratios in ``[0, 1]``.
    """
    ra_safe = min(ra, 0.99)
    lam = 20.0 * (1.0 - ra_safe)
    u = torch.rand(shape, device=device)
    ratio = (-1.0 / lam) * torch.log(
        1.0 - u * (1.0 - math.exp(-lam))
    )
    return ratio


# ---------------------------------------------------------------------------
# Visualization marker configs
# ---------------------------------------------------------------------------
# Current wrist positions (solid, small)
_LEFT_CURR_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/LeftWristCurrent",
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.03,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.2, 0.2),
            ),
        ),
    },
)

_RIGHT_CURR_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/RightWristCurrent",
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.03,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.4, 1.0),
            ),
        ),
    },
)

# Target wrist positions (larger, translucent)
_LEFT_TARGET_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/LeftWristTarget",
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.05,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.6, 0.6),
                opacity=0.4,
            ),
        ),
    },
)

_RIGHT_TARGET_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/RightWristTarget",
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.05,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.6, 0.7, 1.0),
                opacity=0.4,
            ),
        ),
    },
)


class BodyPoseCommand(CommandTerm):
    """Command generating random body joint position targets.

    Targets are split into four groups with independent offset ranges:

    * **Legs** -- tight range near standing default for balance stability.
    * **Waist roll/pitch** -- moderate range for torso posture variation.
    * **Waist yaw** -- curriculum-controlled like arms (larger range).
    * **Arms** -- OpenHomie exponential curriculum that ramps from zero
      to the full ``arm_offset_range``.

    New targets are linearly interpolated over ``interpolation_duration``
    seconds.  Wrist FK positions are visualized as colored sphere markers.
    """

    cfg: "BodyPoseCommandCfg"

    def __init__(self, cfg: "BodyPoseCommandCfg", env: ManagerBasedEnv):
        """Initialize the body pose command.

        Args:
            cfg: Command term configuration.
            env: The environment instance.
        """
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]

        # Resolve joint indices per group
        self._all_ids, self._all_names = self.robot.find_joints(cfg.joint_names)
        self._num_cmd = len(self._all_ids)

        self._leg_ids, _ = self.robot.find_joints(cfg.leg_joint_names)
        self._waist_ids, _ = self.robot.find_joints(cfg.waist_joint_names)
        self._waist_yaw_ids, _ = self.robot.find_joints(cfg.waist_yaw_joint_names)
        self._arm_ids, _ = self.robot.find_joints(cfg.arm_joint_names)

        # Remap to local indices within self._all_ids
        all_set = list(self._all_ids)
        self._leg_local = [all_set.index(i) for i in self._leg_ids]
        self._waist_local = [all_set.index(i) for i in self._waist_ids]
        self._waist_yaw_local = [all_set.index(i) for i in self._waist_yaw_ids]
        self._arm_local = [all_set.index(i) for i in self._arm_ids]

        self._default_pos = (
            self.robot.data.default_joint_pos[0, self._all_ids].clone()
        )

        default_expanded = self._default_pos.unsqueeze(0).expand(
            self.num_envs, -1
        )
        self._body_command = default_expanded.clone()
        self._target_command = default_expanded.clone()

        # Interpolation state
        self._dt = self._env.step_dt
        self._interp_steps = max(
            1, int(cfg.interpolation_duration / self._dt)
        )
        self._delta = torch.zeros_like(self._body_command)
        self._interp_remaining = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        # Curriculum ratio for arms (advanced by external curriculum term)
        self._curriculum_ratio: float = cfg.initial_ratio

        self.metrics["error_body_pos"] = torch.zeros(
            self.num_envs, device=self.device
        )

        # Resolve wrist body indices for FK visualization
        self._left_wrist_idx = self.robot.find_bodies(
            cfg.left_wrist_body_name
        )[0][0]
        self._right_wrist_idx = self.robot.find_bodies(
            cfg.right_wrist_body_name
        )[0][0]

        # Reason: cache target wrist positions so we only compute FK
        # at resample time, not every render step.
        self._target_left_pos_w = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._target_right_pos_w = torch.zeros(
            self.num_envs, 3, device=self.device
        )

    def __str__(self) -> str:
        """Return string representation."""
        msg = "BodyPoseCommand:\n"
        msg += f"\tJoints ({self._num_cmd}): {self._all_names}\n"
        msg += f"\tLeg offset: {self.cfg.leg_offset_range}\n"
        msg += f"\tWaist offset: {self.cfg.waist_offset_range}\n"
        msg += f"\tWaist yaw offset: {self.cfg.waist_yaw_offset_range}\n"
        msg += f"\tArm offset: {self.cfg.arm_offset_range}\n"
        msg += f"\tInterpolation: {self.cfg.interpolation_duration}s "
        msg += f"({self._interp_steps} steps)\n"
        msg += f"\tArm curriculum ratio: {self._curriculum_ratio:.2f}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The body target tensor. Shape: ``(num_envs, num_cmd)``."""
        return self._body_command

    def _update_metrics(self):
        """Track L1 error between actual and commanded body positions."""
        error = torch.sum(
            torch.abs(
                self.robot.data.joint_pos[:, self._all_ids]
                - self._body_command
            ),
            dim=1,
        )
        self.metrics["error_body_pos"] = error

    def _resample_command(self, env_ids: Sequence[int]):
        """Sample new body targets with per-group offset ranges."""
        n = len(env_ids)

        # Start from defaults
        targets = self._default_pos.unsqueeze(0).expand(n, -1).clone()

        # --- Legs: small uniform offsets ---
        lo, hi = self.cfg.leg_offset_range
        leg_offsets = (hi - lo) * torch.rand(
            n, len(self._leg_local), device=self.device
        ) + lo
        targets[:, self._leg_local] += leg_offsets

        # --- Waist roll/pitch: moderate uniform offsets ---
        lo, hi = self.cfg.waist_offset_range
        waist_offsets = (hi - lo) * torch.rand(
            n, len(self._waist_local), device=self.device
        ) + lo
        targets[:, self._waist_local] += waist_offsets

        # --- Waist yaw: curriculum-controlled exponential (like WBC) ---
        _, wyaw_hi = self.cfg.waist_yaw_offset_range
        wyaw_ratio = _sample_exponential_ratio(
            self._curriculum_ratio,
            (n, len(self._waist_yaw_local)),
            self.device,
        )
        wyaw_sign = torch.sign(
            torch.rand(n, len(self._waist_yaw_local), device=self.device) - 0.5
        )
        targets[:, self._waist_yaw_local] += wyaw_sign * wyaw_ratio * wyaw_hi

        # --- Arms: OpenHomie exponential curriculum ---
        _, arm_hi = self.cfg.arm_offset_range
        ratio_sample = _sample_exponential_ratio(
            self._curriculum_ratio,
            (n, len(self._arm_local)),
            self.device,
        )
        sign = torch.sign(
            torch.rand(n, len(self._arm_local), device=self.device) - 0.5
        )
        arm_offsets = sign * ratio_sample * arm_hi
        targets[:, self._arm_local] += arm_offsets

        self._target_command[env_ids] = targets

        steps = self._interp_steps
        self._delta[env_ids] = (
            self._target_command[env_ids] - self._body_command[env_ids]
        ) / steps
        self._interp_remaining[env_ids] = steps

        # Compute target wrist FK once at resample time (not every step).
        # Temporarily write target joints, read body_pos_w, restore.
        if self.cfg.debug_vis and hasattr(self, "_left_target_marker"):
            original_pos = self.robot.data.joint_pos[env_ids].clone()
            self.robot.data.joint_pos[env_ids][:, self._all_ids] = targets
            self.robot.write_joint_state_to_sim(
                self.robot.data.joint_pos, self.robot.data.joint_vel
            )
            self.robot.update(dt=0.0)

            self._target_left_pos_w[env_ids] = (
                self.robot.data.body_pos_w[env_ids, self._left_wrist_idx, :3]
                .clone()
            )
            self._target_right_pos_w[env_ids] = (
                self.robot.data.body_pos_w[env_ids, self._right_wrist_idx, :3]
                .clone()
            )

            self.robot.data.joint_pos[env_ids] = original_pos
            self.robot.write_joint_state_to_sim(
                self.robot.data.joint_pos, self.robot.data.joint_vel
            )
            self.robot.update(dt=0.0)

    def _update_command(self):
        """Linearly interpolate command toward the sampled target."""
        active = self._interp_remaining > 0
        if active.any():
            self._body_command[active] += self._delta[active]
            self._interp_remaining[active] -= 1
            finished = self._interp_remaining == 0
            if finished.any():
                self._body_command[finished] = self._target_command[finished]

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Create or toggle wrist markers (current + target)."""
        if debug_vis:
            if not hasattr(self, "_left_curr_marker"):
                self._left_curr_marker = VisualizationMarkers(
                    _LEFT_CURR_MARKER_CFG
                )
                self._right_curr_marker = VisualizationMarkers(
                    _RIGHT_CURR_MARKER_CFG
                )
                self._left_target_marker = VisualizationMarkers(
                    _LEFT_TARGET_MARKER_CFG
                )
                self._right_target_marker = VisualizationMarkers(
                    _RIGHT_TARGET_MARKER_CFG
                )
            self._left_curr_marker.set_visibility(True)
            self._right_curr_marker.set_visibility(True)
            self._left_target_marker.set_visibility(True)
            self._right_target_marker.set_visibility(True)
        else:
            if hasattr(self, "_left_curr_marker"):
                self._left_curr_marker.set_visibility(False)
                self._right_curr_marker.set_visibility(False)
                self._left_target_marker.set_visibility(False)
                self._right_target_marker.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Update wrist markers — current (cheap) + cached target.

        Current wrist positions: read directly from body_pos_w (no cost).
        Target wrist positions: cached at resample time (no per-step FK).
        """
        if not self.robot.is_initialized:
            return

        default_quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device
        ).expand(self.num_envs, -1)

        # Current wrist positions (solid red/blue — fast read)
        left_curr = self.robot.data.body_pos_w[
            :, self._left_wrist_idx, :3
        ]
        right_curr = self.robot.data.body_pos_w[
            :, self._right_wrist_idx, :3
        ]
        self._left_curr_marker.visualize(left_curr, default_quat)
        self._right_curr_marker.visualize(right_curr, default_quat)

        # Target wrist positions (translucent pink/cyan — cached)
        self._left_target_marker.visualize(
            self._target_left_pos_w, default_quat
        )
        self._right_target_marker.visualize(
            self._target_right_pos_w, default_quat
        )


@configclass
class BodyPoseCommandCfg(CommandTermCfg):
    """Configuration for the body pose command term.

    Attributes:
        asset_name: Name of the robot articulation in the scene.
        joint_names: Regex patterns for ALL 29 body joints.
        leg_joint_names: Regex patterns for leg joints.
        waist_joint_names: Regex patterns for waist roll/pitch joints.
        waist_yaw_joint_names: Regex patterns for waist yaw joint.
        arm_joint_names: Regex patterns for arm joints.
        leg_offset_range: Uniform offset range for legs (rad).
        waist_offset_range: Uniform offset range for waist roll/pitch (rad).
        waist_yaw_offset_range: Max offset range for waist yaw (rad); curriculum-scaled.
        arm_offset_range: Max offset range for arms (rad); scaled by curriculum.
        initial_ratio: Starting curriculum ratio (0 = arms at default).
        interpolation_duration: Time in seconds to interpolate to new target.
        left_wrist_body_name: Body name for left wrist FK visualization.
        right_wrist_body_name: Body name for right wrist FK visualization.
    """

    class_type: type = BodyPoseCommand
    asset_name: str = "robot"

    joint_names: list[str] = MISSING
    leg_joint_names: list[str] = field(default_factory=lambda: [
        ".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_hip_yaw_joint",
        ".*_knee_joint", ".*_ankle_pitch_joint", ".*_ankle_roll_joint",
    ])
    waist_joint_names: list[str] = field(default_factory=lambda: [
        "waist_roll_joint", "waist_pitch_joint",
    ])
    waist_yaw_joint_names: list[str] = field(default_factory=lambda: [
        "waist_yaw_joint",
    ])
    arm_joint_names: list[str] = field(default_factory=lambda: [
        ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint", ".*_elbow_joint",
        ".*_wrist_roll_joint", ".*_wrist_pitch_joint",
        ".*_wrist_yaw_joint",
    ])

    leg_offset_range: tuple[float, float] = (-0.05, 0.05)
    waist_offset_range: tuple[float, float] = (-0.2, 0.2)
    waist_yaw_offset_range: tuple[float, float] = (-1.0, 1.0)
    arm_offset_range: tuple[float, float] = (-0.5, 0.5)
    initial_ratio: float = 0.0
    interpolation_duration: float = 1.0

    left_wrist_body_name: str = "left_wrist_yaw_link"
    right_wrist_body_name: str = "right_wrist_yaw_link"

    debug_vis: bool = True
