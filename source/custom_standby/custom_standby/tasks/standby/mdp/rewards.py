"""Custom reward functions for the G1 XHand standby controller.

Focused on maintaining upright stance while tracking arbitrary body
joint position commands.  No locomotion rewards (gait, feet clearance,
velocity tracking, etc.).
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def body_target_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize deviation of all body joints from commanded targets.

    Args:
        env: The RL environment instance.
        command_name: Name of the body pose command term.
        asset_cfg: Scene entity config with the 29 body joint IDs.

    Returns:
        torch.Tensor: Per-environment L1 tracking error.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    targets = env.command_manager.get_command(command_name)
    actual_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(actual_pos - targets), dim=1)


def base_xy_vel_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize lateral base velocity to enforce standing still.

    Args:
        env: The RL environment instance.
        asset_cfg: Scene entity configuration for the robot.

    Returns:
        torch.Tensor: Per-environment L2 squared xy velocity.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(
        torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1
    )


def upward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize deviation from upright orientation.

    Args:
        env: The RL environment instance.
        asset_cfg: Scene entity configuration for the robot.

    Returns:
        torch.Tensor: Per-environment upward penalty.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(1 - asset.data.projected_gravity_b[:, 2])


def energy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize mechanical energy (|torque * velocity|) summed across joints.

    Args:
        env: The RL environment instance.
        asset_cfg: Scene entity configuration for the robot.

    Returns:
        torch.Tensor: Per-environment scalar energy penalty.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def feet_contact_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward both feet being in contact with the ground.

    Args:
        env: The RL environment instance.
        sensor_cfg: Contact sensor configuration for feet.

    Returns:
        torch.Tensor: Number of feet in contact (0, 1, or 2).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = (
        contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    )
    return torch.sum(is_contact, dim=-1).float()
