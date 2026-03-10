"""Custom event functions for domain randomization.

Adapted from whole_body_tracking for sim2real robustness.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize joint default positions to simulate calibration errors.

    Args:
        env: The environment instance.
        env_ids: Environment indices to randomize.
        asset_cfg: Scene entity config for the robot.
        pos_distribution_params: Min/max for the randomization.
        operation: How to combine with nominal values.
        distribution: Sampling distribution type.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Reason: save the original default pos before first randomization
    # so the exporter can embed the true nominal values in ONNX metadata.
    if not hasattr(asset.data, "default_joint_pos_nominal"):
        asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids,
            operation=operation, distribution=distribution,
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos

        # Reason: update the action manager offset so position targets
        # remain consistent with the new defaults. Wrapped in try/except
        # because our 15-DOF action space doesn't cover all 53 joints.
        try:
            action_term = env.action_manager.get_term("joint_pos")
            new_defaults = asset.data.default_joint_pos[:, action_term._joint_ids]
            action_term._offset[:] = new_defaults
        except (AttributeError, KeyError, IndexError):
            pass


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize center of mass of rigid bodies.

    Adds a random offset sampled from the given ranges to simulate
    payload or manufacturing variation.

    Args:
        env: The environment instance.
        env_ids: Environment indices to randomize.
        com_range: Per-axis (x, y, z) offset ranges in meters.
        asset_cfg: Scene entity config for the robot.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu",
    ).unsqueeze(1)

    coms = asset.root_physx_view.get_coms().clone()
    coms[:, body_ids, :3] += rand_samples
    asset.root_physx_view.set_coms(coms, env_ids)
