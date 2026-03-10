"""Curriculum terms for G1 XHand standby controller.

Progressively widens arm command offset ranges using OpenHomie's
exponential curriculum as the policy demonstrates stable standing.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def arm_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "joint_target_tracking",
    step_size: float = 0.05,
    threshold_ratio: float = 0.6,
) -> torch.Tensor:
    """Advance arm curriculum ratio when pose tracking is sufficient.

    Checks the mean episode reward for the tracking term on every
    qualifying env reset.  If it exceeds ``threshold_ratio`` of the
    reward weight, the ``_curriculum_ratio`` is advanced.

    Args:
        env: The RL environment instance.
        env_ids: Indices of environments being evaluated.
        reward_term_name: Reward term to monitor for curriculum progression.
        step_size: How much to increase the ratio per trigger.
        threshold_ratio: Fraction of the reward weight that must be exceeded.

    Returns:
        torch.Tensor: Current arm curriculum ratio in [0, 1].
    """
    command_term = env.command_manager.get_term("body_targets")

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = (
        torch.mean(
            env.reward_manager._episode_sums[reward_term_name][env_ids]
        )
        / env.max_episode_length_s
    )

    # Reason: negative weight means penalty — compare absolute values
    threshold = abs(reward_term.weight) * threshold_ratio
    if abs(reward) < threshold:
        command_term._curriculum_ratio = min(
            command_term._curriculum_ratio + step_size, 1.0
        )

    return torch.tensor(
        command_term._curriculum_ratio, device=env.device
    )
