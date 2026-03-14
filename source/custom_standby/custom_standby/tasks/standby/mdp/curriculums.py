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


_LAST_PRINTED_RATIO: float = -1.0


def arm_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    step_size: float = 0.002,
    min_mean_ep_buf: float = 400.0,
) -> torch.Tensor:
    """Advance arm curriculum ratio when robot survives long enough.

    Uses the instantaneous mean of ``episode_length_buf`` across ALL
    environments.  Because resets are staggered, this buffer holds the
    *current* step count (not completed length) so its mean plateaus
    at roughly half the max episode length for a well-standing policy.
    The threshold is set accordingly.

    Args:
        env: The RL environment instance.
        env_ids: Indices of environments being reset (unused; global
            mean is used instead).
        step_size: How much to increase the ratio per curriculum call.
        min_mean_ep_buf: Minimum global mean of ``episode_length_buf``
            before curriculum advances.

    Returns:
        torch.Tensor: Current arm curriculum ratio in [0, 1].
    """
    global _LAST_PRINTED_RATIO
    command_term = env.command_manager.get_term("body_targets")

    global_mean_ep_len = torch.mean(
        env.episode_length_buf.float()
    ).item()

    if global_mean_ep_len > min_mean_ep_buf:
        command_term._curriculum_ratio = min(
            command_term._curriculum_ratio + step_size, 1.0
        )

    if abs(command_term._curriculum_ratio - _LAST_PRINTED_RATIO) > 0.005:
        _LAST_PRINTED_RATIO = command_term._curriculum_ratio
        print(
            f"[Curriculum] arm_cmd_levels={command_term._curriculum_ratio:.3f}"
            f"  global_mean_ep_buf={global_mean_ep_len:.1f}"
            f"  threshold={min_mean_ep_buf}"
        )

    return torch.tensor(
        command_term._curriculum_ratio, device=env.device
    )
