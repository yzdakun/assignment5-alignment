from __future__ import annotations

from collections.abc import Callable

import torch

from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute raw rewards and group-normalized advantages."""
    raise NotImplementedError


__all__ = [
    "compute_group_normalized_rewards",
    "question_only_reward_fn",
    "r1_zero_reward_fn",
]
