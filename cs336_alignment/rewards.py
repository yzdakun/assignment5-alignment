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
    reward_infos: list[dict[str, float]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute raw rewards and group-normalized advantages."""
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError("rollout_responses and repeated_ground_truths must have the same length")
    if len(rollout_responses) % group_size != 0:
        raise ValueError("Number of rollout responses must be divisible by group_size")

    if reward_infos is None:
        reward_infos = [
            reward_fn(response, ground_truth)
            for response, ground_truth in zip(rollout_responses, repeated_ground_truths, strict=True)
        ]
    elif len(reward_infos) != len(rollout_responses):
        raise ValueError("reward_infos must have the same length as rollout_responses")
    raw_rewards = torch.tensor(
        [reward_info["reward"] for reward_info in reward_infos],
        dtype=torch.float32,
    )

    grouped_rewards = raw_rewards.view(-1, group_size)
    group_means = grouped_rewards.mean(dim=1, keepdim=True)
    centered_rewards = grouped_rewards - group_means

    if normalize_by_std:
        group_stds = grouped_rewards.std(dim=1, keepdim=True)
        advantages = centered_rewards / (group_stds + advantage_eps)
    else:
        group_stds = grouped_rewards.std(dim=1, keepdim=True)
        advantages = centered_rewards

    flattened_advantages = advantages.reshape(-1)
    metadata = {
        "mean_raw_reward": raw_rewards.mean().item(),
        "std_raw_reward": raw_rewards.std().item(),
        "mean_group_reward_std": group_stds.mean().item(),
        "normalize_by_std": float(normalize_by_std),
    }
    return flattened_advantages, raw_rewards, metadata


__all__ = [
    "compute_group_normalized_rewards",
    "question_only_reward_fn",
    "r1_zero_reward_fn",
]
