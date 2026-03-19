from __future__ import annotations

from typing import Literal

import torch


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """Average tensor values over masked positions."""
    masked_tensor = tensor * mask.to(dtype=tensor.dtype)
    return masked_tensor.mean(dim=dim)
    raise NotImplementedError


def masked_normalize(
    tensor: torch.Tensor, # [batch_size, sequence_length]
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """Sum over masked positions and divide by a constant."""
    masked_tensor = tensor * mask.to(dtype=tensor.dtype)
    return masked_tensor.sum(dim=dim) / normalize_constant
    raise NotImplementedError

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor, # [batch_size, sequence_length]
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run one SFT microbatch train step and backpropagate."""
    per_token_loss = -policy_log_probs
    loss = masked_normalize(
        tensor=per_token_loss,
        mask=response_mask,
        normalize_constant=normalize_constant,
        dim = 1
    ).mean()
    loss = loss / gradient_accumulation_steps
    loss.backward()

    metadata = {
        "unnormalized_loss": masked_normalize(
            tensor=per_token_loss,
            mask=response_mask,
            normalize_constant=normalize_constant,
        ).detach(),
    }
    return loss, metadata
    raise NotImplementedError



def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute naive policy gradient loss."""
    raise NotImplementedError


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute per-token GRPO-Clip loss."""
    raise NotImplementedError


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Dispatch to the selected policy gradient loss."""
    raise NotImplementedError


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run one GRPO microbatch train step and backpropagate."""
    raise NotImplementedError
