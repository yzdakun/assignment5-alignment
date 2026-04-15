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
    return masked_tensor.sum(dim=dim) / mask.sum(dim=dim)
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
    return -(raw_rewards_or_advantages * policy_log_probs)
    raise NotImplementedError


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute per-token GRPO-Clip loss."""
    if policy_log_probs.shape != old_log_probs.shape:
        raise ValueError("policy_log_probs and old_log_probs must have the same shape")
    if advantages.ndim != 2:
        raise ValueError("advantages must have shape (batch_size, 1) or (batch_size, sequence_length)")
    if advantages.shape[0] != policy_log_probs.shape[0]:
        raise ValueError("advantages batch dimension must match policy_log_probs")
    if advantages.shape[1] not in (1, policy_log_probs.shape[1]):
        raise ValueError("advantages must broadcast over the sequence dimension")

    # One scalar advantage is shared by every token in the same response.
    broadcast_advantages = advantages.expand_as(policy_log_probs)

    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    unclipped_objective = ratio * broadcast_advantages
    clipped_objective = clipped_ratio * broadcast_advantages
    per_token_objective = torch.minimum(unclipped_objective, clipped_objective)
    per_token_loss = -per_token_objective

    metadata = {
        "ratio": ratio.detach(),
        "clipped_ratio": clipped_ratio.detach(),
        "is_clipped": (ratio != clipped_ratio).detach(),
        "clip_fraction": (ratio != clipped_ratio).to(dtype=policy_log_probs.dtype).mean().detach(),
    }
    return per_token_loss, metadata


def compute_grpo_no_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the unclipped off-policy GRPO loss."""
    if policy_log_probs.shape != old_log_probs.shape:
        raise ValueError("policy_log_probs and old_log_probs must have the same shape")
    if advantages.ndim != 2:
        raise ValueError("advantages must have shape (batch_size, 1) or (batch_size, sequence_length)")
    if advantages.shape[0] != policy_log_probs.shape[0]:
        raise ValueError("advantages batch dimension must match policy_log_probs")
    if advantages.shape[1] not in (1, policy_log_probs.shape[1]):
        raise ValueError("advantages must broadcast over the sequence dimension")

    broadcast_advantages = advantages.expand_as(policy_log_probs)
    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    per_token_loss = -(ratio * broadcast_advantages)
    metadata = {
        "ratio": ratio.detach(),
        "clip_fraction": torch.zeros((), dtype=policy_log_probs.dtype, device=policy_log_probs.device),
    }
    return per_token_loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Dispatch to the selected policy gradient loss."""
    if loss_type == "no_baseline":
        if raw_rewards is None:
            raise ValueError("raw_rewards must be provided for no_baseline loss")
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=raw_rewards,
            policy_log_probs=policy_log_probs,
        )
        return loss, {"loss_type": loss_type}
    if loss_type == "reinforce_with_baseline":
        if advantages is None:
            raise ValueError("advantages must be provided for reinforce_with_baseline loss")
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=advantages,
            policy_log_probs=policy_log_probs,
        )
        return loss, {"loss_type": loss_type}
    if loss_type == "grpo_clip":
        if advantages is None or old_log_probs is None or cliprange is None:
            raise ValueError("advantages, old_log_probs, and cliprange must be provided for grpo_clip loss")
        loss, metadata = compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )
        return loss, {"loss_type": loss_type, **metadata}
    if loss_type == "grpo_no_clip":
        if advantages is None or old_log_probs is None:
            raise ValueError("advantages and old_log_probs must be provided for grpo_no_clip loss")
        loss, metadata = compute_grpo_no_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
        )
        return loss, {"loss_type": loss_type, **metadata}
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    normalize_mode: Literal["masked_mean", "masked_normalize"] = "masked_mean",
    normalize_constant: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run one GRPO microbatch train step and backpropagate."""
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    # Average only over response tokens so prompt/padding tokens do not affect
    # the gradient signal.
    if normalize_mode == "masked_mean":
        per_example_loss = masked_mean(
            tensor=per_token_loss,
            mask=response_mask,
            dim=1,
        )
    elif normalize_mode == "masked_normalize":
        if normalize_constant is None:
            normalize_constant = float(response_mask.shape[1])
        per_example_loss = masked_normalize(
            tensor=per_token_loss,
            mask=response_mask,
            normalize_constant=normalize_constant,
            dim=1,
        )
    else:
        raise ValueError(f"Unsupported normalize_mode: {normalize_mode}")

    loss = per_example_loss.mean()
    loss = loss / gradient_accumulation_steps
    loss.backward()

    return loss, {
        **metadata,
        "normalize_mode": normalize_mode,
        "unnormalized_loss": per_example_loss.detach(),
    }
