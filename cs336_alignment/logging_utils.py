from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from tokenization import tokenize_prompt_and_output
from modeling import get_response_log_probs
from losses import masked_mean

def log_generations(
    vllm_model,
    policy_model,
    tokenizer,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    ground_truths: list[str],
    sampling_params,
    max_examples_to_log: int = 8,
) -> dict[str, Any]:
    if len(prompts) != len(ground_truths):
        raise ValueError("prompts and ground_truths must have the same length")

    outputs = vllm_model.generate(prompts, sampling_params)
    responses = [output.outputs[0].text for output in outputs]

    reward_infos = [
        reward_fn(response, ground_truth)
        for response, ground_truth in zip(responses, ground_truths, strict=True)
    ]

    batch = tokenize_prompt_and_output(prompts, responses, tokenizer)
    device = next(policy_model.parameters()).device

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    response_mask = batch["response_mask"].to(device)

    with torch.inference_mode():
        scored = get_response_log_probs(
            model=policy_model,
            input_ids=input_ids,
            labels=labels,
            return_token_entropy=True,
        )

    avg_token_entropy = masked_mean(
        scored["token_entropy"],
        response_mask,
        dim=1,
    ).detach().cpu()

    response_lengths = response_mask.sum(dim=1).detach().cpu().to(torch.float32)

    total_rewards = torch.tensor([info["reward"] for info in reward_infos], dtype=torch.float32)
    format_rewards = torch.tensor([info["format_reward"] for info in reward_infos], dtype=torch.float32)
    answer_rewards = torch.tensor([info["answer_reward"] for info in reward_infos], dtype=torch.float32)

    correct_mask = answer_rewards > 0
    incorrect_mask = ~correct_mask

    def safe_mean(values: torch.Tensor, mask: torch.Tensor | None = None) -> float:
        if mask is None:
            return values.mean().item()
        if mask.any():
            return values[mask].mean().item()
        return 0.0

    examples = []
    for i in range(min(max_examples_to_log, len(prompts))):
        examples.append(
            {
                "prompt": prompts[i],
                "response": responses[i],
                "ground_truth": ground_truths[i],
                "format_reward": reward_infos[i]["format_reward"],
                "answer_reward": reward_infos[i]["answer_reward"],
                "reward": reward_infos[i]["reward"],
                "avg_token_entropy": avg_token_entropy[i].item(),
                "response_length": response_lengths[i].item(),
            }
        )

    return {
        "mean_reward": total_rewards.mean().item(),
        "mean_format_reward": format_rewards.mean().item(),
        "mean_answer_reward": answer_rewards.mean().item(),
        "mean_token_entropy": avg_token_entropy.mean().item(),
        "mean_response_length": response_lengths.mean().item(),
        "mean_response_length_correct": safe_mean(response_lengths, correct_mask),
        "mean_response_length_incorrect": safe_mean(response_lengths, incorrect_mask),
        "examples": examples,
    }
