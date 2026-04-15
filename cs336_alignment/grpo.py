from __future__ import annotations

import gc
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable, Literal

import torch
import typer

from .data import format_prompt, load_math_qa_examples, read_prompt_template
from .losses import grpo_microbatch_train_step, masked_mean
from .modeling import (
    get_response_log_probs,
    init_vllm,
    load_policy_and_tokenizer,
    load_policy_into_vllm_instance,
)
from .rewards import compute_group_normalized_rewards, question_only_reward_fn, r1_zero_reward_fn
from .tokenization import tokenize_prompt_and_output

LossType = Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"]
NormalizeMode = Literal["masked_mean", "masked_normalize"]
_EXPLICIT_NONE = object()

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Run GRPO experiments for the CS336 alignment assignment.",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _normalize_optional_device(device: str | None) -> str | None:
    if device is None:
        return None
    if device.lower() in {"none", "null", "cpu-none"}:
        return None
    return device


def _optional_device_override(device: str | None) -> str | None | object:
    if device is None:
        return None
    normalized = _normalize_optional_device(device)
    if normalized is None:
        return _EXPLICIT_NONE
    return normalized


def _parse_torch_dtype(dtype_name: str) -> torch.dtype:
    normalized = dtype_name.lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[normalized]


def _cli_bool_override(true_flag: str, false_flag: str) -> bool | None:
    argv = set(sys.argv[1:])
    if true_flag in argv:
        return True
    if false_flag in argv:
        return False
    return None


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def _profile_defaults(profile: str) -> dict[str, Any]:
    root = _repo_root()
    if profile == "local-smoke":
        return {
            "model_path": root / "models" / "Qwen2.5-Math-1.5B",
            "train_dataset_path": root / "data" / "MATH" / "train.jsonl",
            "validation_dataset_path": root / "data" / "MATH" / "validation.jsonl",
            "prompt_path": root / "cs336_alignment" / "prompts" / "r1_zero.prompt",
            "validation_prompt_path": root / "cs336_alignment" / "prompts" / "r1_zero.prompt",
            "output_dir": root / "cs336_alignment" / "results" / "grpo_experiment",
            "train_device": "cuda:0",
            "rollout_device": "cuda:0",
            "eval_device": None,
            "learning_rate": 1e-5,
            "weight_decay": 0.0,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "n_grpo_steps": 10,
            "advantage_eps": 1e-6,
            "rollout_batch_size": 64,
            "group_size": 4,
            "sampling_temperature": 1.0,
            "sampling_min_tokens": 4,
            "sampling_max_tokens": 256,
            "epochs_per_rollout_batch": 1,
            "train_batch_size": 64,
            "gradient_accumulation_steps": 32,
            "gpu_memory_utilization": 0.75,
            "loss_type": "reinforce_with_baseline",
            "normalize_loss_by": "masked_mean",
            "loss_normalize_constant": None,
            "use_std_normalization": True,
            "cliprange": 0.2,
            "max_grad_norm": 1.0,
            "max_train_examples": 128,
            "eval_num_examples": 64,
            "eval_every_steps": 5,
            "final_eval_num_examples": 64,
            "run_final_full_eval": False,
            "save_rollouts": True,
            "save_checkpoint": False,
            "save_best_checkpoint": False,
            "reward_fn_name": "r1_zero",
            "validation_reward_fn_name": "r1_zero",
            "seed": 42,
            "shuffle": True,
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
            "use_wandb": False,
            "wandb_project": "cs336-alignment-grpo",
            "wandb_run_name": None,
            "wandb_mode": None,
            "wandb_tags": ("grpo", "local-smoke"),
        }
    if profile == "cloud":
        return {
            "model_path": root / "models" / "Qwen2.5-Math-1.5B",
            "train_dataset_path": root / "data" / "MATH" / "train.jsonl",
            "validation_dataset_path": root / "data" / "MATH" / "validation.jsonl",
            "prompt_path": root / "cs336_alignment" / "prompts" / "r1_zero.prompt",
            "validation_prompt_path": root / "cs336_alignment" / "prompts" / "r1_zero.prompt",
            "output_dir": root / "cs336_alignment" / "results" / "grpo_experiment",
            "train_device": "cuda:0",
            "rollout_device": "cuda:1",
            "eval_device": "cuda:1",
            "learning_rate": 1e-5,
            "weight_decay": 0.0,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "n_grpo_steps": 200,
            "advantage_eps": 1e-6,
            "rollout_batch_size": 256,
            "group_size": 8,
            "sampling_temperature": 1.0,
            "sampling_min_tokens": 4,
            "sampling_max_tokens": 1024,
            "epochs_per_rollout_batch": 1,
            "train_batch_size": 256,
            "gradient_accumulation_steps": 128,
            "gpu_memory_utilization": 0.85,
            "loss_type": "reinforce_with_baseline",
            "normalize_loss_by": "masked_mean",
            "loss_normalize_constant": None,
            "use_std_normalization": True,
            "cliprange": 0.2,
            "max_grad_norm": 1.0,
            "max_train_examples": None,
            "eval_num_examples": 1024,
            "eval_every_steps": 10,
            "final_eval_num_examples": None,
            "run_final_full_eval": True,
            "save_rollouts": False,
            "save_checkpoint": True,
            "save_best_checkpoint": True,
            "reward_fn_name": "r1_zero",
            "validation_reward_fn_name": "r1_zero",
            "seed": 42,
            "shuffle": True,
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
            "use_wandb": False,
            "wandb_project": "cs336-alignment-grpo",
            "wandb_run_name": None,
            "wandb_mode": None,
            "wandb_tags": ("grpo", "cloud"),
        }
    raise ValueError(f"Unsupported profile: {profile}")


def _merge_profile_overrides(profile: str, overrides: dict[str, Any]) -> dict[str, Any]:
    config = _profile_defaults(profile)
    for key, value in overrides.items():
        if value is _EXPLICIT_NONE:
            config[key] = None
        elif value is not None:
            config[key] = value
    return config


def _get_reward_fn(name: str) -> Callable[[str, str], dict[str, float]]:
    normalized = name.lower()
    mapping = {
        "r1_zero": r1_zero_reward_fn,
        "question_only": question_only_reward_fn,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported reward function: {name}")
    return mapping[normalized]


def _select_unique_examples(
    examples: list[dict[str, Any]],
    max_examples: int | None,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    if max_examples is None or max_examples >= len(examples):
        return list(examples)

    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(range(len(examples)), k=max_examples))
    return [examples[idx] for idx in selected_indices]


def _select_step_examples(
    examples: list[dict[str, Any]],
    batch_size: int,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if batch_size > len(examples):
        raise ValueError("Not enough training examples to sample a full rollout batch")

    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(range(len(examples)), k=batch_size))
    return [examples[idx] for idx in selected_indices]


def _build_sampling_params(
    *,
    max_tokens: int,
    min_tokens: int,
    seed: int,
    n: int = 1,
    temperature: float = 1.0,
    top_p: float = 1.0,
):
    from vllm import SamplingParams

    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        n=n,
        seed=seed,
    )


def _summarize_reward_infos(reward_infos: list[dict[str, float]]) -> dict[str, float]:
    num_examples = len(reward_infos)
    if num_examples == 0:
        return {
            "num_examples": 0,
            "avg_reward": 0.0,
            "avg_format_reward": 0.0,
            "avg_answer_reward": 0.0,
            "count_format1_answer1": 0,
            "count_format1_answer0": 0,
            "count_format0_answer0": 0,
        }

    return {
        "num_examples": num_examples,
        "avg_reward": sum(x["reward"] for x in reward_infos) / num_examples,
        "avg_format_reward": sum(x["format_reward"] for x in reward_infos) / num_examples,
        "avg_answer_reward": sum(x["answer_reward"] for x in reward_infos) / num_examples,
        "count_format1_answer1": sum(
            x["format_reward"] == 1.0 and x["answer_reward"] == 1.0 for x in reward_infos
        ),
        "count_format1_answer0": sum(
            x["format_reward"] == 1.0 and x["answer_reward"] == 0.0 for x in reward_infos
        ),
        "count_format0_answer0": sum(
            x["format_reward"] == 0.0 and x["answer_reward"] == 0.0 for x in reward_infos
        ),
    }


def _prefix_metrics(prefix: str, metrics: dict[str, float | int]) -> dict[str, float | int]:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


def _extract_numeric_metrics(metrics: dict[str, Any]) -> dict[str, float | int]:
    numeric_metrics: dict[str, float | int] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            numeric_metrics[key] = value
    return numeric_metrics


def _maybe_init_wandb(
    *,
    use_wandb: bool,
    output_dir: Path,
    config: dict[str, Any],
    project: str,
    run_name: str | None,
    mode: str | None,
    tags: tuple[str, ...] | None,
):
    if not use_wandb:
        return None

    import wandb

    wandb_run = wandb.init(
        project=project,
        name=run_name,
        dir=str(output_dir),
        config=_jsonify(config),
        mode=mode,
        tags=list(tags) if tags is not None else None,
    )
    wandb.define_metric("optimizer_step")
    wandb.define_metric("grpo_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="optimizer_step")
    wandb.define_metric("rollout/*", step_metric="grpo_step")
    wandb.define_metric("eval/*", step_metric="eval_step")
    return wandb_run


def _score_response_batch(
    *,
    policy_model,
    tokenizer,
    prompts: list[str],
    responses: list[str],
    device: str,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor]:
    batch = tokenize_prompt_and_output(
        prompt_strs=prompts,
        output_strs=responses,
        tokenizer=tokenizer,
    )
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    response_mask = batch["response_mask"].to(device)
    scored = get_response_log_probs(
        model=policy_model,
        input_ids=input_ids,
        labels=labels,
        return_token_entropy=return_token_entropy,
    )
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
        **scored,
    }


def _build_rollout_batch(
    *,
    policy_model,
    rollout_llm,
    rollout_examples: list[dict[str, Any]],
    prompt_template: str,
    reward_fn: Callable[[str, str], dict[str, float]],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
    rollout_sampling_params,
    grpo_step: int,
) -> dict[str, Any]:
    load_policy_into_vllm_instance(policy_model, rollout_llm)

    prompts = [format_prompt(example["problem"], prompt_template) for example in rollout_examples]
    outputs = rollout_llm.generate(prompts=prompts, sampling_params=rollout_sampling_params)

    flat_prompts: list[str] = []
    flat_responses: list[str] = []
    flat_ground_truths: list[str] = []
    reward_infos: list[dict[str, float]] = []
    rows: list[dict[str, Any]] = []

    for example_idx, (example, prompt, output_group) in enumerate(
        zip(rollout_examples, prompts, outputs, strict=True)
    ):
        ground_truth = example["answer"]
        for rollout_idx, candidate in enumerate(output_group.outputs):
            response = candidate.text
            reward_info = reward_fn(response, ground_truth)
            flat_prompts.append(prompt)
            flat_responses.append(response)
            flat_ground_truths.append(ground_truth)
            reward_infos.append(reward_info)
            rows.append(
                {
                    "grpo_step": grpo_step,
                    "problem_index": example_idx,
                    "rollout_index": rollout_idx,
                    "problem": example["problem"],
                    "ground_truth": ground_truth,
                    "prompt": prompt,
                    "response": response,
                    **reward_info,
                }
            )

    advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
        reward_fn=reward_fn,
        rollout_responses=flat_responses,
        repeated_ground_truths=flat_ground_truths,
        group_size=group_size,
        advantage_eps=advantage_eps,
        normalize_by_std=normalize_by_std,
        reward_infos=reward_infos,
    )

    return {
        "prompts": flat_prompts,
        "responses": flat_responses,
        "ground_truths": flat_ground_truths,
        "reward_infos": reward_infos,
        "advantages": advantages,
        "raw_rewards": raw_rewards,
        "reward_metadata": reward_metadata,
        "rows": rows,
    }


def _evaluate_policy(
    *,
    policy_model,
    tokenizer,
    llm,
    validation_examples: list[dict[str, Any]],
    prompt_template: str,
    reward_fn: Callable[[str, str], dict[str, float]],
    sampling_params,
    score_device: str,
    max_examples_to_log: int = 8,
) -> dict[str, Any]:
    prompts = [format_prompt(example["problem"], prompt_template) for example in validation_examples]
    ground_truths = [example["answer"] for example in validation_examples]

    policy_model.eval()
    load_policy_into_vllm_instance(policy_model, llm)

    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    reward_infos = [
        reward_fn(response, ground_truth)
        for response, ground_truth in zip(responses, ground_truths, strict=True)
    ]

    policy_model.eval()
    scored = _score_response_batch(
        policy_model=policy_model,
        tokenizer=tokenizer,
        prompts=prompts,
        responses=responses,
        device=score_device,
        return_token_entropy=True,
    )
    avg_token_entropy = masked_mean(
        scored["token_entropy"],
        scored["response_mask"],
        dim=1,
    ).detach().cpu()
    response_lengths = scored["response_mask"].sum(dim=1).detach().cpu().to(torch.float32)
    policy_model.train()

    answer_rewards = torch.tensor(
        [reward_info["answer_reward"] for reward_info in reward_infos],
        dtype=torch.float32,
    )
    correct_mask = answer_rewards > 0
    incorrect_mask = ~correct_mask

    def safe_mean(values: torch.Tensor, mask: torch.Tensor | None = None) -> float:
        if mask is None:
            return values.mean().item()
        if mask.any():
            return values[mask].mean().item()
        return 0.0

    preview = []
    for idx in range(min(max_examples_to_log, len(validation_examples))):
        preview.append(
            {
                "prompt": prompts[idx],
                "response": responses[idx],
                "ground_truth": ground_truths[idx],
                "reward": reward_infos[idx]["reward"],
                "format_reward": reward_infos[idx]["format_reward"],
                "answer_reward": reward_infos[idx]["answer_reward"],
                "avg_token_entropy": avg_token_entropy[idx].item(),
                "response_length": response_lengths[idx].item(),
            }
        )

    return {
        **_summarize_reward_infos(reward_infos),
        "mean_token_entropy": avg_token_entropy.mean().item(),
        "mean_response_length": response_lengths.mean().item(),
        "mean_response_length_correct": safe_mean(response_lengths, correct_mask),
        "mean_response_length_incorrect": safe_mean(response_lengths, incorrect_mask),
        "examples": preview,
    }


def _effective_normalize_constant(
    *,
    normalize_loss_by: NormalizeMode,
    loss_normalize_constant: float | None,
    sampling_max_tokens: int,
) -> float | None:
    if normalize_loss_by == "masked_mean":
        return loss_normalize_constant
    if loss_normalize_constant is not None:
        return loss_normalize_constant
    return float(sampling_max_tokens)


def run_grpo_experiment(
    *,
    model_path: str | Path | None = None,
    train_dataset_path: str | Path | None = None,
    validation_dataset_path: str | Path | None = None,
    prompt_path: str | Path | None = None,
    validation_prompt_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    train_device: str = "cuda:0",
    rollout_device: str = "cuda:1",
    eval_device: str | None = "cuda:1",
    learning_rate: float = 1e-5,
    weight_decay: float = 0.0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.95,
    n_grpo_steps: int = 200,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 1024,
    epochs_per_rollout_batch: int = 1,
    train_batch_size: int = 256,
    gradient_accumulation_steps: int = 128,
    gpu_memory_utilization: float = 0.85,
    loss_type: LossType = "reinforce_with_baseline",
    normalize_loss_by: NormalizeMode = "masked_mean",
    loss_normalize_constant: float | None = None,
    use_std_normalization: bool = True,
    cliprange: float = 0.2,
    max_grad_norm: float = 1.0,
    max_train_examples: int | None = None,
    eval_num_examples: int | None = 1024,
    eval_every_steps: int = 10,
    final_eval_num_examples: int | None = None,
    run_final_full_eval: bool = True,
    save_rollouts: bool = False,
    save_checkpoint: bool = True,
    save_best_checkpoint: bool = True,
    reward_fn_name: str = "r1_zero",
    validation_reward_fn_name: str | None = None,
    seed: int = 42,
    shuffle: bool = True,
    torch_dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "flash_attention_2",
    use_wandb: bool = False,
    wandb_project: str = "cs336-alignment-grpo",
    wandb_run_name: str | None = None,
    wandb_mode: str | None = None,
    wandb_tags: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Run one GRPO experiment."""

    if n_grpo_steps <= 0:
        raise ValueError("n_grpo_steps must be positive")
    if rollout_batch_size <= 0:
        raise ValueError("rollout_batch_size must be positive")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if rollout_batch_size % group_size != 0:
        raise ValueError("rollout_batch_size must be divisible by group_size")
    if train_batch_size <= 0:
        raise ValueError("train_batch_size must be positive")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if train_batch_size % gradient_accumulation_steps != 0:
        raise ValueError("train_batch_size must be divisible by gradient_accumulation_steps")
    if train_batch_size < group_size:
        raise ValueError("train_batch_size must be greater than or equal to group_size")
    if epochs_per_rollout_batch <= 0:
        raise ValueError("epochs_per_rollout_batch must be positive")
    if eval_every_steps < 0:
        raise ValueError("eval_every_steps must be non-negative")

    root = _repo_root()
    model_path = Path(model_path or root / "models" / "Qwen2.5-Math-1.5B")
    train_dataset_path = Path(train_dataset_path or root / "data" / "MATH" / "train.jsonl")
    validation_dataset_path = Path(
        validation_dataset_path or root / "data" / "MATH" / "validation.jsonl"
    )
    prompt_path = Path(prompt_path or root / "cs336_alignment" / "prompts" / "r1_zero.prompt")
    validation_prompt_path = Path(validation_prompt_path or prompt_path)
    output_dir = Path(output_dir or root / "cs336_alignment" / "results" / "grpo_experiment")
    output_dir.mkdir(parents=True, exist_ok=True)

    rollout_samples_dir = output_dir / "rollout_samples"
    if save_rollouts:
        rollout_samples_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(seed)

    prompt_template = read_prompt_template(prompt_path)
    validation_prompt_template = read_prompt_template(validation_prompt_path)
    reward_fn = _get_reward_fn(reward_fn_name)
    validation_reward_fn = _get_reward_fn(validation_reward_fn_name or reward_fn_name)

    full_train_examples = load_math_qa_examples(train_dataset_path)
    if not full_train_examples:
        raise ValueError("No training examples found")
    train_examples = _select_unique_examples(
        full_train_examples,
        max_train_examples,
        seed=seed,
    )

    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    if len(train_examples) < n_prompts_per_rollout_batch:
        raise ValueError("Training example pool is smaller than the number of prompts per rollout batch")

    full_validation_examples = load_math_qa_examples(validation_dataset_path)
    validation_examples = list(full_validation_examples)
    if eval_num_examples is not None:
        validation_examples = validation_examples[:eval_num_examples]
    final_validation_examples = list(full_validation_examples)
    if final_eval_num_examples is not None:
        final_validation_examples = final_validation_examples[:final_eval_num_examples]

    policy_model, tokenizer = load_policy_and_tokenizer(
        str(model_path),
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )
    policy_model.to(train_device)
    policy_model.train()

    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(adam_beta1, adam_beta2),
    )

    rollout_llm = init_vllm(
        model_id=str(model_path),
        device=rollout_device,
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    eval_llm = None
    if eval_device is not None:
        if eval_device == rollout_device:
            eval_llm = rollout_llm
        else:
            eval_llm = init_vllm(
                model_id=str(model_path),
                device=eval_device,
                seed=seed + 17,
                gpu_memory_utilization=gpu_memory_utilization,
            )

    rollout_sampling_params = _build_sampling_params(
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        n=group_size,
        temperature=sampling_temperature,
        seed=seed,
    )
    eval_sampling_params = _build_sampling_params(
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        n=1,
        temperature=sampling_temperature,
        seed=seed + 100,
    )

    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    effective_normalize_constant = _effective_normalize_constant(
        normalize_loss_by=normalize_loss_by,
        loss_normalize_constant=loss_normalize_constant,
        sampling_max_tokens=sampling_max_tokens,
    )

    config = {
        "model_path": model_path,
        "train_dataset_path": train_dataset_path,
        "validation_dataset_path": validation_dataset_path,
        "prompt_path": prompt_path,
        "validation_prompt_path": validation_prompt_path,
        "output_dir": output_dir,
        "train_device": train_device,
        "rollout_device": rollout_device,
        "eval_device": eval_device,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "adam_beta1": adam_beta1,
        "adam_beta2": adam_beta2,
        "n_grpo_steps": n_grpo_steps,
        "advantage_eps": advantage_eps,
        "rollout_batch_size": rollout_batch_size,
        "group_size": group_size,
        "sampling_temperature": sampling_temperature,
        "sampling_min_tokens": sampling_min_tokens,
        "sampling_max_tokens": sampling_max_tokens,
        "epochs_per_rollout_batch": epochs_per_rollout_batch,
        "train_batch_size": train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "micro_train_batch_size": micro_train_batch_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "loss_type": loss_type,
        "normalize_loss_by": normalize_loss_by,
        "loss_normalize_constant": effective_normalize_constant,
        "use_std_normalization": _cli_bool_override(
            "--use-std-normalization",
            "--no-std-normalization",
        ),
        "cliprange": cliprange,
        "max_grad_norm": max_grad_norm,
        "max_train_examples": max_train_examples,
        "num_train_examples_after_subset": len(train_examples),
        "eval_num_examples": eval_num_examples,
        "eval_every_steps": eval_every_steps,
        "final_eval_num_examples": final_eval_num_examples,
        "run_final_full_eval": _cli_bool_override(
            "--run-final-full-eval",
            "--skip-final-full-eval",
        ),
        "save_rollouts": _cli_bool_override("--save-rollouts", "--skip-save-rollouts"),
        "save_checkpoint": _cli_bool_override("--save-checkpoint", "--skip-save-checkpoint"),
        "save_best_checkpoint": _cli_bool_override(
            "--save-best-checkpoint",
            "--skip-save-best-checkpoint",
        ),
        "reward_fn_name": reward_fn_name,
        "validation_reward_fn_name": validation_reward_fn_name or reward_fn_name,
        "seed": seed,
        "shuffle": shuffle,
        "torch_dtype": torch_dtype,
        "attn_implementation": attn_implementation,
        "use_wandb": use_wandb,
        "wandb_project": wandb_project,
        "wandb_run_name": wandb_run_name,
        "wandb_mode": wandb_mode,
        "wandb_tags": wandb_tags,
    }
    history: dict[str, Any] = {
        "config": _jsonify(config),
        "initial_eval": None,
        "rollouts": [],
        "train": [],
        "eval": [],
        "final_eval": None,
    }

    wandb_run = _maybe_init_wandb(
        use_wandb=use_wandb,
        output_dir=output_dir,
        config=history["config"],
        project=wandb_project,
        run_name=wandb_run_name,
        mode=wandb_mode,
        tags=wandb_tags,
    )

    best_eval_answer_reward = float("-inf")
    best_eval: dict[str, Any] | None = None
    best_checkpoint_dir: Path | None = None
    optimizer_step = 0

    if eval_llm is not None and validation_examples:
        initial_eval = _evaluate_policy(
            policy_model=policy_model,
            tokenizer=tokenizer,
            llm=eval_llm,
            validation_examples=validation_examples,
            prompt_template=validation_prompt_template,
            reward_fn=validation_reward_fn,
            sampling_params=eval_sampling_params,
            score_device=train_device,
        )
        initial_eval["step"] = 0
        history["initial_eval"] = initial_eval
        if wandb_run is not None:
            wandb_run.log(
                {
                    "eval_step": 0,
                    **_prefix_metrics("eval", _extract_numeric_metrics(initial_eval)),
                }
            )
        best_eval = initial_eval
        best_eval_answer_reward = initial_eval["avg_answer_reward"]
        if save_best_checkpoint:
            best_checkpoint_dir = output_dir / "best_checkpoint"
            best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            policy_model.save_pretrained(best_checkpoint_dir)
            tokenizer.save_pretrained(best_checkpoint_dir)

    for grpo_step in range(1, n_grpo_steps + 1):
        step_examples = _select_step_examples(
            train_examples,
            n_prompts_per_rollout_batch,
            seed=seed + grpo_step,
        )
        rollout_batch = _build_rollout_batch(
            policy_model=policy_model,
            rollout_llm=rollout_llm,
            rollout_examples=step_examples,
            prompt_template=prompt_template,
            reward_fn=reward_fn,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=use_std_normalization,
            rollout_sampling_params=rollout_sampling_params,
            grpo_step=grpo_step,
        )
        if save_rollouts:
            _save_jsonl(
                rollout_samples_dir / f"step_{grpo_step:04d}.jsonl",
                rollout_batch["rows"],
            )

        policy_model.eval()
        with torch.inference_mode():
            initial_scored = _score_response_batch(
                policy_model=policy_model,
                tokenizer=tokenizer,
                prompts=rollout_batch["prompts"],
                responses=rollout_batch["responses"],
                device=train_device,
                return_token_entropy=True,
            )
        policy_model.train()

        response_mask = initial_scored["response_mask"]
        avg_token_entropy = masked_mean(
            initial_scored["token_entropy"],
            response_mask,
            dim=1,
        ).detach().cpu()
        response_lengths = response_mask.sum(dim=1).detach().cpu().to(torch.float32)

        reward_infos = rollout_batch["reward_infos"]
        raw_rewards = rollout_batch["raw_rewards"].to(train_device).unsqueeze(1)
        advantages = rollout_batch["advantages"].to(train_device).unsqueeze(1)
        old_log_probs = None
        if loss_type in {"grpo_clip", "grpo_no_clip"}:
            old_log_probs = initial_scored["log_probs"].detach()

        reward_tensor = torch.tensor(
            [reward_info["reward"] for reward_info in reward_infos],
            dtype=torch.float32,
        )
        format_reward_tensor = torch.tensor(
            [reward_info["format_reward"] for reward_info in reward_infos],
            dtype=torch.float32,
        )
        answer_reward_tensor = torch.tensor(
            [reward_info["answer_reward"] for reward_info in reward_infos],
            dtype=torch.float32,
        )

        rollout_entry = {
            "grpo_step": grpo_step,
            "num_prompts": len(step_examples),
            "num_rollouts": len(rollout_batch["responses"]),
            "avg_reward": reward_tensor.mean().item(),
            "avg_format_reward": format_reward_tensor.mean().item(),
            "avg_answer_reward": answer_reward_tensor.mean().item(),
            "mean_advantage": advantages.mean().item(),
            "std_advantage": advantages.std().item(),
            "mean_raw_reward": raw_rewards.mean().item(),
            "mean_token_entropy": avg_token_entropy.mean().item(),
            "mean_response_length": response_lengths.mean().item(),
            **rollout_batch["reward_metadata"],
        }
        history["rollouts"].append(rollout_entry)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "grpo_step": grpo_step,
                    **_prefix_metrics("rollout", rollout_entry),
                }
            )

        indices = list(range(len(rollout_batch["responses"])))
        mean_clip_fraction_values: list[float] = []

        for rollout_epoch in range(1, epochs_per_rollout_batch + 1):
            if shuffle:
                random.Random(seed + grpo_step + rollout_epoch).shuffle(indices)

            for batch_start in range(0, len(indices), train_batch_size):
                batch_indices = indices[batch_start : batch_start + train_batch_size]
                microbatches = [
                    batch_indices[i : i + micro_train_batch_size]
                    for i in range(0, len(batch_indices), micro_train_batch_size)
                ]
                actual_accumulation_steps = len(microbatches)
                optimizer.zero_grad(set_to_none=True)

                step_loss = 0.0
                step_raw_loss = 0.0
                step_clip_fractions: list[float] = []

                for microbatch_indices in microbatches:
                    microbatch_index_tensor = torch.tensor(
                        microbatch_indices,
                        dtype=torch.long,
                        device=train_device,
                    )
                    scored = get_response_log_probs(
                        model=policy_model,
                        input_ids=initial_scored["input_ids"].index_select(0, microbatch_index_tensor),
                        labels=initial_scored["labels"].index_select(0, microbatch_index_tensor),
                        return_token_entropy=False,
                    )
                    loss, metadata = grpo_microbatch_train_step(
                        policy_log_probs=scored["log_probs"],
                        response_mask=response_mask.index_select(0, microbatch_index_tensor),
                        gradient_accumulation_steps=actual_accumulation_steps,
                        loss_type=loss_type,
                        raw_rewards=raw_rewards.index_select(0, microbatch_index_tensor),
                        advantages=advantages.index_select(0, microbatch_index_tensor),
                        old_log_probs=(
                            old_log_probs.index_select(0, microbatch_index_tensor)
                            if old_log_probs is not None
                            else None
                        ),
                        cliprange=cliprange,
                        normalize_mode=normalize_loss_by,
                        normalize_constant=effective_normalize_constant,
                    )
                    step_loss += loss.detach().item()
                    step_raw_loss += metadata["unnormalized_loss"].mean().item()
                    clip_fraction = metadata.get("clip_fraction")
                    if clip_fraction is not None:
                        step_clip_fractions.append(float(clip_fraction.detach().mean().item()))

                grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_grad_norm)
                optimizer.step()
                policy_model.train()

                optimizer_step += 1
                mean_clip_fraction = (
                    sum(step_clip_fractions) / len(step_clip_fractions) if step_clip_fractions else 0.0
                )
                mean_clip_fraction_values.append(mean_clip_fraction)
                train_entry = {
                    "optimizer_step": optimizer_step,
                    "grpo_step": grpo_step,
                    "rollout_epoch": rollout_epoch,
                    "loss": step_loss / actual_accumulation_steps,
                    "unnormalized_loss": step_raw_loss / actual_accumulation_steps,
                    "grad_norm": float(grad_norm),
                    "clip_fraction": mean_clip_fraction,
                    "avg_reward": rollout_entry["avg_reward"],
                    "avg_format_reward": rollout_entry["avg_format_reward"],
                    "avg_answer_reward": rollout_entry["avg_answer_reward"],
                    "mean_advantage": rollout_entry["mean_advantage"],
                    "mean_token_entropy": rollout_entry["mean_token_entropy"],
                    "mean_response_length": rollout_entry["mean_response_length"],
                }
                history["train"].append(train_entry)
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "optimizer_step": optimizer_step,
                            **_prefix_metrics("train", train_entry),
                        }
                    )

        if mean_clip_fraction_values:
            history["rollouts"][-1]["mean_clip_fraction"] = (
                sum(mean_clip_fraction_values) / len(mean_clip_fraction_values)
            )

        if eval_llm is not None and eval_every_steps > 0 and grpo_step % eval_every_steps == 0:
            eval_metrics = _evaluate_policy(
                policy_model=policy_model,
                tokenizer=tokenizer,
                llm=eval_llm,
                validation_examples=validation_examples,
                prompt_template=validation_prompt_template,
                reward_fn=validation_reward_fn,
                sampling_params=eval_sampling_params,
                score_device=train_device,
            )
            eval_metrics["step"] = grpo_step
            history["eval"].append(eval_metrics)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "eval_step": grpo_step,
                        **_prefix_metrics("eval", _extract_numeric_metrics(eval_metrics)),
                    }
                )

            if eval_metrics["avg_answer_reward"] > best_eval_answer_reward:
                best_eval_answer_reward = eval_metrics["avg_answer_reward"]
                best_eval = eval_metrics
                if save_best_checkpoint:
                    best_checkpoint_dir = output_dir / "best_checkpoint"
                    best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    policy_model.save_pretrained(best_checkpoint_dir)
                    tokenizer.save_pretrained(best_checkpoint_dir)

    final_eval = None
    if eval_llm is not None and run_final_full_eval and final_validation_examples:
        final_eval = _evaluate_policy(
            policy_model=policy_model,
            tokenizer=tokenizer,
            llm=eval_llm,
            validation_examples=final_validation_examples,
            prompt_template=validation_prompt_template,
            reward_fn=validation_reward_fn,
            sampling_params=eval_sampling_params,
            score_device=train_device,
        )
        final_eval["step"] = n_grpo_steps
        history["final_eval"] = final_eval
        if wandb_run is not None:
            wandb_run.log(
                {
                    "eval_step": n_grpo_steps,
                    **_prefix_metrics("eval", _extract_numeric_metrics(final_eval)),
                }
            )

        if final_eval["avg_answer_reward"] > best_eval_answer_reward:
            best_eval_answer_reward = final_eval["avg_answer_reward"]
            best_eval = final_eval
            if save_best_checkpoint:
                best_checkpoint_dir = output_dir / "best_checkpoint"
                best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                policy_model.save_pretrained(best_checkpoint_dir)
                tokenizer.save_pretrained(best_checkpoint_dir)

    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(_jsonify(history), indent=2, ensure_ascii=True))

    checkpoint_dir = None
    if save_checkpoint:
        checkpoint_dir = output_dir / "final_checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        policy_model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

    summary = {
        "config": history["config"],
        "history_path": str(history_path),
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
        "best_checkpoint_dir": str(best_checkpoint_dir) if best_checkpoint_dir is not None else None,
        "best_eval": _jsonify(best_eval),
        "final_eval": _jsonify(final_eval),
        "num_train_examples_pool": len(train_examples),
        "num_rollout_steps": n_grpo_steps,
        "num_optimizer_steps": optimizer_step,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True))

    if wandb_run is not None:
        wandb_run.summary["summary_path"] = str(summary_path)
        wandb_run.summary["history_path"] = str(history_path)
        wandb_run.summary["num_train_examples_pool"] = len(train_examples)
        wandb_run.summary["num_rollout_steps"] = n_grpo_steps
        wandb_run.summary["num_optimizer_steps"] = optimizer_step
        if best_eval is not None:
            wandb_run.summary["best_avg_answer_reward"] = best_eval["avg_answer_reward"]
        if final_eval is not None:
            wandb_run.summary["final_avg_answer_reward"] = final_eval["avg_answer_reward"]
        if best_checkpoint_dir is not None:
            wandb_run.summary["best_checkpoint_dir"] = str(best_checkpoint_dir)
        if checkpoint_dir is not None:
            wandb_run.summary["final_checkpoint_dir"] = str(checkpoint_dir)
        wandb_run.finish()

    if eval_llm is not None and eval_llm is not rollout_llm:
        del eval_llm
    del rollout_llm
    del policy_model
    del tokenizer
    _cleanup_memory()

    return summary


def _result_score(result: dict[str, Any]) -> float:
    if result.get("final_eval") is not None:
        return float(result["final_eval"]["avg_answer_reward"])
    if result.get("best_eval") is not None:
        return float(result["best_eval"]["avg_answer_reward"])
    return float("-inf")


def _adjust_gradient_accumulation_steps(
    *,
    base_train_batch_size: int,
    base_gradient_accumulation_steps: int,
    train_batch_size: int,
) -> int:
    microbatch_size = base_train_batch_size // base_gradient_accumulation_steps
    if train_batch_size % microbatch_size != 0:
        raise ValueError("train_batch_size must preserve the base microbatch size")
    return train_batch_size // microbatch_size


def _run_named_sweep(
    *,
    output_dir: str | Path,
    run_specs: list[tuple[str, dict[str, Any]]],
    base_config: dict[str, Any],
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_results: dict[str, Any] = {}
    best_run_name: str | None = None
    best_score = float("-inf")

    for run_name, overrides in run_specs:
        run_output_dir = output_dir / run_name
        run_config = dict(base_config)
        run_config.update(overrides)
        run_config["output_dir"] = run_output_dir
        if run_config.get("wandb_run_name") is not None:
            run_config["wandb_run_name"] = f"{run_config['wandb_run_name']}-{run_name}"

        result = run_grpo_experiment(**run_config)
        score = _result_score(result)
        sweep_results[run_name] = {
            "config": _jsonify(run_config),
            "summary_path": str(run_output_dir / "summary.json"),
            "history_path": result["history_path"],
            "checkpoint_dir": result["checkpoint_dir"],
            "best_checkpoint_dir": result["best_checkpoint_dir"],
            "best_eval": result["best_eval"],
            "final_eval": result["final_eval"],
            "score": score,
        }
        if score > best_score:
            best_score = score
            best_run_name = run_name

    summary = {
        "output_dir": str(output_dir),
        "runs": sweep_results,
        "best_run_name": best_run_name,
        "best_run": sweep_results.get(best_run_name) if best_run_name is not None else None,
    }
    summary_path = output_dir / "sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True))
    summary["summary_path"] = str(summary_path)
    return summary


def run_grpo_learning_rate_sweep(
    *,
    learning_rates: tuple[float, ...] = (3e-6, 1e-5, 3e-5),
    output_dir: str | Path | None = None,
    **experiment_kwargs: Any,
) -> dict[str, Any]:
    root = _repo_root()
    output_dir = Path(output_dir or root / "results" / "GRPO" / "learning_rate")
    run_specs = [
        (f"lr_{learning_rate:.0e}".replace("-", "m"), {"learning_rate": learning_rate})
        for learning_rate in learning_rates
    ]
    return _run_named_sweep(output_dir=output_dir, run_specs=run_specs, base_config=experiment_kwargs)


def run_grpo_baseline_ablation(
    *,
    loss_types: tuple[LossType, ...] = ("no_baseline", "reinforce_with_baseline"),
    output_dir: str | Path | None = None,
    **experiment_kwargs: Any,
) -> dict[str, Any]:
    root = _repo_root()
    output_dir = Path(output_dir or root / "results" / "GRPO" / "baselines")
    run_specs = [(str(loss_type), {"loss_type": loss_type}) for loss_type in loss_types]
    return _run_named_sweep(output_dir=output_dir, run_specs=run_specs, base_config=experiment_kwargs)


def run_grpo_length_normalization_ablation(
    *,
    normalize_modes: tuple[NormalizeMode, ...] = ("masked_mean", "masked_normalize"),
    output_dir: str | Path | None = None,
    **experiment_kwargs: Any,
) -> dict[str, Any]:
    root = _repo_root()
    output_dir = Path(output_dir or root / "results" / "GRPO" / "length_normalization")
    run_specs = [
        (str(normalize_mode), {"normalize_loss_by": normalize_mode})
        for normalize_mode in normalize_modes
    ]
    return _run_named_sweep(output_dir=output_dir, run_specs=run_specs, base_config=experiment_kwargs)


def run_grpo_std_normalization_ablation(
    *,
    output_dir: str | Path | None = None,
    **experiment_kwargs: Any,
) -> dict[str, Any]:
    root = _repo_root()
    output_dir = Path(output_dir or root / "results" / "GRPO" / "std_normalization")
    run_specs = [
        ("std_on", {"use_std_normalization": True}),
        ("std_off", {"use_std_normalization": False}),
    ]
    return _run_named_sweep(output_dir=output_dir, run_specs=run_specs, base_config=experiment_kwargs)


def run_grpo_off_policy_sweep(
    *,
    configs: tuple[tuple[int, int], ...] = ((1, 256), (2, 256), (4, 256), (2, 128), (4, 128)),
    output_dir: str | Path | None = None,
    adjust_gradient_accumulation: bool = True,
    **experiment_kwargs: Any,
) -> dict[str, Any]:
    root = _repo_root()
    output_dir = Path(output_dir or root / "results" / "GRPO" / "off_policy")

    base_train_batch_size = int(experiment_kwargs["train_batch_size"])
    base_gradient_accumulation_steps = int(experiment_kwargs["gradient_accumulation_steps"])
    run_specs: list[tuple[str, dict[str, Any]]] = []
    for epochs_per_rollout_batch, train_batch_size in configs:
        overrides: dict[str, Any] = {
            "epochs_per_rollout_batch": epochs_per_rollout_batch,
            "train_batch_size": train_batch_size,
            "loss_type": "grpo_clip",
        }
        if adjust_gradient_accumulation:
            overrides["gradient_accumulation_steps"] = _adjust_gradient_accumulation_steps(
                base_train_batch_size=base_train_batch_size,
                base_gradient_accumulation_steps=base_gradient_accumulation_steps,
                train_batch_size=train_batch_size,
            )
        run_specs.append(
            (
                f"epochs_{epochs_per_rollout_batch}_batch_{train_batch_size}",
                overrides,
            )
        )

    return _run_named_sweep(output_dir=output_dir, run_specs=run_specs, base_config=experiment_kwargs)


def run_grpo_off_policy_clip_ablation(
    *,
    output_dir: str | Path | None = None,
    **experiment_kwargs: Any,
) -> dict[str, Any]:
    root = _repo_root()
    output_dir = Path(output_dir or root / "results" / "GRPO" / "off_policy_clip_ablation")
    run_specs = [
        ("grpo_clip", {"loss_type": "grpo_clip"}),
        ("grpo_no_clip", {"loss_type": "grpo_no_clip"}),
    ]
    return _run_named_sweep(output_dir=output_dir, run_specs=run_specs, base_config=experiment_kwargs)


def run_grpo_prompt_ablation(
    *,
    output_dir: str | Path | None = None,
    **experiment_kwargs: Any,
) -> dict[str, Any]:
    root = _repo_root()
    output_dir = Path(output_dir or root / "results" / "GRPO" / "prompt_ablation")
    run_specs = [
        (
            "r1_zero",
            {
                "prompt_path": root / "cs336_alignment" / "prompts" / "r1_zero.prompt",
                "validation_prompt_path": root / "cs336_alignment" / "prompts" / "r1_zero.prompt",
                "reward_fn_name": "r1_zero",
                "validation_reward_fn_name": "r1_zero",
            },
        ),
        (
            "question_only",
            {
                "prompt_path": root / "cs336_alignment" / "prompts" / "question_only.prompt",
                "validation_prompt_path": root / "cs336_alignment" / "prompts" / "question_only.prompt",
                "reward_fn_name": "question_only",
                "validation_reward_fn_name": "question_only",
            },
        ),
    ]
    return _run_named_sweep(output_dir=output_dir, run_specs=run_specs, base_config=experiment_kwargs)


def run_all_required_grpo_experiments(
    *,
    learning_rates: tuple[float, ...] = (3e-6, 1e-5, 3e-5),
    off_policy_configs: tuple[tuple[int, int], ...] = (
        (1, 256),
        (2, 256),
        (4, 256),
        (2, 128),
        (4, 128),
    ),
    output_dir: str | Path | None = None,
    **experiment_kwargs: Any,
) -> dict[str, Any]:
    root = _repo_root()
    output_dir = Path(output_dir or root / "results" / "GRPO" / "chapter8_required")
    output_dir.mkdir(parents=True, exist_ok=True)

    learning_rate_result = run_grpo_learning_rate_sweep(
        learning_rates=learning_rates,
        output_dir=output_dir / "learning_rate",
        **experiment_kwargs,
    )
    best_learning_rate = learning_rate_result["best_run"]["config"]["learning_rate"]

    baseline_kwargs = dict(experiment_kwargs)
    baseline_kwargs["learning_rate"] = best_learning_rate
    baseline_result = run_grpo_baseline_ablation(
        output_dir=output_dir / "baselines",
        **baseline_kwargs,
    )
    best_loss_type = baseline_result["best_run"]["config"]["loss_type"]

    length_kwargs = dict(baseline_kwargs)
    length_kwargs["loss_type"] = best_loss_type
    length_result = run_grpo_length_normalization_ablation(
        output_dir=output_dir / "length_normalization",
        **length_kwargs,
    )
    best_normalize_mode = length_result["best_run"]["config"]["normalize_loss_by"]

    std_kwargs = dict(length_kwargs)
    std_kwargs["normalize_loss_by"] = best_normalize_mode
    std_result = run_grpo_std_normalization_ablation(
        output_dir=output_dir / "std_normalization",
        **std_kwargs,
    )
    best_use_std_normalization = std_result["best_run"]["config"]["use_std_normalization"]

    off_policy_kwargs = dict(experiment_kwargs)
    off_policy_kwargs["learning_rate"] = best_learning_rate
    off_policy_kwargs["normalize_loss_by"] = best_normalize_mode
    off_policy_kwargs["use_std_normalization"] = best_use_std_normalization
    off_policy_result = run_grpo_off_policy_sweep(
        configs=off_policy_configs,
        output_dir=output_dir / "off_policy",
        **off_policy_kwargs,
    )
    best_off_policy_config = off_policy_result["best_run"]["config"]

    clip_kwargs = dict(experiment_kwargs)
    clip_kwargs.update(
        {
            "learning_rate": best_learning_rate,
            "normalize_loss_by": best_normalize_mode,
            "use_std_normalization": best_use_std_normalization,
            "epochs_per_rollout_batch": best_off_policy_config["epochs_per_rollout_batch"],
            "train_batch_size": best_off_policy_config["train_batch_size"],
            "gradient_accumulation_steps": best_off_policy_config["gradient_accumulation_steps"],
        }
    )
    clip_ablation_result = run_grpo_off_policy_clip_ablation(
        output_dir=output_dir / "off_policy_clip_ablation",
        **clip_kwargs,
    )

    prompt_kwargs = dict(clip_kwargs)
    prompt_kwargs["loss_type"] = "grpo_clip"
    prompt_result = run_grpo_prompt_ablation(
        output_dir=output_dir / "prompt_ablation",
        **prompt_kwargs,
    )

    summary = {
        "output_dir": str(output_dir),
        "stages": {
            "learning_rate": learning_rate_result,
            "baselines": baseline_result,
            "length_normalization": length_result,
            "std_normalization": std_result,
            "off_policy": off_policy_result,
            "off_policy_clip_ablation": clip_ablation_result,
            "prompt_ablation": prompt_result,
        },
        "selected_config": {
            "learning_rate": best_learning_rate,
            "loss_type": best_loss_type,
            "normalize_loss_by": best_normalize_mode,
            "use_std_normalization": best_use_std_normalization,
            "off_policy_epochs_per_rollout_batch": best_off_policy_config["epochs_per_rollout_batch"],
            "off_policy_train_batch_size": best_off_policy_config["train_batch_size"],
            "off_policy_gradient_accumulation_steps": best_off_policy_config[
                "gradient_accumulation_steps"
            ],
        },
    }
    summary_path = output_dir / "chapter8_summary.json"
    summary_path.write_text(json.dumps(_jsonify(summary), indent=2, ensure_ascii=True))
    summary["summary_path"] = str(summary_path)
    return summary


def run_grpo_train_loop(*args, **kwargs):
    """Backward-compatible programmatic entry point for one GRPO run."""
    return run_grpo_experiment(*args, **kwargs)


def _parse_float_csv(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _parse_off_policy_configs(value: str) -> tuple[tuple[int, int], ...]:
    configs: list[tuple[int, int]] = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        epochs_value, batch_value = stripped.split("x", maxsplit=1)
        configs.append((int(epochs_value), int(batch_value)))
    return tuple(configs)


def _base_sweep_overrides(
    *,
    profile: str,
    model_path: Path | None,
    train_dataset_path: Path | None,
    validation_dataset_path: Path | None,
    prompt_path: Path | None,
    validation_prompt_path: Path | None,
    train_device: str | None,
    rollout_device: str | None,
    eval_device: str | None,
    n_grpo_steps: int | None,
    eval_num_examples: int | None,
    eval_every_steps: int | None,
    final_eval_num_examples: int | None,
    use_wandb: bool | None,
    wandb_project: str | None,
    wandb_run_name: str | None,
    wandb_mode: str | None,
    torch_dtype: str | None,
    attn_implementation: str | None,
) -> dict[str, Any]:
    return _merge_profile_overrides(
        profile,
        {
            "model_path": model_path,
            "train_dataset_path": train_dataset_path,
            "validation_dataset_path": validation_dataset_path,
            "prompt_path": prompt_path,
            "validation_prompt_path": validation_prompt_path,
            "train_device": train_device,
            "rollout_device": rollout_device,
            "eval_device": _optional_device_override(eval_device),
            "n_grpo_steps": n_grpo_steps,
            "eval_num_examples": eval_num_examples,
            "eval_every_steps": eval_every_steps,
            "final_eval_num_examples": final_eval_num_examples,
            "use_wandb": use_wandb,
            "wandb_project": wandb_project,
            "wandb_run_name": wandb_run_name,
            "wandb_mode": wandb_mode,
            "torch_dtype": _parse_torch_dtype(torch_dtype) if torch_dtype is not None else None,
            "attn_implementation": attn_implementation,
        },
    )


@app.command("single")
def cli_run_single(
    profile: str = typer.Option("local-smoke", help="Preset to start from: local-smoke or cloud."),
    model_path: Path | None = typer.Option(None, help="Override the model path."),
    train_dataset_path: Path | None = typer.Option(None, help="Override the GRPO train dataset path."),
    validation_dataset_path: Path | None = typer.Option(
        None,
        help="Override the validation dataset path.",
    ),
    prompt_path: Path | None = typer.Option(None, help="Override the rollout prompt template path."),
    validation_prompt_path: Path | None = typer.Option(
        None,
        help="Override the validation prompt template path.",
    ),
    output_dir: Path | None = typer.Option(None, help="Directory where GRPO logs are written."),
    train_device: str | None = typer.Option(None, help="Training device, e.g. cuda:0."),
    rollout_device: str | None = typer.Option(None, help="Rollout device, e.g. cuda:1."),
    eval_device: str | None = typer.Option(None, help="Evaluation device or none to disable."),
    learning_rate: float | None = typer.Option(None, help="AdamW learning rate."),
    weight_decay: float | None = typer.Option(None, help="AdamW weight decay."),
    adam_beta1: float | None = typer.Option(None, help="AdamW beta1."),
    adam_beta2: float | None = typer.Option(None, help="AdamW beta2."),
    n_grpo_steps: int | None = typer.Option(None, help="Number of rollout batches to train for."),
    advantage_eps: float | None = typer.Option(None, help="Small constant used in reward normalization."),
    rollout_batch_size: int | None = typer.Option(None, help="Number of rollout responses per GRPO step."),
    group_size: int | None = typer.Option(None, help="Number of responses sampled per prompt."),
    sampling_temperature: float | None = typer.Option(None, help="Sampling temperature."),
    sampling_min_tokens: int | None = typer.Option(None, help="Minimum rollout length."),
    sampling_max_tokens: int | None = typer.Option(None, help="Maximum rollout length."),
    epochs_per_rollout_batch: int | None = typer.Option(None, help="Policy epochs per rollout batch."),
    train_batch_size: int | None = typer.Option(None, help="Effective policy batch size."),
    gradient_accumulation_steps: int | None = typer.Option(
        None,
        help="Microbatches per optimizer step.",
    ),
    gpu_memory_utilization: float | None = typer.Option(None, help="vLLM gpu_memory_utilization."),
    loss_type: str | None = typer.Option(None, help="Policy-gradient loss type."),
    normalize_loss_by: str | None = typer.Option(
        None,
        help="masked_mean or masked_normalize.",
    ),
    loss_normalize_constant: float | None = typer.Option(
        None,
        help="Constant used by masked_normalize.",
    ),
    use_std_normalization: bool | None = typer.Option(
        None,
        "--use-std-normalization/--no-std-normalization",
        help="Normalize group advantages by per-group std.",
    ),
    cliprange: float | None = typer.Option(None, help="Clip parameter for GRPO-Clip."),
    max_grad_norm: float | None = typer.Option(None, help="Gradient clipping threshold."),
    max_train_examples: int | None = typer.Option(None, help="Cap the training example pool."),
    eval_num_examples: int | None = typer.Option(None, help="Validation subset size."),
    eval_every_steps: int | None = typer.Option(None, help="Run validation every N rollout steps."),
    final_eval_num_examples: int | None = typer.Option(
        None,
        help="Validation subset size for final evaluation.",
    ),
    run_final_full_eval: bool | None = typer.Option(
        None,
        "--run-final-full-eval/--skip-final-full-eval",
        help="Run a final validation pass after training.",
    ),
    save_rollouts: bool | None = typer.Option(
        None,
        "--save-rollouts/--skip-save-rollouts",
        help="Persist rollout samples to disk.",
    ),
    save_checkpoint: bool | None = typer.Option(
        None,
        "--save-checkpoint/--skip-save-checkpoint",
        help="Save the final checkpoint.",
    ),
    save_best_checkpoint: bool | None = typer.Option(
        None,
        "--save-best-checkpoint/--skip-save-best-checkpoint",
        help="Save the best checkpoint based on validation answer reward.",
    ),
    reward_fn_name: str | None = typer.Option(None, help="Training reward function name."),
    validation_reward_fn_name: str | None = typer.Option(
        None,
        help="Validation reward function name.",
    ),
    seed: int | None = typer.Option(None, help="Random seed."),
    shuffle: bool | None = typer.Option(None, "--shuffle/--no-shuffle", help="Shuffle training batches."),
    torch_dtype: str | None = typer.Option(None, help="float16, bfloat16, or float32."),
    attn_implementation: str | None = typer.Option(
        None,
        help="Attention implementation, e.g. sdpa, eager, flash_attention_2.",
    ),
    use_wandb: bool | None = typer.Option(None, "--use-wandb/--no-wandb", help="Enable wandb."),
    wandb_project: str | None = typer.Option(None, help="wandb project name."),
    wandb_run_name: str | None = typer.Option(None, help="wandb run name."),
    wandb_mode: str | None = typer.Option(None, help="wandb mode, e.g. online, offline."),
) -> None:
    overrides = {
        "model_path": model_path,
        "train_dataset_path": train_dataset_path,
        "validation_dataset_path": validation_dataset_path,
        "prompt_path": prompt_path,
        "validation_prompt_path": validation_prompt_path,
        "output_dir": output_dir,
        "train_device": train_device,
        "rollout_device": rollout_device,
        "eval_device": _optional_device_override(eval_device),
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "adam_beta1": adam_beta1,
        "adam_beta2": adam_beta2,
        "n_grpo_steps": n_grpo_steps,
        "advantage_eps": advantage_eps,
        "rollout_batch_size": rollout_batch_size,
        "group_size": group_size,
        "sampling_temperature": sampling_temperature,
        "sampling_min_tokens": sampling_min_tokens,
        "sampling_max_tokens": sampling_max_tokens,
        "epochs_per_rollout_batch": epochs_per_rollout_batch,
        "train_batch_size": train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gpu_memory_utilization": gpu_memory_utilization,
        "loss_type": loss_type,
        "normalize_loss_by": normalize_loss_by,
        "loss_normalize_constant": loss_normalize_constant,
        "use_std_normalization": use_std_normalization,
        "cliprange": cliprange,
        "max_grad_norm": max_grad_norm,
        "max_train_examples": max_train_examples,
        "eval_num_examples": eval_num_examples,
        "eval_every_steps": eval_every_steps,
        "final_eval_num_examples": final_eval_num_examples,
        "run_final_full_eval": run_final_full_eval,
        "save_rollouts": save_rollouts,
        "save_checkpoint": save_checkpoint,
        "save_best_checkpoint": save_best_checkpoint,
        "reward_fn_name": reward_fn_name,
        "validation_reward_fn_name": validation_reward_fn_name,
        "seed": seed,
        "shuffle": _cli_bool_override("--shuffle", "--no-shuffle"),
        "torch_dtype": _parse_torch_dtype(torch_dtype) if torch_dtype is not None else None,
        "attn_implementation": attn_implementation,
        "use_wandb": _cli_bool_override("--use-wandb", "--no-wandb"),
        "wandb_project": wandb_project,
        "wandb_run_name": wandb_run_name,
        "wandb_mode": wandb_mode,
    }
    config = _merge_profile_overrides(profile, overrides)
    result = run_grpo_experiment(**config)
    typer.echo(json.dumps(_jsonify(result), indent=2, ensure_ascii=True))


@app.command("learning-rate")
def cli_run_learning_rate(
    profile: str = typer.Option("cloud", help="Preset to start from: local-smoke or cloud."),
    learning_rates: str = typer.Option("3e-6,1e-5,3e-5", help="Comma-separated learning rates."),
    output_dir: Path | None = typer.Option(None, help="Directory where sweep results are written."),
    model_path: Path | None = typer.Option(None, help="Override the model path."),
    train_dataset_path: Path | None = typer.Option(None, help="Override the GRPO train dataset path."),
    validation_dataset_path: Path | None = typer.Option(None, help="Override the validation dataset path."),
    prompt_path: Path | None = typer.Option(None, help="Override the rollout prompt template path."),
    validation_prompt_path: Path | None = typer.Option(None, help="Override the validation prompt template path."),
    train_device: str | None = typer.Option(None, help="Training device."),
    rollout_device: str | None = typer.Option(None, help="Rollout device."),
    eval_device: str | None = typer.Option(None, help="Evaluation device or none."),
    n_grpo_steps: int | None = typer.Option(None, help="Number of GRPO steps."),
    eval_num_examples: int | None = typer.Option(None, help="Validation subset size."),
    eval_every_steps: int | None = typer.Option(None, help="Evaluate every N steps."),
    final_eval_num_examples: int | None = typer.Option(None, help="Final evaluation subset size."),
    use_wandb: bool | None = typer.Option(None, "--use-wandb/--no-wandb", help="Enable wandb."),
    wandb_project: str | None = typer.Option(None, help="wandb project name."),
    wandb_run_name: str | None = typer.Option(None, help="wandb run name prefix."),
    wandb_mode: str | None = typer.Option(None, help="wandb mode."),
    torch_dtype: str | None = typer.Option(None, help="float16, bfloat16, or float32."),
    attn_implementation: str | None = typer.Option(None, help="Attention implementation."),
) -> None:
    config = _base_sweep_overrides(
        profile=profile,
        model_path=model_path,
        train_dataset_path=train_dataset_path,
        validation_dataset_path=validation_dataset_path,
        prompt_path=prompt_path,
        validation_prompt_path=validation_prompt_path,
        train_device=train_device,
        rollout_device=rollout_device,
        eval_device=eval_device,
        n_grpo_steps=n_grpo_steps,
        eval_num_examples=eval_num_examples,
        eval_every_steps=eval_every_steps,
        final_eval_num_examples=final_eval_num_examples,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_mode=wandb_mode,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )
    result = run_grpo_learning_rate_sweep(
        learning_rates=_parse_float_csv(learning_rates),
        output_dir=output_dir,
        **config,
    )
    typer.echo(json.dumps(_jsonify(result), indent=2, ensure_ascii=True))


@app.command("baselines")
def cli_run_baselines(
    profile: str = typer.Option("cloud", help="Preset to start from: local-smoke or cloud."),
    output_dir: Path | None = typer.Option(None, help="Directory where sweep results are written."),
    model_path: Path | None = typer.Option(None, help="Override the model path."),
    train_dataset_path: Path | None = typer.Option(None, help="Override the GRPO train dataset path."),
    validation_dataset_path: Path | None = typer.Option(None, help="Override the validation dataset path."),
    prompt_path: Path | None = typer.Option(None, help="Override the rollout prompt template path."),
    validation_prompt_path: Path | None = typer.Option(None, help="Override the validation prompt template path."),
    train_device: str | None = typer.Option(None, help="Training device."),
    rollout_device: str | None = typer.Option(None, help="Rollout device."),
    eval_device: str | None = typer.Option(None, help="Evaluation device or none."),
    n_grpo_steps: int | None = typer.Option(None, help="Number of GRPO steps."),
    eval_num_examples: int | None = typer.Option(None, help="Validation subset size."),
    eval_every_steps: int | None = typer.Option(None, help="Evaluate every N steps."),
    final_eval_num_examples: int | None = typer.Option(None, help="Final evaluation subset size."),
    use_wandb: bool | None = typer.Option(None, "--use-wandb/--no-wandb", help="Enable wandb."),
    wandb_project: str | None = typer.Option(None, help="wandb project name."),
    wandb_run_name: str | None = typer.Option(None, help="wandb run name prefix."),
    wandb_mode: str | None = typer.Option(None, help="wandb mode."),
    torch_dtype: str | None = typer.Option(None, help="float16, bfloat16, or float32."),
    attn_implementation: str | None = typer.Option(None, help="Attention implementation."),
) -> None:
    config = _base_sweep_overrides(
        profile=profile,
        model_path=model_path,
        train_dataset_path=train_dataset_path,
        validation_dataset_path=validation_dataset_path,
        prompt_path=prompt_path,
        validation_prompt_path=validation_prompt_path,
        train_device=train_device,
        rollout_device=rollout_device,
        eval_device=eval_device,
        n_grpo_steps=n_grpo_steps,
        eval_num_examples=eval_num_examples,
        eval_every_steps=eval_every_steps,
        final_eval_num_examples=final_eval_num_examples,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_mode=wandb_mode,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )
    result = run_grpo_baseline_ablation(output_dir=output_dir, **config)
    typer.echo(json.dumps(_jsonify(result), indent=2, ensure_ascii=True))


@app.command("length-normalization")
def cli_run_length_normalization(
    profile: str = typer.Option("cloud", help="Preset to start from: local-smoke or cloud."),
    output_dir: Path | None = typer.Option(None, help="Directory where sweep results are written."),
    model_path: Path | None = typer.Option(None, help="Override the model path."),
    train_dataset_path: Path | None = typer.Option(None, help="Override the GRPO train dataset path."),
    validation_dataset_path: Path | None = typer.Option(None, help="Override the validation dataset path."),
    prompt_path: Path | None = typer.Option(None, help="Override the rollout prompt template path."),
    validation_prompt_path: Path | None = typer.Option(None, help="Override the validation prompt template path."),
    train_device: str | None = typer.Option(None, help="Training device."),
    rollout_device: str | None = typer.Option(None, help="Rollout device."),
    eval_device: str | None = typer.Option(None, help="Evaluation device or none."),
    n_grpo_steps: int | None = typer.Option(None, help="Number of GRPO steps."),
    eval_num_examples: int | None = typer.Option(None, help="Validation subset size."),
    eval_every_steps: int | None = typer.Option(None, help="Evaluate every N steps."),
    final_eval_num_examples: int | None = typer.Option(None, help="Final evaluation subset size."),
    use_wandb: bool | None = typer.Option(None, "--use-wandb/--no-wandb", help="Enable wandb."),
    wandb_project: str | None = typer.Option(None, help="wandb project name."),
    wandb_run_name: str | None = typer.Option(None, help="wandb run name prefix."),
    wandb_mode: str | None = typer.Option(None, help="wandb mode."),
    torch_dtype: str | None = typer.Option(None, help="float16, bfloat16, or float32."),
    attn_implementation: str | None = typer.Option(None, help="Attention implementation."),
) -> None:
    config = _base_sweep_overrides(
        profile=profile,
        model_path=model_path,
        train_dataset_path=train_dataset_path,
        validation_dataset_path=validation_dataset_path,
        prompt_path=prompt_path,
        validation_prompt_path=validation_prompt_path,
        train_device=train_device,
        rollout_device=rollout_device,
        eval_device=eval_device,
        n_grpo_steps=n_grpo_steps,
        eval_num_examples=eval_num_examples,
        eval_every_steps=eval_every_steps,
        final_eval_num_examples=final_eval_num_examples,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_mode=wandb_mode,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )
    result = run_grpo_length_normalization_ablation(output_dir=output_dir, **config)
    typer.echo(json.dumps(_jsonify(result), indent=2, ensure_ascii=True))


@app.command("std-normalization")
def cli_run_std_normalization(
    profile: str = typer.Option("cloud", help="Preset to start from: local-smoke or cloud."),
    output_dir: Path | None = typer.Option(None, help="Directory where sweep results are written."),
    model_path: Path | None = typer.Option(None, help="Override the model path."),
    train_dataset_path: Path | None = typer.Option(None, help="Override the GRPO train dataset path."),
    validation_dataset_path: Path | None = typer.Option(None, help="Override the validation dataset path."),
    prompt_path: Path | None = typer.Option(None, help="Override the rollout prompt template path."),
    validation_prompt_path: Path | None = typer.Option(None, help="Override the validation prompt template path."),
    train_device: str | None = typer.Option(None, help="Training device."),
    rollout_device: str | None = typer.Option(None, help="Rollout device."),
    eval_device: str | None = typer.Option(None, help="Evaluation device or none."),
    n_grpo_steps: int | None = typer.Option(None, help="Number of GRPO steps."),
    eval_num_examples: int | None = typer.Option(None, help="Validation subset size."),
    eval_every_steps: int | None = typer.Option(None, help="Evaluate every N steps."),
    final_eval_num_examples: int | None = typer.Option(None, help="Final evaluation subset size."),
    use_wandb: bool | None = typer.Option(None, "--use-wandb/--no-wandb", help="Enable wandb."),
    wandb_project: str | None = typer.Option(None, help="wandb project name."),
    wandb_run_name: str | None = typer.Option(None, help="wandb run name prefix."),
    wandb_mode: str | None = typer.Option(None, help="wandb mode."),
    torch_dtype: str | None = typer.Option(None, help="float16, bfloat16, or float32."),
    attn_implementation: str | None = typer.Option(None, help="Attention implementation."),
) -> None:
    config = _base_sweep_overrides(
        profile=profile,
        model_path=model_path,
        train_dataset_path=train_dataset_path,
        validation_dataset_path=validation_dataset_path,
        prompt_path=prompt_path,
        validation_prompt_path=validation_prompt_path,
        train_device=train_device,
        rollout_device=rollout_device,
        eval_device=eval_device,
        n_grpo_steps=n_grpo_steps,
        eval_num_examples=eval_num_examples,
        eval_every_steps=eval_every_steps,
        final_eval_num_examples=final_eval_num_examples,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_mode=wandb_mode,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )
    result = run_grpo_std_normalization_ablation(output_dir=output_dir, **config)
    typer.echo(json.dumps(_jsonify(result), indent=2, ensure_ascii=True))


@app.command("off-policy-sweep")
def cli_run_off_policy_sweep(
    profile: str = typer.Option("cloud", help="Preset to start from: local-smoke or cloud."),
    configs: str = typer.Option(
        "1x256,2x256,4x256,2x128,4x128",
        help="Comma-separated epochxbatch specs, e.g. 1x256,2x256,4x128.",
    ),
    output_dir: Path | None = typer.Option(None, help="Directory where sweep results are written."),
    model_path: Path | None = typer.Option(None, help="Override the model path."),
    train_dataset_path: Path | None = typer.Option(None, help="Override the GRPO train dataset path."),
    validation_dataset_path: Path | None = typer.Option(None, help="Override the validation dataset path."),
    prompt_path: Path | None = typer.Option(None, help="Override the rollout prompt template path."),
    validation_prompt_path: Path | None = typer.Option(None, help="Override the validation prompt template path."),
    train_device: str | None = typer.Option(None, help="Training device."),
    rollout_device: str | None = typer.Option(None, help="Rollout device."),
    eval_device: str | None = typer.Option(None, help="Evaluation device or none."),
    n_grpo_steps: int | None = typer.Option(None, help="Number of GRPO steps."),
    eval_num_examples: int | None = typer.Option(None, help="Validation subset size."),
    eval_every_steps: int | None = typer.Option(None, help="Evaluate every N steps."),
    final_eval_num_examples: int | None = typer.Option(None, help="Final evaluation subset size."),
    use_wandb: bool | None = typer.Option(None, "--use-wandb/--no-wandb", help="Enable wandb."),
    wandb_project: str | None = typer.Option(None, help="wandb project name."),
    wandb_run_name: str | None = typer.Option(None, help="wandb run name prefix."),
    wandb_mode: str | None = typer.Option(None, help="wandb mode."),
    torch_dtype: str | None = typer.Option(None, help="float16, bfloat16, or float32."),
    attn_implementation: str | None = typer.Option(None, help="Attention implementation."),
) -> None:
    config = _base_sweep_overrides(
        profile=profile,
        model_path=model_path,
        train_dataset_path=train_dataset_path,
        validation_dataset_path=validation_dataset_path,
        prompt_path=prompt_path,
        validation_prompt_path=validation_prompt_path,
        train_device=train_device,
        rollout_device=rollout_device,
        eval_device=eval_device,
        n_grpo_steps=n_grpo_steps,
        eval_num_examples=eval_num_examples,
        eval_every_steps=eval_every_steps,
        final_eval_num_examples=final_eval_num_examples,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_mode=wandb_mode,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )
    result = run_grpo_off_policy_sweep(
        configs=_parse_off_policy_configs(configs),
        output_dir=output_dir,
        **config,
    )
    typer.echo(json.dumps(_jsonify(result), indent=2, ensure_ascii=True))


@app.command("off-policy-no-clip")
def cli_run_off_policy_no_clip(
    profile: str = typer.Option("cloud", help="Preset to start from: local-smoke or cloud."),
    output_dir: Path | None = typer.Option(None, help="Directory where sweep results are written."),
    model_path: Path | None = typer.Option(None, help="Override the model path."),
    train_dataset_path: Path | None = typer.Option(None, help="Override the GRPO train dataset path."),
    validation_dataset_path: Path | None = typer.Option(None, help="Override the validation dataset path."),
    prompt_path: Path | None = typer.Option(None, help="Override the rollout prompt template path."),
    validation_prompt_path: Path | None = typer.Option(None, help="Override the validation prompt template path."),
    train_device: str | None = typer.Option(None, help="Training device."),
    rollout_device: str | None = typer.Option(None, help="Rollout device."),
    eval_device: str | None = typer.Option(None, help="Evaluation device or none."),
    n_grpo_steps: int | None = typer.Option(None, help="Number of GRPO steps."),
    eval_num_examples: int | None = typer.Option(None, help="Validation subset size."),
    eval_every_steps: int | None = typer.Option(None, help="Evaluate every N steps."),
    final_eval_num_examples: int | None = typer.Option(None, help="Final evaluation subset size."),
    use_wandb: bool | None = typer.Option(None, "--use-wandb/--no-wandb", help="Enable wandb."),
    wandb_project: str | None = typer.Option(None, help="wandb project name."),
    wandb_run_name: str | None = typer.Option(None, help="wandb run name prefix."),
    wandb_mode: str | None = typer.Option(None, help="wandb mode."),
    torch_dtype: str | None = typer.Option(None, help="float16, bfloat16, or float32."),
    attn_implementation: str | None = typer.Option(None, help="Attention implementation."),
) -> None:
    config = _base_sweep_overrides(
        profile=profile,
        model_path=model_path,
        train_dataset_path=train_dataset_path,
        validation_dataset_path=validation_dataset_path,
        prompt_path=prompt_path,
        validation_prompt_path=validation_prompt_path,
        train_device=train_device,
        rollout_device=rollout_device,
        eval_device=eval_device,
        n_grpo_steps=n_grpo_steps,
        eval_num_examples=eval_num_examples,
        eval_every_steps=eval_every_steps,
        final_eval_num_examples=final_eval_num_examples,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_mode=wandb_mode,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )
    result = run_grpo_off_policy_clip_ablation(output_dir=output_dir, **config)
    typer.echo(json.dumps(_jsonify(result), indent=2, ensure_ascii=True))


@app.command("prompt-ablation")
def cli_run_prompt_ablation(
    profile: str = typer.Option("cloud", help="Preset to start from: local-smoke or cloud."),
    output_dir: Path | None = typer.Option(None, help="Directory where sweep results are written."),
    model_path: Path | None = typer.Option(None, help="Override the model path."),
    train_dataset_path: Path | None = typer.Option(None, help="Override the GRPO train dataset path."),
    validation_dataset_path: Path | None = typer.Option(None, help="Override the validation dataset path."),
    train_device: str | None = typer.Option(None, help="Training device."),
    rollout_device: str | None = typer.Option(None, help="Rollout device."),
    eval_device: str | None = typer.Option(None, help="Evaluation device or none."),
    n_grpo_steps: int | None = typer.Option(None, help="Number of GRPO steps."),
    eval_num_examples: int | None = typer.Option(None, help="Validation subset size."),
    eval_every_steps: int | None = typer.Option(None, help="Evaluate every N steps."),
    final_eval_num_examples: int | None = typer.Option(None, help="Final evaluation subset size."),
    use_wandb: bool | None = typer.Option(None, "--use-wandb/--no-wandb", help="Enable wandb."),
    wandb_project: str | None = typer.Option(None, help="wandb project name."),
    wandb_run_name: str | None = typer.Option(None, help="wandb run name prefix."),
    wandb_mode: str | None = typer.Option(None, help="wandb mode."),
    torch_dtype: str | None = typer.Option(None, help="float16, bfloat16, or float32."),
    attn_implementation: str | None = typer.Option(None, help="Attention implementation."),
) -> None:
    config = _base_sweep_overrides(
        profile=profile,
        model_path=model_path,
        train_dataset_path=train_dataset_path,
        validation_dataset_path=validation_dataset_path,
        prompt_path=None,
        validation_prompt_path=None,
        train_device=train_device,
        rollout_device=rollout_device,
        eval_device=eval_device,
        n_grpo_steps=n_grpo_steps,
        eval_num_examples=eval_num_examples,
        eval_every_steps=eval_every_steps,
        final_eval_num_examples=final_eval_num_examples,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_mode=wandb_mode,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )
    result = run_grpo_prompt_ablation(output_dir=output_dir, **config)
    typer.echo(json.dumps(_jsonify(result), indent=2, ensure_ascii=True))


@app.command("all-required")
def cli_run_all_required(
    profile: str = typer.Option("cloud", help="Preset to start from: local-smoke or cloud."),
    learning_rates: str = typer.Option("3e-6,1e-5,3e-5", help="Comma-separated learning rates."),
    off_policy_configs: str = typer.Option(
        "1x256,2x256,4x256,2x128,4x128",
        help="Comma-separated epochxbatch specs, e.g. 1x256,2x256,4x128.",
    ),
    output_dir: Path | None = typer.Option(None, help="Directory where the full Chapter 8 bundle is written."),
    model_path: Path | None = typer.Option(None, help="Override the model path."),
    train_dataset_path: Path | None = typer.Option(None, help="Override the GRPO train dataset path."),
    validation_dataset_path: Path | None = typer.Option(None, help="Override the validation dataset path."),
    prompt_path: Path | None = typer.Option(None, help="Override the rollout prompt template path."),
    validation_prompt_path: Path | None = typer.Option(None, help="Override the validation prompt template path."),
    train_device: str | None = typer.Option(None, help="Training device."),
    rollout_device: str | None = typer.Option(None, help="Rollout device."),
    eval_device: str | None = typer.Option(None, help="Evaluation device or none."),
    n_grpo_steps: int | None = typer.Option(None, help="Number of GRPO steps."),
    eval_num_examples: int | None = typer.Option(None, help="Validation subset size."),
    eval_every_steps: int | None = typer.Option(None, help="Evaluate every N steps."),
    final_eval_num_examples: int | None = typer.Option(None, help="Final evaluation subset size."),
    use_wandb: bool | None = typer.Option(None, "--use-wandb/--no-wandb", help="Enable wandb."),
    wandb_project: str | None = typer.Option(None, help="wandb project name."),
    wandb_run_name: str | None = typer.Option(None, help="wandb run name prefix."),
    wandb_mode: str | None = typer.Option(None, help="wandb mode."),
    torch_dtype: str | None = typer.Option(None, help="float16, bfloat16, or float32."),
    attn_implementation: str | None = typer.Option(None, help="Attention implementation."),
) -> None:
    config = _base_sweep_overrides(
        profile=profile,
        model_path=model_path,
        train_dataset_path=train_dataset_path,
        validation_dataset_path=validation_dataset_path,
        prompt_path=prompt_path,
        validation_prompt_path=validation_prompt_path,
        train_device=train_device,
        rollout_device=rollout_device,
        eval_device=eval_device,
        n_grpo_steps=n_grpo_steps,
        eval_num_examples=eval_num_examples,
        eval_every_steps=eval_every_steps,
        final_eval_num_examples=final_eval_num_examples,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_mode=wandb_mode,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )
    result = run_all_required_grpo_experiments(
        learning_rates=_parse_float_csv(learning_rates),
        off_policy_configs=_parse_off_policy_configs(off_policy_configs),
        output_dir=output_dir,
        **config,
    )
    typer.echo(json.dumps(_jsonify(result), indent=2, ensure_ascii=True))


def main() -> None:
    app()
