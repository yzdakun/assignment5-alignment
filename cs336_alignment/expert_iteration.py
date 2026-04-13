from __future__ import annotations

import gc
import json
import random
from pathlib import Path
from typing import Any

import torch
import typer

from .data import format_prompt, load_math_qa_examples, read_prompt_template
from .drgrpo_grader import r1_zero_reward_fn
from .losses import masked_mean
from .modeling import get_response_log_probs, init_vllm, load_policy_and_tokenizer
from .sft import run_sft_experiment
from .tokenization import tokenize_prompt_and_output

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Run expert iteration experiments for the CS336 alignment assignment.",
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


def _profile_defaults(profile: str) -> dict[str, Any]:
    if profile == "local-smoke":
        return {
            "train_device": "cuda:0",
            "rollout_device": "cuda:0",
            "eval_device": None,
            "learning_rate": 1e-5,
            "weight_decay": 0.0,
            "train_batch_size": 2,
            "gradient_accumulation_steps": 2,
            "num_epochs": 1,
            "ei_batch_size": 64,
            "rollouts_per_question": 2,
            "n_ei_steps": 1,
            "eval_num_examples": 64,
            "eval_every_steps": 0,
            "entropy_num_examples": 32,
            "final_eval_num_examples": 64,
            "run_final_full_eval": False,
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
            "gpu_memory_utilization": 0.75,
            "save_rollouts": True,
            "use_wandb": False,
        }
    if profile == "cloud":
        return {
            "train_device": "cuda:0",
            "rollout_device": "cuda:1",
            "eval_device": "cuda:1",
            "learning_rate": 1e-5,
            "weight_decay": 0.0,
            "train_batch_size": 16,
            "gradient_accumulation_steps": 4,
            "num_epochs": 1,
            "ei_batch_size": 512,
            "rollouts_per_question": 4,
            "n_ei_steps": 5,
            "eval_num_examples": 1024,
            "eval_every_steps": 50,
            "entropy_num_examples": 128,
            "final_eval_num_examples": None,
            "run_final_full_eval": True,
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
            "gpu_memory_utilization": 0.85,
            "save_rollouts": True,
            "use_wandb": False,
        }
    raise ValueError(f"Unsupported profile: {profile}")


def _merge_profile_overrides(profile: str, overrides: dict[str, Any]) -> dict[str, Any]:
    config = _profile_defaults(profile)
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


def _select_step_examples(
    examples: list[dict[str, Any]],
    batch_size: int | None,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    if batch_size is None or batch_size >= len(examples):
        return list(examples)

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


def _generate_rollout_rows(
    *,
    model_path: Path,
    rollout_examples: list[dict[str, Any]],
    prompt_template: str,
    rollout_device: str,
    rollouts_per_question: int,
    sampling_max_tokens: int,
    sampling_min_tokens: int,
    sampling_temperature: float,
    gpu_memory_utilization: float,
    seed: int,
    ei_step: int,
) -> list[dict[str, Any]]:
    llm = init_vllm(
        model_id=str(model_path),
        device=rollout_device,
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    sampling_params = _build_sampling_params(
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        n=rollouts_per_question,
        temperature=sampling_temperature,
        seed=seed,
    )

    prompts = [format_prompt(example["problem"], prompt_template) for example in rollout_examples]
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)

    rows: list[dict[str, Any]] = []
    for example_idx, (example, prompt, output_group) in enumerate(
        zip(rollout_examples, prompts, outputs, strict=True)
    ):
        ground_truth = example["answer"]
        for rollout_idx, candidate in enumerate(output_group.outputs):
            response = candidate.text
            reward_info = r1_zero_reward_fn(response, ground_truth)
            rows.append(
                {
                    "ei_step": ei_step,
                    "problem_index": example_idx,
                    "rollout_index": rollout_idx,
                    "problem": example["problem"],
                    "ground_truth": ground_truth,
                    "prompt": prompt,
                    "response": response,
                    **reward_info,
                }
            )

    del llm
    _cleanup_memory()
    return rows


def _collect_entropy_metrics(
    *,
    model_path: Path,
    validation_examples: list[dict[str, Any]],
    prompt_template: str,
    rollout_device: str,
    score_device: str,
    sampling_max_tokens: int,
    sampling_min_tokens: int,
    sampling_temperature: float,
    gpu_memory_utilization: float,
    seed: int,
    torch_dtype: torch.dtype,
    attn_implementation: str,
    max_examples_to_log: int = 8,
) -> dict[str, Any]:
    prompts = [format_prompt(example["problem"], prompt_template) for example in validation_examples]
    ground_truths = [example["answer"] for example in validation_examples]

    llm = init_vllm(
        model_id=str(model_path),
        device=rollout_device,
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    sampling_params = _build_sampling_params(
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        n=1,
        temperature=sampling_temperature,
        seed=seed,
    )
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
    responses = [output.outputs[0].text for output in outputs]

    reward_infos = [
        r1_zero_reward_fn(response, ground_truth)
        for response, ground_truth in zip(responses, ground_truths, strict=True)
    ]

    del llm
    _cleanup_memory()

    policy_model, tokenizer = load_policy_and_tokenizer(
        str(model_path),
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )
    policy_model.to(score_device)
    policy_model.eval()

    batch = tokenize_prompt_and_output(
        prompt_strs=prompts,
        output_strs=responses,
        tokenizer=tokenizer,
    )
    input_ids = batch["input_ids"].to(score_device)
    labels = batch["labels"].to(score_device)
    response_mask = batch["response_mask"].to(score_device)

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

    del policy_model
    del tokenizer
    _cleanup_memory()

    return {
        **_summarize_reward_infos(reward_infos),
        "mean_token_entropy": avg_token_entropy.mean().item(),
        "mean_response_length": response_lengths.mean().item(),
        "mean_response_length_correct": safe_mean(response_lengths, correct_mask),
        "mean_response_length_incorrect": safe_mean(response_lengths, incorrect_mask),
        "examples": preview,
    }


def run_expert_iteration_experiment(
    *,
    model_path: str | Path | None = None,
    train_dataset_path: str | Path | None = None,
    validation_dataset_path: str | Path | None = None,
    prompt_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    train_device: str = "cuda:0",
    rollout_device: str = "cuda:1",
    eval_device: str | None = "cuda:1",
    learning_rate: float = 1e-5,
    weight_decay: float = 0.0,
    train_batch_size: int = 16,
    gradient_accumulation_steps: int = 4,
    num_epochs: int = 1,
    ei_batch_size: int = 512,
    rollouts_per_question: int = 4,
    n_ei_steps: int = 5,
    eval_num_examples: int | None = 1024,
    eval_every_steps: int = 50,
    entropy_num_examples: int = 128,
    max_grad_norm: float = 1.0,
    loss_normalize_constant: float = 1.0,
    seed: int = 42,
    shuffle: bool = True,
    torch_dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "flash_attention_2",
    gpu_memory_utilization: float = 0.85,
    sampling_max_tokens: int = 1024,
    sampling_min_tokens: int = 4,
    sampling_temperature: float = 1.0,
    final_eval_num_examples: int | None = None,
    run_final_full_eval: bool = True,
    save_rollouts: bool = True,
    save_checkpoint: bool = True,
    save_best_checkpoint: bool = True,
    use_wandb: bool = False,
    wandb_project: str = "cs336-alignment-ei",
    wandb_run_name: str | None = None,
    wandb_mode: str | None = None,
) -> dict[str, Any]:
    """Run expert iteration by alternating rollout filtering and SFT updates."""

    if ei_batch_size <= 0:
        raise ValueError("ei_batch_size must be positive")
    if rollouts_per_question <= 0:
        raise ValueError("rollouts_per_question must be positive")
    if n_ei_steps <= 0:
        raise ValueError("n_ei_steps must be positive")
    if entropy_num_examples <= 0:
        raise ValueError("entropy_num_examples must be positive")

    root = _repo_root()
    model_path = Path(model_path or root / "models" / "Qwen2.5-Math-1.5B")
    train_dataset_path = Path(train_dataset_path or root / "data" / "MATH" / "train.jsonl")
    validation_dataset_path = Path(
        validation_dataset_path or root / "data" / "MATH" / "validation.jsonl"
    )
    prompt_path = Path(prompt_path or root / "cs336_alignment" / "prompts" / "r1_zero.prompt")
    output_dir = Path(output_dir or root / "cs336_alignment" / "results" / "expert_iteration")
    output_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(seed)

    prompt_template = read_prompt_template(prompt_path)
    full_train_examples = load_math_qa_examples(train_dataset_path)
    full_validation_examples = load_math_qa_examples(validation_dataset_path)

    validation_examples = list(full_validation_examples)
    if eval_num_examples is not None:
        validation_examples = validation_examples[:eval_num_examples]
    entropy_examples = validation_examples[: min(entropy_num_examples, len(validation_examples))]

    current_model_path = model_path
    summary: dict[str, Any] = {
        "config": {
            "model_path": str(model_path),
            "train_dataset_path": str(train_dataset_path),
            "validation_dataset_path": str(validation_dataset_path),
            "prompt_path": str(prompt_path),
            "output_dir": str(output_dir),
            "train_device": train_device,
            "rollout_device": rollout_device,
            "eval_device": eval_device,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "train_batch_size": train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "num_epochs": num_epochs,
            "ei_batch_size": ei_batch_size,
            "rollouts_per_question": rollouts_per_question,
            "n_ei_steps": n_ei_steps,
            "eval_num_examples": eval_num_examples,
            "eval_every_steps": eval_every_steps,
            "entropy_num_examples": entropy_num_examples,
            "max_grad_norm": max_grad_norm,
            "loss_normalize_constant": loss_normalize_constant,
            "seed": seed,
            "torch_dtype": str(torch_dtype),
            "attn_implementation": attn_implementation,
            "gpu_memory_utilization": gpu_memory_utilization,
            "sampling_max_tokens": sampling_max_tokens,
            "sampling_min_tokens": sampling_min_tokens,
            "sampling_temperature": sampling_temperature,
            "final_eval_num_examples": final_eval_num_examples,
            "run_final_full_eval": run_final_full_eval,
            "save_rollouts": save_rollouts,
            "save_checkpoint": save_checkpoint,
            "save_best_checkpoint": save_best_checkpoint,
            "use_wandb": use_wandb,
            "wandb_project": wandb_project,
            "wandb_run_name": wandb_run_name,
            "wandb_mode": wandb_mode,
        },
        "initial_eval": None,
        "steps": [],
    }

    if entropy_examples:
        summary["initial_eval"] = _collect_entropy_metrics(
            model_path=current_model_path,
            validation_examples=entropy_examples,
            prompt_template=prompt_template,
            rollout_device=rollout_device,
            score_device=train_device,
            sampling_max_tokens=sampling_max_tokens,
            sampling_min_tokens=sampling_min_tokens,
            sampling_temperature=sampling_temperature,
            gpu_memory_utilization=gpu_memory_utilization,
            seed=seed,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )

    best_step: dict[str, Any] | None = None
    best_eval_answer_reward = float("-inf")

    for ei_step in range(1, n_ei_steps + 1):
        step_dir = output_dir / f"step_{ei_step}"
        step_dir.mkdir(parents=True, exist_ok=True)

        step_examples = _select_step_examples(
            full_train_examples,
            ei_batch_size,
            seed=seed + ei_step,
        )
        rollout_rows = _generate_rollout_rows(
            model_path=current_model_path,
            rollout_examples=step_examples,
            prompt_template=prompt_template,
            rollout_device=rollout_device,
            rollouts_per_question=rollouts_per_question,
            sampling_max_tokens=sampling_max_tokens,
            sampling_min_tokens=sampling_min_tokens,
            sampling_temperature=sampling_temperature,
            gpu_memory_utilization=gpu_memory_utilization,
            seed=seed + ei_step,
            ei_step=ei_step,
        )

        if save_rollouts:
            _save_jsonl(step_dir / "rollout_samples.jsonl", rollout_rows)

        filtered_examples = [
            {
                "prompt": row["prompt"],
                "response": row["response"],
                "ground_truth": row["ground_truth"],
                "problem": row["problem"],
            }
            for row in rollout_rows
            if row["answer_reward"] == 1.0
        ]
        _save_jsonl(step_dir / "filtered_sft.jsonl", filtered_examples)

        step_summary: dict[str, Any] = {
            "ei_step": ei_step,
            "input_model_path": str(current_model_path),
            "num_selected_questions": len(step_examples),
            "num_rollouts_total": len(rollout_rows),
            "num_correct_rollouts": len(filtered_examples),
            "correct_rollout_ratio": (
                len(filtered_examples) / len(rollout_rows) if rollout_rows else 0.0
            ),
            "training_skipped": False,
            "sft_result": None,
            "post_step_eval": None,
            "next_model_path": str(current_model_path),
        }

        if filtered_examples:
            sft_result = run_sft_experiment(
                model_path=current_model_path,
                sft_dataset_path=step_dir / "filtered_sft.jsonl",
                validation_dataset_path=validation_dataset_path,
                validation_prompt_path=prompt_path,
                output_dir=step_dir,
                train_device=train_device,
                eval_device=eval_device,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                train_batch_size=train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                num_epochs=num_epochs,
                max_train_examples=None,
                eval_num_examples=eval_num_examples,
                eval_every_steps=eval_every_steps,
                max_grad_norm=max_grad_norm,
                loss_normalize_constant=loss_normalize_constant,
                seed=seed + ei_step,
                shuffle=shuffle,
                filter_to_correct_responses=False,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
                gpu_memory_utilization=gpu_memory_utilization,
                final_eval_num_examples=final_eval_num_examples,
                run_final_full_eval=run_final_full_eval,
                save_checkpoint=save_checkpoint,
                save_best_checkpoint=save_best_checkpoint,
                use_wandb=use_wandb,
                wandb_project=wandb_project,
                wandb_run_name=(
                    f"{wandb_run_name}-step-{ei_step}" if wandb_run_name is not None else None
                ),
                wandb_mode=wandb_mode,
                wandb_tags=("expert_iteration", f"step_{ei_step}"),
            )
            next_model_path_str = (
                sft_result["best_checkpoint_dir"]
                or sft_result["checkpoint_dir"]
                or str(current_model_path)
            )
            current_model_path = Path(next_model_path_str)
            step_summary["sft_result"] = {
                "history_path": sft_result["history_path"],
                "checkpoint_dir": sft_result["checkpoint_dir"],
                "best_checkpoint_dir": sft_result["best_checkpoint_dir"],
                "num_train_examples": sft_result["num_train_examples"],
                "final_eval": sft_result["final_eval"],
            }
        else:
            step_summary["training_skipped"] = True

        if entropy_examples:
            step_summary["post_step_eval"] = _collect_entropy_metrics(
                model_path=current_model_path,
                validation_examples=entropy_examples,
                prompt_template=prompt_template,
                rollout_device=rollout_device,
                score_device=train_device,
                sampling_max_tokens=sampling_max_tokens,
                sampling_min_tokens=sampling_min_tokens,
                sampling_temperature=sampling_temperature,
                gpu_memory_utilization=gpu_memory_utilization,
                seed=seed + 1000 + ei_step,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
            )
            answer_reward = step_summary["post_step_eval"]["avg_answer_reward"]
            if answer_reward > best_eval_answer_reward:
                best_eval_answer_reward = answer_reward
                best_step = {
                    "ei_step": ei_step,
                    "avg_answer_reward": answer_reward,
                    "model_path": str(current_model_path),
                }

        step_summary["next_model_path"] = str(current_model_path)
        step_summary_path = step_dir / "step_summary.json"
        step_summary_path.write_text(json.dumps(step_summary, indent=2, ensure_ascii=True))
        summary["steps"].append(step_summary)

        summary_path = output_dir / "summary.json"
        summary["best_step"] = best_step
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True))

    summary["best_step"] = best_step
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True))

    return {
        "summary_path": str(summary_path),
        "best_step": best_step,
        "final_model_path": str(current_model_path),
        "steps": summary["steps"],
    }


@app.command("single")
def cli_run_single(
    profile: str = typer.Option("local-smoke", help="Preset to start from: local-smoke or cloud."),
    model_path: Path | None = typer.Option(None, help="Override the starting model path."),
    train_dataset_path: Path | None = typer.Option(None, help="Override the EI train dataset path."),
    validation_dataset_path: Path | None = typer.Option(
        None,
        help="Override the validation dataset path.",
    ),
    prompt_path: Path | None = typer.Option(None, help="Override the prompt template path."),
    output_dir: Path | None = typer.Option(None, help="Directory where EI logs are written."),
    train_device: str | None = typer.Option(None, help="Training device, e.g. cuda:0."),
    rollout_device: str | None = typer.Option(None, help="Rollout device, e.g. cuda:1."),
    eval_device: str | None = typer.Option(
        None,
        help="Validation device used inside per-step SFT, or none to disable.",
    ),
    learning_rate: float | None = typer.Option(None, help="AdamW learning rate."),
    weight_decay: float | None = typer.Option(None, help="AdamW weight decay."),
    train_batch_size: int | None = typer.Option(None, help="Effective SFT train batch size."),
    gradient_accumulation_steps: int | None = typer.Option(
        None,
        help="Microbatches per optimizer step.",
    ),
    num_epochs: int | None = typer.Option(None, help="Number of SFT epochs per EI step."),
    ei_batch_size: int | None = typer.Option(None, help="Number of problems sampled per EI step."),
    rollouts_per_question: int | None = typer.Option(
        None,
        help="Number of sampled trajectories per question.",
    ),
    n_ei_steps: int | None = typer.Option(None, help="How many EI steps to run."),
    eval_num_examples: int | None = typer.Option(
        None,
        help="Validation subset size used inside per-step SFT.",
    ),
    eval_every_steps: int | None = typer.Option(None, help="Run validation every N SFT steps."),
    entropy_num_examples: int | None = typer.Option(
        None,
        help="Number of validation examples used to compute entropy summaries.",
    ),
    max_grad_norm: float | None = typer.Option(None, help="Gradient clipping threshold."),
    loss_normalize_constant: float | None = typer.Option(
        None,
        help="Normalization constant passed to the SFT loss.",
    ),
    seed: int | None = typer.Option(None, help="Random seed."),
    shuffle: bool | None = typer.Option(None, "--shuffle/--no-shuffle", help="Shuffle SFT data."),
    torch_dtype: str | None = typer.Option(None, help="float16, bfloat16, or float32."),
    attn_implementation: str | None = typer.Option(
        None,
        help="Attention implementation, e.g. sdpa, eager, flash_attention_2.",
    ),
    gpu_memory_utilization: float | None = typer.Option(
        None,
        help="vLLM gpu_memory_utilization.",
    ),
    sampling_max_tokens: int | None = typer.Option(None, help="Maximum rollout length."),
    sampling_min_tokens: int | None = typer.Option(
        None,
        help="Minimum rollout length to avoid empty generations.",
    ),
    sampling_temperature: float | None = typer.Option(
        None,
        help="Sampling temperature for EI rollouts.",
    ),
    final_eval_num_examples: int | None = typer.Option(
        None,
        help="Subset size for the final per-step SFT evaluation.",
    ),
    run_final_full_eval: bool | None = typer.Option(
        None,
        "--run-final-full-eval/--skip-final-full-eval",
        help="Run one final validation pass inside each per-step SFT run.",
    ),
    save_rollouts: bool | None = typer.Option(
        None,
        "--save-rollouts/--skip-save-rollouts",
        help="Persist raw rollout samples to disk.",
    ),
    save_checkpoint: bool | None = typer.Option(
        None,
        "--save-checkpoint/--skip-save-checkpoint",
        help="Save the final checkpoint for each EI step.",
    ),
    save_best_checkpoint: bool | None = typer.Option(
        None,
        "--save-best-checkpoint/--skip-save-best-checkpoint",
        help="Save the best checkpoint for each EI step.",
    ),
    use_wandb: bool | None = typer.Option(None, "--use-wandb/--no-wandb", help="Enable wandb."),
    wandb_project: str | None = typer.Option(None, help="wandb project name."),
    wandb_run_name: str | None = typer.Option(None, help="wandb run name prefix."),
    wandb_mode: str | None = typer.Option(None, help="wandb mode, e.g. online, offline."),
) -> None:
    """Run one expert iteration configuration."""

    overrides = {
        "model_path": model_path,
        "train_dataset_path": train_dataset_path,
        "validation_dataset_path": validation_dataset_path,
        "prompt_path": prompt_path,
        "output_dir": output_dir,
        "train_device": train_device,
        "rollout_device": rollout_device,
        "eval_device": _normalize_optional_device(eval_device),
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "train_batch_size": train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_epochs": num_epochs,
        "ei_batch_size": ei_batch_size,
        "rollouts_per_question": rollouts_per_question,
        "n_ei_steps": n_ei_steps,
        "eval_num_examples": eval_num_examples,
        "eval_every_steps": eval_every_steps,
        "entropy_num_examples": entropy_num_examples,
        "max_grad_norm": max_grad_norm,
        "loss_normalize_constant": loss_normalize_constant,
        "seed": seed,
        "shuffle": shuffle,
        "torch_dtype": _parse_torch_dtype(torch_dtype) if torch_dtype is not None else None,
        "attn_implementation": attn_implementation,
        "gpu_memory_utilization": gpu_memory_utilization,
        "sampling_max_tokens": sampling_max_tokens,
        "sampling_min_tokens": sampling_min_tokens,
        "sampling_temperature": sampling_temperature,
        "final_eval_num_examples": final_eval_num_examples,
        "run_final_full_eval": run_final_full_eval,
        "save_rollouts": save_rollouts,
        "save_checkpoint": save_checkpoint,
        "save_best_checkpoint": save_best_checkpoint,
        "use_wandb": use_wandb,
        "wandb_project": wandb_project,
        "wandb_run_name": wandb_run_name,
        "wandb_mode": wandb_mode,
    }
    config = _merge_profile_overrides(profile, overrides)
    result = run_expert_iteration_experiment(**config)
    typer.echo(json.dumps(result, indent=2, ensure_ascii=True))


def main() -> None:
    app()

