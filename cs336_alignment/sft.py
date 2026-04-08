from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
import typer

from .data import load_math_qa_examples, load_math_sft_examples, read_prompt_template
from .drgrpo_grader import r1_zero_reward_fn
from .losses import sft_microbatch_train_step
from .modeling import (
    get_response_log_probs,
    init_vllm,
    load_policy_and_tokenizer,
    load_policy_into_vllm_instance,
)
from .tokenization import tokenize_prompt_and_output

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Run SFT experiments for the CS336 alignment assignment.",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _slice_examples(
    examples: list[dict[str, str]],
    batch_size: int,
) -> list[list[dict[str, str]]]:
    return [examples[idx : idx + batch_size] for idx in range(0, len(examples), batch_size)]


def _select_unique_examples(
    examples: list[dict[str, str]],
    max_examples: int | None,
    *,
    seed: int,
) -> list[dict[str, str]]:
    if max_examples is None or max_examples >= len(examples):
        return list(examples)

    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(range(len(examples)), k=max_examples))
    return [examples[idx] for idx in selected_indices]


def _filter_correct_sft_examples(
    examples: list[dict[str, str]],
) -> list[dict[str, str]]:
    filtered_examples = []
    for example in examples:
        ground_truth = example.get("ground_truth")
        if ground_truth is None:
            continue
        reward_info = r1_zero_reward_fn(example["response"], ground_truth)
        if reward_info["answer_reward"] == 1.0:
            filtered_examples.append(example)
    return filtered_examples


def _build_default_eval_sampling_params() -> Any:
    from vllm import SamplingParams

    return SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
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
            "eval_device": None,
            "learning_rate": 1e-5,
            "train_batch_size": 2,
            "gradient_accumulation_steps": 2,
            "num_epochs": 1,
            "max_train_examples": 32,
            "eval_num_examples": 16,
            "eval_every_steps": 0,
            "torch_dtype": torch.float16,
            "attn_implementation": "sdpa",
            "run_final_full_eval": False,
            "save_best_checkpoint": False,
            "use_wandb": False,
        }
    if profile == "cloud":
        return {
            "train_device": "cuda:0",
            "eval_device": "cuda:1",
            "learning_rate": 1e-5,
            "train_batch_size": 16,
            "gradient_accumulation_steps": 4,
            "num_epochs": 1,
            "eval_num_examples": 1024,
            "eval_every_steps": 50,
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
            "run_final_full_eval": True,
            "save_best_checkpoint": True,
            "use_wandb": False,
        }
    raise ValueError(f"Unsupported profile: {profile}")


def _merge_profile_overrides(profile: str, overrides: dict[str, Any]) -> dict[str, Any]:
    config = _profile_defaults(profile)
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


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
        config=config,
        mode=mode,
        tags=list(tags) if tags is not None else None,
    )
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")
    return wandb_run


def _evaluate_policy_with_vllm(
    *,
    policy_model,
    llm,
    validation_examples: list[dict[str, str]],
    prompt_template: str,
    sampling_params: Any,
) -> dict[str, float]:
    # Keep the vLLM copy in sync with the latest policy weights before evaluation.
    load_policy_into_vllm_instance(policy_model, llm)

    prompts = [prompt_template.format(question=example["problem"]) for example in validation_examples]
    ground_truths = [example["answer"] for example in validation_examples]
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)

    reward_infos = []
    for ground_truth, output in zip(ground_truths, outputs, strict=True):
        response = output.outputs[0].text
        reward_infos.append(r1_zero_reward_fn(response, ground_truth))

    return _summarize_reward_infos(reward_infos)


def run_sft_experiment(
    *,
    model_path: str | Path | None = None,
    sft_dataset_path: str | Path | None = None,
    validation_dataset_path: str | Path | None = None,
    validation_prompt_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    train_device: str = "cuda:0",
    eval_device: str | None = "cuda:1",
    learning_rate: float = 1e-5,
    weight_decay: float = 0.0,
    train_batch_size: int = 16,
    gradient_accumulation_steps: int = 4,
    num_epochs: int = 1,
    max_train_examples: int | None = None,
    eval_num_examples: int | None = 1024,
    eval_every_steps: int = 50,
    max_grad_norm: float = 1.0,
    loss_normalize_constant: float = 1.0,
    seed: int = 42,
    shuffle: bool = True,
    filter_to_correct_responses: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "flash_attention_2",
    gpu_memory_utilization: float = 0.85,
    final_eval_num_examples: int | None = None,
    run_final_full_eval: bool = True,
    save_checkpoint: bool = True,
    save_best_checkpoint: bool = True,
    use_wandb: bool = False,
    wandb_project: str = "cs336-alignment-sft",
    wandb_run_name: str | None = None,
    wandb_mode: str | None = None,
    wandb_tags: tuple[str, ...] | None = None,
    eval_sampling_params: Any | None = None,
) -> dict[str, Any]:
    """Run a straightforward SFT loop on prompt/response supervision data.

    The training path follows the standard teacher-forcing recipe:
    1. Read prompt/response examples from ``sft.jsonl``.
    2. Tokenize prompt and response together while tracking which labels belong
       to the response.
    3. Score the target response tokens under the policy model.
    4. Compute an SFT loss only on response tokens and update the policy.
    5. Optionally copy policy weights into a separate vLLM instance to run
       validation-time generation on another GPU.
    """

    root = _repo_root()
    model_path = Path(model_path or root / "models" / "Qwen2.5-Math-1.5B")
    sft_dataset_path = Path(sft_dataset_path or root / "data" / "MATH" / "sft.jsonl")
    validation_dataset_path = Path(
        validation_dataset_path or root / "data" / "MATH" / "validation.jsonl"
    )
    validation_prompt_path = Path(
        validation_prompt_path or root / "cs336_alignment" / "prompts" / "r1_zero.prompt"
    )
    output_dir = Path(output_dir or root / "cs336_alignment" / "results" / "sft_experiment")
    output_dir.mkdir(parents=True, exist_ok=True)

    if train_batch_size <= 0:
        raise ValueError("train_batch_size must be positive")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if train_batch_size % gradient_accumulation_steps != 0:
        raise ValueError("train_batch_size must be divisible by gradient_accumulation_steps")

    _set_seed(seed)

    train_examples = load_math_sft_examples(sft_dataset_path)
    num_train_examples_before_filter = len(train_examples)
    if filter_to_correct_responses:
        train_examples = _filter_correct_sft_examples(train_examples)
    train_examples = _select_unique_examples(
        train_examples,
        max_train_examples,
        seed=seed,
    )
    if not train_examples:
        raise ValueError("No SFT examples were loaded")

    full_validation_examples = load_math_qa_examples(validation_dataset_path)
    validation_examples = list(full_validation_examples)
    if eval_num_examples is not None:
        validation_examples = validation_examples[:eval_num_examples]
    final_validation_examples = list(full_validation_examples)
    if final_eval_num_examples is not None:
        final_validation_examples = final_validation_examples[:final_eval_num_examples]
    validation_prompt_template = read_prompt_template(validation_prompt_path)

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
    )

    llm = None
    if eval_device is not None:
        llm = init_vllm(
            model_id=str(model_path),
            device=eval_device,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        if eval_sampling_params is None:
            eval_sampling_params = _build_default_eval_sampling_params()

    microbatch_size = train_batch_size // gradient_accumulation_steps
    history: dict[str, Any] = {
        "config": {
            "model_path": str(model_path),
            "sft_dataset_path": str(sft_dataset_path),
            "validation_dataset_path": str(validation_dataset_path),
            "validation_prompt_path": str(validation_prompt_path),
            "train_device": train_device,
            "eval_device": eval_device,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "train_batch_size": train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "num_epochs": num_epochs,
            "max_train_examples": max_train_examples,
            "num_train_examples_before_filter": num_train_examples_before_filter,
            "num_train_examples_after_filter": len(train_examples),
            "filter_to_correct_responses": filter_to_correct_responses,
            "eval_num_examples": eval_num_examples,
            "eval_every_steps": eval_every_steps,
            "final_eval_num_examples": final_eval_num_examples,
            "run_final_full_eval": run_final_full_eval,
            "max_grad_norm": max_grad_norm,
            "loss_normalize_constant": loss_normalize_constant,
            "torch_dtype": str(torch_dtype),
            "attn_implementation": attn_implementation,
            "gpu_memory_utilization": gpu_memory_utilization,
            "seed": seed,
            "use_wandb": use_wandb,
            "wandb_project": wandb_project,
            "wandb_run_name": wandb_run_name,
            "wandb_mode": wandb_mode,
            "wandb_tags": list(wandb_tags) if wandb_tags is not None else None,
        },
        "train": [],
        "eval": [],
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

    optimizer_step = 0
    examples_seen = 0
    best_eval_answer_reward = float("-inf")
    best_checkpoint_dir = None

    for epoch_idx in range(num_epochs):
        epoch_examples = list(train_examples)
        if shuffle:
            random.shuffle(epoch_examples)

        # Group several microbatches together so one optimizer step sees the
        # intended effective batch size.
        group_span = microbatch_size * gradient_accumulation_steps
        for group_start in range(0, len(epoch_examples), group_span):
            microbatch_group = epoch_examples[group_start : group_start + group_span]
            microbatches = _slice_examples(microbatch_group, microbatch_size)
            actual_accumulation_steps = len(microbatches)
            optimizer.zero_grad(set_to_none=True)

            step_loss = 0.0
            step_raw_loss = 0.0

            for microbatch_examples in microbatches:
                prompts = [example["prompt"] for example in microbatch_examples]
                responses = [example["response"] for example in microbatch_examples]

                batch = tokenize_prompt_and_output(
                    prompt_strs=prompts,
                    output_strs=responses,
                    tokenizer=tokenizer,
                )

                input_ids = batch["input_ids"].to(train_device)
                labels = batch["labels"].to(train_device)
                response_mask = batch["response_mask"].to(train_device)

                scored = get_response_log_probs(
                    model=policy_model,
                    input_ids=input_ids,
                    labels=labels,
                    return_token_entropy=False,
                )

                loss, metadata = sft_microbatch_train_step(
                    policy_log_probs=scored["log_probs"],
                    response_mask=response_mask,
                    gradient_accumulation_steps=actual_accumulation_steps,
                    normalize_constant=loss_normalize_constant,
                )

                step_loss += loss.detach().item()
                step_raw_loss += metadata["unnormalized_loss"].mean().item()
                examples_seen += len(microbatch_examples)

            # Clip once per optimizer step to keep large gradients from destabilizing training.
            grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_grad_norm)
            optimizer.step()
            policy_model.train()

            optimizer_step += 1
            history["train"].append(
                {
                    "step": optimizer_step,
                    "epoch": epoch_idx + 1,
                    "examples_seen": examples_seen,
                    "loss": step_loss / actual_accumulation_steps,
                    "unnormalized_loss": step_raw_loss / actual_accumulation_steps,
                    "grad_norm": float(grad_norm),
                }
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train_step": optimizer_step,
                        **_prefix_metrics(
                            "train",
                            {
                                "loss": step_loss / actual_accumulation_steps,
                                "unnormalized_loss": step_raw_loss / actual_accumulation_steps,
                                "grad_norm": float(grad_norm),
                                "examples_seen": examples_seen,
                                "epoch": epoch_idx + 1,
                            },
                        ),
                    }
                )

            if llm is not None and eval_every_steps > 0 and optimizer_step % eval_every_steps == 0:
                policy_model.eval()
                eval_metrics = _evaluate_policy_with_vllm(
                    policy_model=policy_model,
                    llm=llm,
                    validation_examples=validation_examples,
                    prompt_template=validation_prompt_template,
                    sampling_params=eval_sampling_params,
                )
                eval_metrics["step"] = optimizer_step
                history["eval"].append(eval_metrics)
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "eval_step": optimizer_step,
                            **_prefix_metrics("eval", eval_metrics),
                        }
                    )

                if save_best_checkpoint and eval_metrics["avg_answer_reward"] > best_eval_answer_reward:
                    best_eval_answer_reward = eval_metrics["avg_answer_reward"]
                    best_checkpoint_dir = output_dir / "best_checkpoint"
                    best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    policy_model.save_pretrained(best_checkpoint_dir)
                    tokenizer.save_pretrained(best_checkpoint_dir)

                policy_model.train()

    final_eval = None
    if llm is not None and run_final_full_eval:
        policy_model.eval()
        final_eval = _evaluate_policy_with_vllm(
            policy_model=policy_model,
            llm=llm,
            validation_examples=final_validation_examples,
            prompt_template=validation_prompt_template,
            sampling_params=eval_sampling_params,
        )
        final_eval["step"] = optimizer_step
        history["final_eval"] = final_eval
        if wandb_run is not None:
            wandb_run.log(
                {
                    "eval_step": optimizer_step,
                    **_prefix_metrics(
                        "eval",
                        {
                            "final_num_examples": final_eval["num_examples"],
                            "final_avg_reward": final_eval["avg_reward"],
                            "final_avg_format_reward": final_eval["avg_format_reward"],
                            "final_avg_answer_reward": final_eval["avg_answer_reward"],
                            "final_count_format1_answer1": final_eval["count_format1_answer1"],
                            "final_count_format1_answer0": final_eval["count_format1_answer0"],
                            "final_count_format0_answer0": final_eval["count_format0_answer0"],
                        },
                    ),
                }
            )

        if save_best_checkpoint and final_eval["avg_answer_reward"] > best_eval_answer_reward:
            best_eval_answer_reward = final_eval["avg_answer_reward"]
            best_checkpoint_dir = output_dir / "best_checkpoint"
            best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            policy_model.save_pretrained(best_checkpoint_dir)
            tokenizer.save_pretrained(best_checkpoint_dir)
        policy_model.train()

    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=True))

    checkpoint_dir = None
    if save_checkpoint:
        checkpoint_dir = output_dir / "final_checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        policy_model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

    if wandb_run is not None:
        if final_eval is not None:
            wandb_run.summary["final_avg_answer_reward"] = final_eval["avg_answer_reward"]
            wandb_run.summary["final_avg_reward"] = final_eval["avg_reward"]
        wandb_run.summary["num_train_examples"] = len(train_examples)
        wandb_run.summary["num_train_examples_before_filter"] = num_train_examples_before_filter
        wandb_run.summary["num_train_examples_after_filter"] = len(train_examples)
        if best_checkpoint_dir is not None:
            wandb_run.summary["best_checkpoint_dir"] = str(best_checkpoint_dir)
        if checkpoint_dir is not None:
            wandb_run.summary["final_checkpoint_dir"] = str(checkpoint_dir)
        wandb_run.finish()

    return {
        "history": history,
        "history_path": str(history_path),
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
        "best_checkpoint_dir": str(best_checkpoint_dir) if best_checkpoint_dir is not None else None,
        "num_train_examples": len(train_examples),
        "final_eval": final_eval,
    }


def run_sft_dataset_size_sweep(
    *,
    dataset_sizes: tuple[int | None, ...] = (128, 256, 512, 1024, None),
    output_dir: str | Path | None = None,
    filter_to_correct_responses: bool = False,
    **experiment_kwargs: Any,
) -> dict[str, Any]:
    """Run the standard 4.3 dataset-size sweep and save a compact summary."""

    root = _repo_root()
    output_dir = Path(output_dir or root / "cs336_alignment" / "results" / "sft_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_results: dict[str, Any] = {}
    for dataset_size in dataset_sizes:
        run_name = "full" if dataset_size is None else str(dataset_size)
        run_output_dir = output_dir / run_name
        result = run_sft_experiment(
            max_train_examples=dataset_size,
            output_dir=run_output_dir,
            filter_to_correct_responses=filter_to_correct_responses,
            **experiment_kwargs,
        )
        sweep_results[run_name] = {
            "history_path": result["history_path"],
            "checkpoint_dir": result["checkpoint_dir"],
            "best_checkpoint_dir": result["best_checkpoint_dir"],
            "num_train_examples": result["num_train_examples"],
            "final_eval": result["final_eval"],
        }

    summary_path = output_dir / "sweep_summary.json"
    summary_path.write_text(json.dumps(sweep_results, indent=2, ensure_ascii=True))
    return {
        "output_dir": str(output_dir),
        "summary_path": str(summary_path),
        "runs": sweep_results,
    }


@app.command("single")
def cli_run_single(
    profile: str = typer.Option("local-smoke", help="Preset to start from: local-smoke or cloud."),
    model_path: Path | None = typer.Option(None, help="Override the model path."),
    sft_dataset_path: Path | None = typer.Option(None, help="Override the SFT dataset path."),
    validation_dataset_path: Path | None = typer.Option(None, help="Override the validation dataset path."),
    validation_prompt_path: Path | None = typer.Option(None, help="Override the validation prompt template."),
    output_dir: Path | None = typer.Option(None, help="Directory where logs and checkpoints are written."),
    train_device: str | None = typer.Option(None, help="Training device, e.g. cuda:0."),
    eval_device: str | None = typer.Option(None, help="Evaluation device, e.g. cuda:1, or none to disable."),
    learning_rate: float | None = typer.Option(None, help="AdamW learning rate."),
    weight_decay: float | None = typer.Option(None, help="AdamW weight decay."),
    train_batch_size: int | None = typer.Option(None, help="Effective train batch size."),
    gradient_accumulation_steps: int | None = typer.Option(None, help="Microbatches per optimizer step."),
    num_epochs: int | None = typer.Option(None, help="Number of epochs to train for."),
    max_train_examples: int | None = typer.Option(None, help="Randomly sample at most this many SFT examples."),
    eval_num_examples: int | None = typer.Option(None, help="Validation subset size for periodic evaluation."),
    eval_every_steps: int | None = typer.Option(None, help="Run validation every N optimizer steps."),
    max_grad_norm: float | None = typer.Option(None, help="Gradient clipping threshold."),
    loss_normalize_constant: float | None = typer.Option(None, help="Normalization constant passed to the SFT loss."),
    seed: int | None = typer.Option(None, help="Random seed."),
    shuffle: bool | None = typer.Option(None, "--shuffle/--no-shuffle", help="Shuffle training data each epoch."),
    filter_to_correct_responses: bool | None = typer.Option(
        None,
        "--filter-to-correct-responses/--keep-all-responses",
        help="Filter SFT examples to only those with correct final answers.",
    ),
    torch_dtype: str | None = typer.Option(None, help="float16, bfloat16, or float32."),
    attn_implementation: str | None = typer.Option(None, help="Attention implementation, e.g. sdpa, eager, flash_attention_2."),
    gpu_memory_utilization: float | None = typer.Option(None, help="vLLM gpu_memory_utilization."),
    final_eval_num_examples: int | None = typer.Option(None, help="Validation subset size for final evaluation."),
    run_final_full_eval: bool | None = typer.Option(
        None,
        "--run-final-full-eval/--skip-final-full-eval",
        help="Run one more validation pass at the end of training.",
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
    use_wandb: bool | None = typer.Option(None, "--use-wandb/--no-wandb", help="Enable wandb logging."),
    wandb_project: str | None = typer.Option(None, help="wandb project name."),
    wandb_run_name: str | None = typer.Option(None, help="wandb run name."),
    wandb_mode: str | None = typer.Option(None, help="wandb mode, e.g. online, offline, disabled."),
) -> None:
    """Run one SFT experiment, suitable for either local smoke tests or cloud runs."""

    overrides = {
        "model_path": model_path,
        "sft_dataset_path": sft_dataset_path,
        "validation_dataset_path": validation_dataset_path,
        "validation_prompt_path": validation_prompt_path,
        "output_dir": output_dir,
        "train_device": train_device,
        "eval_device": _normalize_optional_device(eval_device),
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "train_batch_size": train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_epochs": num_epochs,
        "max_train_examples": max_train_examples,
        "eval_num_examples": eval_num_examples,
        "eval_every_steps": eval_every_steps,
        "max_grad_norm": max_grad_norm,
        "loss_normalize_constant": loss_normalize_constant,
        "seed": seed,
        "shuffle": shuffle,
        "filter_to_correct_responses": filter_to_correct_responses,
        "torch_dtype": _parse_torch_dtype(torch_dtype) if torch_dtype is not None else None,
        "attn_implementation": attn_implementation,
        "gpu_memory_utilization": gpu_memory_utilization,
        "final_eval_num_examples": final_eval_num_examples,
        "run_final_full_eval": run_final_full_eval,
        "save_checkpoint": save_checkpoint,
        "save_best_checkpoint": save_best_checkpoint,
        "use_wandb": use_wandb,
        "wandb_project": wandb_project,
        "wandb_run_name": wandb_run_name,
        "wandb_mode": wandb_mode,
    }
    config = _merge_profile_overrides(profile, overrides)
    result = run_sft_experiment(**config)
    typer.echo(json.dumps(result, indent=2, ensure_ascii=True))


@app.command("sweep")
def cli_run_sweep(
    profile: str = typer.Option("cloud", help="Preset to start from: local-smoke or cloud."),
    dataset_sizes: str = typer.Option(
        "128,256,512,1024,full",
        help="Comma-separated list like 128,256,512,1024,full",
    ),
    output_dir: Path | None = typer.Option(None, help="Directory where sweep results are written."),
    filter_to_correct_responses: bool = typer.Option(
        False,
        "--filter-to-correct-responses/--keep-all-responses",
        help="Filter SFT examples to only those with correct final answers.",
    ),
    use_wandb: bool | None = typer.Option(None, "--use-wandb/--no-wandb", help="Enable wandb logging."),
    wandb_project: str | None = typer.Option(None, help="wandb project name."),
    wandb_mode: str | None = typer.Option(None, help="wandb mode, e.g. online, offline, disabled."),
) -> None:
    """Run the 4.3 dataset-size sweep."""

    parsed_dataset_sizes: list[int | None] = []
    for item in dataset_sizes.split(","):
        stripped = item.strip().lower()
        if stripped == "full":
            parsed_dataset_sizes.append(None)
        elif stripped:
            parsed_dataset_sizes.append(int(stripped))

    config = _merge_profile_overrides(
        profile,
        {
            "use_wandb": use_wandb,
            "wandb_project": wandb_project,
            "wandb_mode": wandb_mode,
        },
    )
    result = run_sft_dataset_size_sweep(
        dataset_sizes=tuple(parsed_dataset_sizes),
        output_dir=output_dir,
        filter_to_correct_responses=filter_to_correct_responses,
        **config,
    )
    typer.echo(json.dumps(result, indent=2, ensure_ascii=True))


def main() -> None:
    app()
