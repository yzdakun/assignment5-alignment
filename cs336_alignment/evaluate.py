from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import json


def evaluate_vllm(
    vllm_model,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    ground_truths: list[str],
    eval_sampling_params,
) -> dict[str, object]:
    """Run batched vLLM evaluation and aggregate reward metrics."""
    raise NotImplementedError


def save_evaluation_results(path: str | Path, payload: dict[str, object]) -> None:
    """Persist evaluation outputs for later analysis."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
