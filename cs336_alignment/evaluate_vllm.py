import typing
from typing import Callable
import json
from pathlib import Path

from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def evaluate_vllm(
        vllm_model: LLM,
        reward_fn: Callable[[str, str], dict[str, float]],
        prompts: list[str],
        ground_truths: list[str],
        eval_sampling_params: SamplingParams
) -> list:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    outputs = vllm_model.generate(prompts=prompts, sampling_params=eval_sampling_params)

    rows = []
    for prompt, gt, output in zip(prompts, ground_truths, outputs):
        response = output.outputs[0].text
        scores = reward_fn(response, gt)
        rows.append(
            {
                "prompt": prompt,
                "ground_truth": gt,
                "response": response,
                **scores,
            }
        )
    return rows


def summarize(rows):
    n = len(rows)
    reward = sum(x["reward"] for x in rows) / n
    format_reward = sum(x["format_reward"] for x in rows) / n
    answer_reward = sum(x["answer_reward"] for x in rows) / n

    both_1 = sum(
        x["format_reward"] == 1.0 and x["answer_reward"] == 1.0 for x in rows
    )
    format_1_answer_0 = sum(
        x["format_reward"] == 1.0 and x["answer_reward"] == 0.0 for x in rows
    )
    both_0 = sum(
        x["format_reward"] == 0.0 and x["answer_reward"] == 0.0 for x in rows
    )

    return {
        "num_examples": n,
        "avg_reward": reward,
        "avg_format_reward": format_reward,
        "avg_answer_reward": answer_reward,
        "count_format1_answer1": both_1,
        "count_format1_answer0": format_1_answer_0,
        "count_format0_answer0": both_0,
    }

def main():
    data_path = Path("../data/MATH/validation.jsonl")
    model_path = Path("../models/Qwen2.5-Math-1.5B")
    prompt_path = Path("prompts/r1_zero.prompt")

    examples = load_jsonl(data_path)
    # print(examples[0:10])
    prompt_template = prompt_path.read_text(encoding="utf-8")
    # print(prompt_template)
    questions = [ex["problem"] for ex in examples]
    ground_truths = [ex["answer"] for ex in examples]
    prompts = [prompt_template.format(question=q) for q in questions]
    # print(prompts[:10])

    llm = LLM(
        model=str(model_path),
        dtype="float16"
    )

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    
    rows = evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params,
    )

    out_dir = Path("results/math_baseline")
    out_dir.mkdir(parents=True, exist_ok=True)

    save_jsonl(out_dir / "predictions.jsonl", rows)

    summary = summarize(rows)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

def main1():
    data_path = Path("./results/math_baseline/predictions.jsonl")
    examples = load_jsonl(data_path)
    sample = [ex["response"] for ex in examples if ex["reward"] == 1.0 ]
    for i in range(10):
        print(f"{i}: {sample[i]}\n")


if __name__ == "__main__":
    main1()