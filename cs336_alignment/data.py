from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries."""
    records: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_math_sft_examples(path: str | Path) -> list[dict[str, str]]:
    """Load SFT examples with keys such as 'prompt' and 'response'."""
    return load_jsonl(path)


def load_math_qa_examples(path: str | Path) -> list[dict[str, str]]:
    """Load question/answer examples used for evaluation or RL."""
    return load_jsonl(path)


def read_prompt_template(path: str | Path) -> str:
    """Read a prompt template from disk."""
    return Path(path).read_text()


def format_prompt(question: str, prompt_template: str) -> str:
    """Fill a prompt template containing a '{question}' slot."""
    return prompt_template.format(question=question)


def repeat_each(items: list[str], n: int) -> list[str]:
    """Repeat each item in order n times."""
    return [item for item in items for _ in range(n)]
