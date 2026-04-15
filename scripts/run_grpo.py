from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cs336_alignment.grpo import main as grpo_main


def main() -> None:
    grpo_main()


if __name__ == "__main__":
    main()
