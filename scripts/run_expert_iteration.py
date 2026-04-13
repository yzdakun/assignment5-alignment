from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cs336_alignment.expert_iteration import main as expert_iteration_main


def main() -> None:
    expert_iteration_main()


if __name__ == "__main__":
    main()
