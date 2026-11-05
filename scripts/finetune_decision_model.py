#!/usr/bin/env python3
"""Prepare reasoning dataset and delegate to the selected model's native fine-tuning tool."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc_decision.finetuning import main


if __name__ == "__main__":
    raise SystemExit(main())