#!/usr/bin/env python3
"""Run the VN-Index volatility forecasting pipeline from the repository root."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

STAGE_COMMANDS: dict[str, list[str]] = {
    "data": [sys.executable, "src/prepare_data.py"],
    "econometric": [sys.executable, "src/train_econometric.py"],
    "lstm": [sys.executable, "src/train_lstm_hybrid.py"],
    "tune": [sys.executable, "src/tune_lstm_hybrid.py"],
    "evaluate": [sys.executable, "src/evaluate_all.py"],
    "paper": ["make", "-C", "paper", "all"],
}

DEFAULT_STAGES = ["data", "econometric", "lstm", "tune", "evaluate", "paper"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one or more stages of the VN-Index volatility experiment."
    )
    parser.add_argument(
        "stages",
        nargs="*",
        choices=STAGE_COMMANDS,
        help="Stages to run. Defaults to the full pipeline.",
    )
    parser.add_argument(
        "--skip-tune",
        action="store_true",
        help="Skip tuned neural variants when running the default full pipeline.",
    )
    parser.add_argument(
        "--skip-paper",
        action="store_true",
        help="Skip LaTeX paper compilation when running the default full pipeline.",
    )
    return parser.parse_args()


def selected_stages(args: argparse.Namespace) -> list[str]:
    if args.stages:
        return list(args.stages)

    stages = list(DEFAULT_STAGES)
    if args.skip_tune:
        stages = [stage for stage in stages if stage != "tune"]
    if args.skip_paper:
        stages = [stage for stage in stages if stage != "paper"]
    return stages


def run_stage(stage: str) -> None:
    command = STAGE_COMMANDS[stage]
    print(f"\n==> Running {stage}: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    for stage in selected_stages(args):
        run_stage(stage)


if __name__ == "__main__":
    main()
