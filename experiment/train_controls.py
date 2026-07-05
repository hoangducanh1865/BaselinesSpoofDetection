#!/usr/bin/env python3
"""Launch optional fresh-training controls after the checkpoint-only analysis."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml

from common import REPO_ROOT, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["asvspoof5", "asvspoof2019la"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--control",
        choices=["default", "no_warmup", "max4", "static_tau"],
        required=True,
    )
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/see_molex.yaml")
    parser.add_argument(
        "--config-output", type=Path, default=REPO_ROOT / "experiment/generated_configs"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.control == "no_warmup":
        cfg["routing"]["warmup_epochs"] = 0
    elif args.control == "max4":
        cfg["routing"]["k_max"] = 4
    elif args.control == "static_tau":
        cfg["routing"]["tau_max"] = cfg["routing"]["tau_min"]

    args.config_output.mkdir(parents=True, exist_ok=True)
    generated = args.config_output / f"{args.dataset}_{args.control}_seed{args.seed}.yaml"
    generated.write_text(yaml.safe_dump(cfg, sort_keys=False))
    command = [
        "python",
        str(REPO_ROOT / "main.py"),
        "--baseline",
        "see_molex",
        "--ablation",
        "M2",
        "--mode",
        "train",
        "--dataset",
        args.dataset,
        "--config",
        str(generated),
        "--seed",
        str(args.seed),
    ]
    print("Launching:", " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
