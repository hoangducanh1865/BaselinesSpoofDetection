import argparse
import importlib
import random
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

from baselines import REGISTRY

DATASETS = ["asvspoof5", "asvspoof2019la", "asvspoof2019pa"]
MODES = ["train", "eval", "score"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified entry point for spoof-detection baselines")
    p.add_argument("--baseline", required=True, choices=list(REGISTRY))
    p.add_argument("--dataset", default="asvspoof5", choices=DATASETS)
    p.add_argument("--mode", default="train", choices=MODES)
    p.add_argument("--config", default=None)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max-steps", type=int, default=None,
                    help="Smoke test: truncate to a tiny dataset subset and 1 epoch.")
    p.add_argument("--dry-run", action="store_true",
                    help="Smoke test: same as --max-steps with a small built-in default.")
    p.add_argument("--ckpt", default=None, help="Checkpoint path for --mode eval/score.")
    p.add_argument("--resume", nargs="?", const="latest", default=None,
                    help="Resume training. Optionally pass a MoLEx run folder like YYYY_MM_DD_HH_MM_SS.")
    return p.parse_args()


def main() -> None:
    # .env may live in the repo root or one level above it on the server.
    # Never print, log, or commit its contents.
    repo_root = Path(__file__).resolve().parent
    load_dotenv(repo_root / ".env")
    load_dotenv(repo_root.parent / ".env")

    args = parse_args()
    if REGISTRY[args.baseline] is None:
        raise NotImplementedError(f"Baseline '{args.baseline}' is registered but not wired up yet.")

    set_seed(args.seed)
    module = importlib.import_module(REGISTRY[args.baseline])
    getattr(module, args.mode)(args)


if __name__ == "__main__":
    main()
