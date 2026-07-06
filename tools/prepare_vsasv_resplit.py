#!/usr/bin/env python3
"""Prepare and audit the non-destructive speaker-disjoint VSASV resplit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.vsasv import prepare_resplit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/user14/anhhd/spoof/datasets/vsasv"),
    )
    parser.add_argument(
        "--meta-dir",
        type=Path,
        default=REPO_ROOT / "outputs/vsasv_resplit/meta/vsasv_resplit",
    )
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_resplit(
        data_root=args.data_root,
        meta_dir=args.meta_dir,
        fold=args.fold,
        force=args.force,
        seed=args.seed,
    )
    print(f"Metadata: {args.meta_dir.resolve()}")
    print(json.dumps(summary["splits"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
