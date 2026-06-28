#!/usr/bin/env python3
"""Create a temporary SEE-MoLEx checkpoint average from validation loss."""

import argparse
import re
from pathlib import Path

import torch


def read_losses(path: Path) -> dict[int, float]:
    losses = {}
    with path.open() as handle:
        for line in handle:
            if not line.startswith("Epoch"):
                continue
            epoch_text, loss_text = line.split(":", 1)
            epoch = int(epoch_text.split()[1]) - 1
            losses[epoch] = float(loss_text.strip())
    return losses


def checkpoint_for_epoch(weights_dir: Path, epoch: int) -> Path | None:
    patterns = (
        f"validation_epoch_{epoch}_loss_*.pth",
        f"epoch_{epoch}_*.pth",
        f"latest_checkpoint_epoch_{epoch}.pth",
    )
    for pattern in patterns:
        matches = sorted(weights_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def average_checkpoints(paths: list[Path]) -> dict:
    state_dicts = [torch.load(path, map_location="cpu") for path in paths]
    keys = state_dicts[0].keys()
    for path, state_dict in zip(paths[1:], state_dicts[1:]):
        if state_dict.keys() != keys:
            raise ValueError(f"Checkpoint keys do not match: {path}")

    averaged = {}
    for key in keys:
        reference = state_dicts[0][key]
        value = sum(state_dict[key].float() for state_dict in state_dicts)
        value = value / len(state_dicts)
        averaged[key] = value.to(reference.dtype)
    return averaged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output-name", default="tmp_averaged_checkpoint.pth"
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Average available members of the top-k when historical weights are missing.",
    )
    args = parser.parse_args()
    if args.top_k <= 0:
        raise SystemExit("--top-k must be greater than zero.")

    run_dir = args.run_dir.resolve()
    weights_dir = run_dir / "weights"
    losses = read_losses(run_dir / "valid_loss.txt")
    if not losses:
        raise SystemExit(f"No validation losses found in {run_dir / 'valid_loss.txt'}.")
    epochs = sorted(losses, key=lambda epoch: (losses[epoch], epoch))[: args.top_k]

    selected = []
    missing = []
    for epoch in epochs:
        checkpoint = checkpoint_for_epoch(weights_dir, epoch)
        if checkpoint is None:
            missing.append(epoch)
        else:
            selected.append((epoch, losses[epoch], checkpoint))

    print(
        "Top validation-loss epochs (log/checkpoint):",
        ", ".join(f"{epoch + 1}/{epoch}" for epoch in epochs),
    )
    for epoch, loss, checkpoint in selected:
        print(
            f"  log_epoch={epoch + 1} checkpoint_epoch={epoch} "
            f"valid_loss={loss:.8f} checkpoint={checkpoint}"
        )
    if missing:
        print("Missing checkpoint weights for epochs:", ", ".join(map(str, missing)))
        if not args.allow_partial:
            raise SystemExit(
                "Cannot create an exact top-k average. Re-run with --allow-partial "
                "to create a temporary average from available top-k members."
            )
    if not selected:
        raise SystemExit("No checkpoint weights are available for the selected epochs.")

    output_path = weights_dir / args.output_name
    torch.save(
        average_checkpoints([checkpoint for _, _, checkpoint in selected]),
        output_path,
    )
    print(f"Saved {len(selected)}-checkpoint average to: {output_path}")


if __name__ == "__main__":
    main()
