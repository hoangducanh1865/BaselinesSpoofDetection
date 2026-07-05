#!/usr/bin/env python3
"""Run the 2x2 routing counterfactual and matched-compute controls."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from common import (
    REPO_ROOT,
    ActiveCountCollector,
    bootstrap_eer_delta,
    build_loader,
    eer_and_threshold,
    evaluate,
    load_manifest,
    load_model,
    manifest_model_specs,
    save_frame,
    save_json,
    set_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "experiment/manifest.yaml")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/see_molex.yaml")
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "experiment/results/counterfactual"
    )
    parser.add_argument("--sources", nargs="+", default=["asvspoof5", "asvspoof2019la"])
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--max-items", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def policies(model_name: str, seed: int) -> list[tuple[str, dict]]:
    if model_name == "M0":
        return [
            ("fixed_k4", {"mode": "fixed", "k": 4}),
            ("entropy_adaptive", {"mode": "entropy"}),
        ]
    values = [("entropy_adaptive", {"mode": "entropy"})]
    values.extend((f"fixed_k{k}", {"mode": "fixed", "k": k}) for k in range(1, 7))
    values.extend(
        [
            ("shuffle_k", {"mode": "shuffle_k", "seed": seed}),
            ("random_experts", {"mode": "random_experts", "seed": seed}),
        ]
    )
    return values


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    datasets = args.datasets or manifest["datasets"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    active_frames = []
    core_predictions: dict[tuple[str, str, str, str], pd.DataFrame] = {}
    checkpoints = []

    for spec in manifest_model_specs(manifest, args.sources, ["M0", "M2"]):
        print(f"[counterfactual] loading {spec.source}/{spec.name}: {spec.checkpoint}")
        model = load_model(spec, args.config, device)
        checkpoints.append(
            {"source": spec.source, "model": spec.name, "checkpoint": str(spec.checkpoint)}
        )
        for dataset_index, dataset in enumerate(datasets):
            loader, sampled = build_loader(
                dataset,
                args.config,
                "evaluation",
                args.max_items,
                args.seed + dataset_index,
                args.batch_size,
                args.num_workers,
            )
            for policy_name, policy in policies(spec.name, args.seed + dataset_index):
                set_policy(model, policy)
                collector = ActiveCountCollector(model)
                scores, skipped = evaluate(
                    model,
                    loader,
                    device,
                    desc=f"cf {spec.source}/{spec.name}/{policy_name}/{dataset}",
                )
                collector.close()
                active = collector.frame()
                active.insert(0, "policy", policy_name)
                active.insert(0, "dataset", dataset)
                active.insert(0, "weights", spec.name)
                active.insert(0, "source", spec.source)
                active_frames.append(active)

                scores.insert(0, "policy", policy_name)
                scores.insert(0, "weights", spec.name)
                scores.insert(0, "dataset", dataset)
                scores.insert(0, "source", spec.source)
                shard = f"{spec.source}__{dataset}__{spec.name}__{policy_name}"
                save_frame(
                    scores,
                    args.output / "details" / f"{shard}__predictions.csv.gz",
                )
                if policy_name in {"fixed_k4", "entropy_adaptive"}:
                    core_predictions[
                        (spec.source, dataset, spec.name, policy_name)
                    ] = scores[["utt_id", "label", "score"]].copy()
                eer, threshold = eer_and_threshold(
                    scores["label"].to_numpy(), scores["score"].to_numpy()
                )
                summary_rows.append(
                    {
                        "source": spec.source,
                        "dataset": dataset,
                        "weights": spec.name,
                        "policy": policy_name,
                        "diagnostic_eer": eer,
                        "threshold": threshold,
                        "items": len(scores),
                        "sampled_items": len(sampled),
                        "skipped_files": len(skipped),
                        "mean_active_experts": float(active["active_mean"].mean()),
                    }
                )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    active = pd.concat(active_frames, ignore_index=True)
    comparisons = []
    for source in args.sources:
        for dataset in datasets:
            variants = {
                (weights, policy): frame
                for (stored_source, stored_dataset, weights, policy), frame
                in core_predictions.items()
                if stored_source == source and stored_dataset == dataset
            }
            pairs = [
                (("M0", "fixed_k4"), ("M0", "entropy_adaptive"), "M0 policy effect"),
                (("M2", "fixed_k4"), ("M2", "entropy_adaptive"), "M2 policy effect"),
                (("M0", "fixed_k4"), ("M2", "fixed_k4"), "training effect at fixed K4"),
                (
                    ("M0", "entropy_adaptive"),
                    ("M2", "entropy_adaptive"),
                    "training effect under adaptive routing",
                ),
            ]
            for left_key, right_key, question in pairs:
                if left_key not in variants or right_key not in variants:
                    continue
                row = bootstrap_eer_delta(
                    variants[left_key],
                    variants[right_key],
                    repeats=args.bootstrap,
                    seed=args.seed,
                )
                row.update(
                    {
                        "source": source,
                        "dataset": dataset,
                        "left": f"{left_key[0]}:{left_key[1]}",
                        "right": f"{right_key[0]}:{right_key[1]}",
                        "question": question,
                    }
                )
                comparisons.append(row)

    save_frame(pd.DataFrame(summary_rows), args.output / "policy_summary.csv")
    save_frame(active, args.output / "active_count_by_layer.csv")
    save_frame(pd.DataFrame(comparisons), args.output / "counterfactual_bootstrap.csv")
    save_json(
        {
            "suite": "counterfactual",
            "max_items_per_dataset": args.max_items,
            "seed": args.seed,
            "bootstrap_repeats": args.bootstrap,
            "checkpoints": checkpoints,
        },
        args.output / "run.json",
    )
    print(f"[counterfactual] results written to {args.output}")


if __name__ == "__main__":
    main()
