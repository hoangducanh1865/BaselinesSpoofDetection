#!/usr/bin/env python3
"""Collect native routing, utilization, and OOD-entropy profiles."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import (
    REPO_ROOT,
    RoutingCollector,
    auroc_binary,
    build_loader,
    eer_and_threshold,
    gini,
    load_manifest,
    load_model,
    mark_shard_done,
    manifest_model_specs,
    prepare_shard_output,
    save_frame,
    save_json,
    set_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "experiment/manifest.yaml")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/see_molex.yaml")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "experiment/results/profile")
    parser.add_argument("--sources", nargs="+", default=["asvspoof5", "asvspoof2019la"])
    parser.add_argument("--models", nargs="+", default=["M0", "M1", "M2", "M3"])
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--max-items", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def native_policy(model_name: str) -> dict:
    if model_name in {"M2", "M3"}:
        return {"mode": "entropy"}
    return {"mode": "fixed", "k": 4}


def selection_nmi(group: pd.DataFrame, metadata: str) -> float:
    counts = group.pivot_table(
        index="expert",
        columns=metadata,
        values="selected_tokens",
        aggfunc="sum",
        fill_value=0,
    ).to_numpy(dtype=np.float64)
    total = counts.sum()
    if total <= 0 or counts.shape[0] < 2 or counts.shape[1] < 2:
        return float("nan")
    joint = counts / total
    expert_prob = joint.sum(axis=1, keepdims=True)
    metadata_prob = joint.sum(axis=0, keepdims=True)
    expected = expert_prob @ metadata_prob
    mask = joint > 0
    mutual_information = float((joint[mask] * np.log(joint[mask] / expected[mask])).sum())
    expert_entropy = float(-(expert_prob[expert_prob > 0] * np.log(expert_prob[expert_prob > 0])).sum())
    metadata_entropy = float(
        -(metadata_prob[metadata_prob > 0] * np.log(metadata_prob[metadata_prob > 0])).sum()
    )
    denominator = math.sqrt(expert_entropy * metadata_entropy)
    return mutual_information / denominator if denominator > 0 else float("nan")


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    datasets = args.datasets or manifest["datasets"]
    if args.quick:
        args.models = ["M0", "M2", "M3"]
    output_root = args.output
    args.output, already_done = prepare_shard_output(output_root, args.shard_index)
    if already_done:
        print(f"[profile] shard already complete, skipping: {args.output}")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)

    profile_summaries = []
    utilization_frames = []
    coactivation_frames = []
    eer_rows = []
    ood_rows = []
    checkpoint_rows = []

    for spec in manifest_model_specs(manifest, args.sources, args.models):
        print(f"[profile] loading {spec.source}/{spec.name}: {spec.checkpoint}")
        model = load_model(spec, args.config, device)
        set_policy(model, native_policy(spec.name))
        checkpoint_rows.append(
            {
                "source": spec.source,
                "model": spec.name,
                "checkpoint": str(spec.checkpoint),
            }
        )
        entropy_by_dataset: dict[str, pd.DataFrame] = {}
        if spec.source not in datasets:
            raise ValueError(
                f"Dataset list must contain source dataset '{spec.source}' for OOD AUROC."
            )

        for dataset_index, dataset in enumerate(datasets):
            loader, sampled = build_loader(
                dataset=dataset,
                config_path=args.config,
                split="evaluation",
                max_items=args.max_items,
                seed=args.seed + dataset_index,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shard_index=args.shard_index,
                shard_size=args.shard_size,
            )
            if sampled.empty:
                print(f"[profile] empty shard for {dataset}; skipping")
                continue
            collector = RoutingCollector(
                model, spec.source, spec.name, dataset, metadata=sampled
            )
            from common import evaluate

            scores, skipped = evaluate(
                model,
                loader,
                device,
                collector=collector,
                desc=f"profile {spec.source}/{spec.name}/{dataset}",
            )
            collector.close()
            profile, utilization, coactivation = collector.frames()
            scores.insert(0, "dataset", dataset)
            scores.insert(0, "model", spec.name)
            scores.insert(0, "source", spec.source)
            scores["skipped_files"] = len(skipped)
            scores["sampled_items"] = len(sampled)
            shard = f"{spec.source}__{spec.name}__{dataset}"
            save_frame(
                profile, args.output / "details" / f"{shard}__routing.csv.gz"
            )
            save_frame(
                scores, args.output / "details" / f"{shard}__predictions.csv.gz"
            )
            profile_summaries.append(
                profile.groupby(
                    ["source", "model", "dataset", "layer"], as_index=False
                )[
                    [
                        "entropy_mean",
                        "entropy_std",
                        "active_mean",
                        "total_active_mean",
                        "active_std",
                        "active_p90",
                        "pmax_mean",
                        "margin_mean",
                    ]
                    + [column for column in profile if re_k(column)]
                ].mean()
            )
            utilization_frames.append(utilization)
            coactivation_frames.append(coactivation)
            entropy_by_dataset[dataset] = profile[["layer", "entropy_mean"]].copy()
            eer, threshold = eer_and_threshold(
                scores["label"].to_numpy(), scores["score"].to_numpy()
            )
            eer_rows.append(
                {
                    "source": spec.source,
                    "model": spec.name,
                    "dataset": dataset,
                    "diagnostic_eer": eer,
                    "diagnostic_threshold": threshold,
                    "items": len(scores),
                }
            )

        if spec.source not in entropy_by_dataset:
            continue
        source_entropy = entropy_by_dataset[spec.source]
        for dataset, target_entropy in entropy_by_dataset.items():
            if dataset == spec.source:
                continue
            for layer in sorted(source_entropy["layer"].unique()):
                source_values = source_entropy[source_entropy["layer"] == layer][
                    "entropy_mean"
                ].to_numpy()
                target_values = target_entropy[target_entropy["layer"] == layer][
                    "entropy_mean"
                ].to_numpy()
                ood_rows.append(
                    {
                        "source": spec.source,
                        "model": spec.name,
                        "target_dataset": dataset,
                        "layer": layer,
                        "entropy_ood_auroc": auroc_binary(source_values, target_values),
                        "source_entropy": float(np.mean(source_values)),
                        "target_entropy": float(np.mean(target_values)),
                    }
                )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    profile_summary = pd.concat(profile_summaries, ignore_index=True)
    utilization = pd.concat(utilization_frames, ignore_index=True)
    coactivation = pd.concat(coactivation_frames, ignore_index=True)

    util_summary_rows = []
    utilization_aggregate = (
        utilization.groupby(
            ["source", "model", "dataset", "layer", "label", "expert"],
            as_index=False,
        )[["selected_tokens", "total_tokens"]]
        .sum()
    )
    utilization_aggregate["selection_rate"] = (
        utilization_aggregate["selected_tokens"]
        / utilization_aggregate["total_tokens"].clip(lower=1)
    )
    for keys, group in utilization_aggregate.groupby(
        ["source", "model", "dataset", "layer", "label"]
    ):
        rates = group.sort_values("expert")["selection_rate"].to_numpy()
        util_summary_rows.append(
            {
                "source": keys[0],
                "model": keys[1],
                "dataset": keys[2],
                "layer": keys[3],
                "label": keys[4],
                "gini_selection": gini(rates),
                "cv_selection": float(np.std(rates) / max(np.mean(rates), 1e-12)),
                "dead_expert_ratio": float(np.mean(rates < 1e-6)),
            }
        )
    utilization_summary = pd.DataFrame(util_summary_rows)
    specialization_rows = []
    for keys, group in utilization.groupby(["source", "model", "dataset", "layer"]):
        for metadata in ("attack", "codec", "label"):
            specialization_rows.append(
                {
                    "source": keys[0],
                    "model": keys[1],
                    "dataset": keys[2],
                    "layer": keys[3],
                    "metadata": metadata,
                    "selection_nmi": selection_nmi(group, metadata),
                }
            )

    save_frame(profile_summary, args.output / "routing_summary.csv")
    save_frame(utilization, args.output / "expert_utilization.csv.gz")
    save_frame(utilization_summary, args.output / "utilization_summary.csv")
    save_frame(coactivation, args.output / "expert_coactivation.csv.gz")
    save_frame(
        pd.DataFrame(specialization_rows),
        args.output / "expert_specialization.csv",
    )
    save_frame(pd.DataFrame(eer_rows), args.output / "diagnostic_eer.csv")
    save_frame(pd.DataFrame(ood_rows), args.output / "entropy_ood_auroc.csv")
    save_json(
        {
            "suite": "profile",
            "max_items_per_dataset": args.max_items,
            "seed": args.seed,
            "shard_index": args.shard_index,
            "shard_size": args.shard_size,
            "quick": args.quick,
            "checkpoints": checkpoint_rows,
        },
        args.output / "run.json",
    )
    mark_shard_done(
        args.output,
        {
            "suite": "profile",
            "shard_index": args.shard_index,
            "shard_size": args.shard_size,
        },
    )
    print(f"[profile] results written to {args.output}")


def re_k(column: str) -> bool:
    return column.startswith("k") and column.endswith("_frac")


if __name__ == "__main__":
    main()
