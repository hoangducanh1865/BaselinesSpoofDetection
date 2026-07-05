#!/usr/bin/env python3
"""Paired rescued/harmed, shared-path, and leave-one-expert-out analyses."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import (
    REPO_ROOT,
    RoutingCollector,
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
    set_shared_scales,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "experiment/manifest.yaml")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/see_molex.yaml")
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "experiment/results/attribution"
    )
    parser.add_argument("--sources", nargs="+", default=["asvspoof5", "asvspoof2019la"])
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument(
        "--expert-datasets",
        nargs="+",
        default=["asvspoof2019la", "dfadd_test", "fake_or_real", "vlsp2025"],
    )
    parser.add_argument("--max-items", type=int, default=8000)
    parser.add_argument("--max-dev-items", type=int, default=20000)
    parser.add_argument("--expert-ablation-items", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def native_policy(name: str, drop_expert: int | None = None) -> dict:
    policy = {"mode": "entropy"} if name in {"M2", "M3"} else {"mode": "fixed", "k": 4}
    if drop_expert is not None:
        policy["drop_expert"] = drop_expert
    return policy


def attach_routing_features(scores: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    features = (
        profile.groupby("utt_id", as_index=False)[
            ["entropy_mean", "active_mean", "pmax_mean", "margin_mean"]
        ]
        .mean()
    )
    return scores.merge(features, on="utt_id", how="left")


def decision_groups(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_threshold: float,
    right_threshold: float,
    left_name: str,
    right_name: str,
) -> pd.DataFrame:
    merged = left.merge(right, on=["utt_id", "label"], suffixes=("_left", "_right"))
    left_correct = (merged["score_left"] >= left_threshold).astype(int) == merged["label"]
    right_correct = (merged["score_right"] >= right_threshold).astype(int) == merged["label"]
    merged["group"] = np.select(
        [
            left_correct & right_correct,
            ~left_correct & right_correct,
            left_correct & ~right_correct,
        ],
        ["both_correct", "rescued", "harmed"],
        default="both_wrong",
    )
    merged["left_model"] = left_name
    merged["right_model"] = right_name
    return merged


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    datasets = args.datasets or manifest["datasets"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)

    native_scores: dict[tuple[str, str, str], pd.DataFrame] = {}
    thresholds: dict[tuple[str, str], float] = {}
    threshold_rows = []
    shared_rows = []
    shared_scores = []
    expert_rows = []
    checkpoints = []

    for spec in manifest_model_specs(manifest, args.sources, ["M0", "M2", "M3"]):
        print(f"[attribution] loading {spec.source}/{spec.name}: {spec.checkpoint}")
        model = load_model(spec, args.config, device)
        set_policy(model, native_policy(spec.name))
        checkpoints.append(
            {"source": spec.source, "model": spec.name, "checkpoint": str(spec.checkpoint)}
        )

        source_dataset = manifest["sources"][spec.source]["source_dataset"]
        dev_loader, _ = build_loader(
            source_dataset,
            args.config,
            "validation",
            args.max_dev_items,
            args.seed,
            args.batch_size,
            args.num_workers,
        )
        dev_scores, _ = evaluate(
            model, dev_loader, device, desc=f"dev threshold {spec.source}/{spec.name}"
        )
        dev_eer, dev_threshold = eer_and_threshold(
            dev_scores["label"].to_numpy(), dev_scores["score"].to_numpy()
        )
        thresholds[(spec.source, spec.name)] = dev_threshold
        threshold_rows.append(
            {
                "source": spec.source,
                "model": spec.name,
                "dev_eer": dev_eer,
                "dev_threshold": dev_threshold,
                "dev_items": len(dev_scores),
            }
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
            collector = None
            if spec.name in {"M2", "M3"}:
                collector = RoutingCollector(model, spec.source, spec.name, dataset)
            scores, _ = evaluate(
                model,
                loader,
                device,
                collector=collector,
                desc=f"native {spec.source}/{spec.name}/{dataset}",
            )
            scores = scores.merge(
                sampled[["utt_id", "attack", "codec"]], on="utt_id", how="left"
            )
            if collector is not None:
                collector.close()
                profile, _, _ = collector.frames()
                save_frame(
                    profile,
                    args.output
                    / "details"
                    / f"{spec.source}__{spec.name}__{dataset}__routing.csv.gz",
                )
                scores = attach_routing_features(scores, profile)
            native_scores[(spec.source, spec.name, dataset)] = scores

            if spec.name == "M3":
                native_eer, native_threshold = eer_and_threshold(
                    scores["label"].to_numpy(), scores["score"].to_numpy()
                )
                shared_rows.append(
                    {
                        "source": spec.source,
                        "dataset": dataset,
                        "intervention": "native",
                        "lambda_s": 1.0,
                        "lambda_r": 1.0,
                        "diagnostic_eer": native_eer,
                        "threshold": native_threshold,
                        "items": len(scores),
                    }
                )
                interventions = [
                    ("shared_off", 0.0, 1.0),
                    ("shared_half", 0.5, 1.0),
                    ("routed_off", 1.0, 0.0),
                ]
                for intervention, lambda_s, lambda_r in interventions:
                    set_shared_scales(model, lambda_s=lambda_s, lambda_r=lambda_r)
                    intervention_scores, _ = evaluate(
                        model,
                        loader,
                        device,
                        desc=f"shared {spec.source}/{intervention}/{dataset}",
                    )
                    eer, threshold = eer_and_threshold(
                        intervention_scores["label"].to_numpy(),
                        intervention_scores["score"].to_numpy(),
                    )
                    shared_rows.append(
                        {
                            "source": spec.source,
                            "dataset": dataset,
                            "intervention": intervention,
                            "lambda_s": lambda_s,
                            "lambda_r": lambda_r,
                            "diagnostic_eer": eer,
                            "threshold": threshold,
                            "items": len(intervention_scores),
                        }
                    )
                    intervention_scores.insert(0, "intervention", intervention)
                    intervention_scores.insert(0, "dataset", dataset)
                    intervention_scores.insert(0, "source", spec.source)
                    shared_scores.append(intervention_scores)
                set_shared_scales(model, lambda_s=1.0, lambda_r=1.0)

        if spec.name == "M2" and args.expert_ablation_items > 0:
            num_experts = next(
                layer.smoe.num_experts
                for layer in model.ssl_model.encoder.layers
                if hasattr(layer, "smoe")
            )
            for dataset_index, dataset in enumerate(args.expert_datasets):
                loader, _ = build_loader(
                    dataset,
                    args.config,
                    "evaluation",
                    args.expert_ablation_items,
                    args.seed + dataset_index,
                    args.batch_size,
                    args.num_workers,
                )
                set_policy(model, native_policy("M2"))
                base_scores, _ = evaluate(
                    model, loader, device, desc=f"expert base {spec.source}/{dataset}"
                )
                base_eer, _ = eer_and_threshold(
                    base_scores["label"].to_numpy(), base_scores["score"].to_numpy()
                )
                for expert in range(num_experts):
                    set_policy(model, native_policy("M2", drop_expert=expert))
                    dropped, _ = evaluate(
                        model,
                        loader,
                        device,
                        desc=f"drop E{expert} {spec.source}/{dataset}",
                    )
                    drop_eer, _ = eer_and_threshold(
                        dropped["label"].to_numpy(), dropped["score"].to_numpy()
                    )
                    expert_rows.append(
                        {
                            "source": spec.source,
                            "dataset": dataset,
                            "expert": expert,
                            "base_eer": base_eer,
                            "drop_eer": drop_eer,
                            "delta_drop_minus_base": drop_eer - base_eer,
                            "items": len(dropped),
                        }
                    )
            set_policy(model, native_policy("M2"))

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    group_frames = []
    group_summary_rows = []
    metadata_group_rows = []
    bootstrap_rows = []
    for source in args.sources:
        for dataset in datasets:
            for left_name, right_name in [("M0", "M2"), ("M2", "M3")]:
                left = native_scores[(source, left_name, dataset)]
                right = native_scores[(source, right_name, dataset)]
                grouped = decision_groups(
                    left,
                    right,
                    thresholds[(source, left_name)],
                    thresholds[(source, right_name)],
                    left_name,
                    right_name,
                )
                grouped.insert(0, "dataset", dataset)
                grouped.insert(0, "source", source)
                group_frames.append(grouped)
                for group_name, group in grouped.groupby("group"):
                    group_summary_rows.append(
                        {
                            "source": source,
                            "dataset": dataset,
                            "comparison": f"{left_name}->{right_name}",
                            "group": group_name,
                            "items": len(group),
                            "fraction": len(group) / max(len(grouped), 1),
                            "right_entropy_mean": group.get(
                                "entropy_mean_right", pd.Series(dtype=float)
                            ).mean(),
                            "right_active_mean": group.get(
                                "active_mean_right", pd.Series(dtype=float)
                            ).mean(),
                            "right_margin_mean": group.get(
                                "margin_mean_right", pd.Series(dtype=float)
                            ).mean(),
                        }
                    )
                    for metadata_name in ("attack_right", "codec_right"):
                        if metadata_name not in group:
                            continue
                        counts = group[metadata_name].fillna("unknown").value_counts()
                        for value, count in counts.items():
                            metadata_group_rows.append(
                                {
                                    "source": source,
                                    "dataset": dataset,
                                    "comparison": f"{left_name}->{right_name}",
                                    "group": group_name,
                                    "metadata": metadata_name.removesuffix("_right"),
                                    "value": value,
                                    "items": int(count),
                                    "within_group_fraction": float(count / len(group)),
                                }
                            )
                bootstrap = bootstrap_eer_delta(
                    left[["utt_id", "label", "score"]],
                    right[["utt_id", "label", "score"]],
                    repeats=args.bootstrap,
                    seed=args.seed,
                )
                bootstrap.update(
                    {
                        "source": source,
                        "dataset": dataset,
                        "comparison": f"{left_name}->{right_name}",
                    }
                )
                bootstrap_rows.append(bootstrap)

    all_native = []
    for (source, model_name, dataset), frame in native_scores.items():
        value = frame.copy()
        value.insert(0, "dataset", dataset)
        value.insert(0, "model", model_name)
        value.insert(0, "source", source)
        all_native.append(value)

    save_frame(pd.concat(all_native, ignore_index=True), args.output / "native_predictions.csv.gz")
    save_frame(pd.concat(group_frames, ignore_index=True), args.output / "rescued_harmed.csv.gz")
    save_frame(pd.DataFrame(group_summary_rows), args.output / "rescued_harmed_summary.csv")
    save_frame(
        pd.DataFrame(metadata_group_rows),
        args.output / "rescued_harmed_metadata.csv",
    )
    save_frame(pd.DataFrame(bootstrap_rows), args.output / "native_bootstrap.csv")
    save_frame(pd.DataFrame(threshold_rows), args.output / "source_dev_thresholds.csv")
    save_frame(pd.DataFrame(shared_rows), args.output / "shared_path_summary.csv")
    if shared_scores:
        save_frame(
            pd.concat(shared_scores, ignore_index=True),
            args.output / "shared_path_predictions.csv.gz",
        )
    save_frame(pd.DataFrame(expert_rows), args.output / "expert_leave_one_out.csv")
    save_json(
        {
            "suite": "attribution",
            "max_items_per_dataset": args.max_items,
            "max_dev_items": args.max_dev_items,
            "expert_ablation_items": args.expert_ablation_items,
            "seed": args.seed,
            "bootstrap_repeats": args.bootstrap,
            "checkpoints": checkpoints,
        },
        args.output / "run.json",
    )
    print(f"[attribution] results written to {args.output}")


if __name__ == "__main__":
    main()
