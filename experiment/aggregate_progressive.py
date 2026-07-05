#!/usr/bin/env python3
"""Aggregate completed progressive shards into cumulative experiment outputs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    REPO_ROOT,
    auroc_binary,
    bootstrap_eer_delta,
    eer_and_threshold,
    gini,
    save_frame,
)
from run_attribution import decision_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT / "experiment/results_progressive",
    )
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def completed_shards(suite_root: Path) -> list[Path]:
    return sorted(
        path
        for path in suite_root.glob("shard_*")
        if path.is_dir() and (path / "DONE.json").exists()
    )


def read_many(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            frames.append(pd.read_csv(path))
        except pd.errors.EmptyDataError:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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


def aggregate_profile(root: Path) -> None:
    suite = root / "profile"
    shards = completed_shards(suite)
    routing = read_many(
        [path for shard in shards for path in (shard / "details").glob("*__routing.csv.gz")]
    )
    predictions = read_many(
        [
            path
            for shard in shards
            for path in (shard / "details").glob("*__predictions.csv.gz")
        ]
    )
    utilization = read_many([shard / "expert_utilization.csv.gz" for shard in shards])
    coactivation = read_many([shard / "expert_coactivation.csv.gz" for shard in shards])
    if routing.empty:
        return

    routing = routing.drop_duplicates(
        ["source", "model", "dataset", "utt_id", "layer"], keep="last"
    )
    predictions = predictions.drop_duplicates(
        ["source", "model", "dataset", "utt_id"], keep="last"
    )
    metrics = [
        "entropy_mean",
        "entropy_std",
        "active_mean",
        "total_active_mean",
        "active_std",
        "active_p90",
        "pmax_mean",
        "margin_mean",
    ] + [column for column in routing if column.startswith("k") and column.endswith("_frac")]
    routing_summary = routing.groupby(
        ["source", "model", "dataset", "layer"], as_index=False
    )[metrics].mean()

    util_group = [
        "source",
        "model",
        "dataset",
        "layer",
        "expert",
        "label",
        "attack",
        "codec",
    ]
    utilization = utilization.groupby(util_group, as_index=False)[
        ["selected_tokens", "total_tokens"]
    ].sum()
    utilization["selection_rate"] = (
        utilization["selected_tokens"] / utilization["total_tokens"].clip(lower=1)
    )
    util_summary_rows = []
    aggregate = utilization.groupby(
        ["source", "model", "dataset", "layer", "label", "expert"], as_index=False
    )[["selected_tokens", "total_tokens"]].sum()
    aggregate["selection_rate"] = (
        aggregate["selected_tokens"] / aggregate["total_tokens"].clip(lower=1)
    )
    for keys, group in aggregate.groupby(["source", "model", "dataset", "layer", "label"]):
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

    specialization = []
    for keys, group in utilization.groupby(["source", "model", "dataset", "layer"]):
        for metadata in ("attack", "codec", "label"):
            specialization.append(
                {
                    "source": keys[0],
                    "model": keys[1],
                    "dataset": keys[2],
                    "layer": keys[3],
                    "metadata": metadata,
                    "selection_nmi": selection_nmi(group, metadata),
                }
            )

    eer_rows = []
    for keys, group in predictions.groupby(["source", "model", "dataset"]):
        eer, threshold = eer_and_threshold(group["label"], group["score"])
        eer_rows.append(
            {
                "source": keys[0],
                "model": keys[1],
                "dataset": keys[2],
                "diagnostic_eer": eer,
                "diagnostic_threshold": threshold,
                "items": len(group),
            }
        )

    ood_rows = []
    for (source, model, layer), group in routing.groupby(["source", "model", "layer"]):
        source_values = group[group["dataset"] == source]["entropy_mean"].to_numpy()
        for dataset, target in group.groupby("dataset"):
            if dataset == source or not len(source_values):
                continue
            target_values = target["entropy_mean"].to_numpy()
            ood_rows.append(
                {
                    "source": source,
                    "model": model,
                    "target_dataset": dataset,
                    "layer": layer,
                    "entropy_ood_auroc": auroc_binary(source_values, target_values),
                    "source_entropy": float(np.mean(source_values)),
                    "target_entropy": float(np.mean(target_values)),
                }
            )

    if not coactivation.empty:
        coactivation = coactivation.groupby(
            ["source", "model", "dataset", "layer", "expert_i", "expert_j"],
            as_index=False,
        )["coactive_tokens"].sum()
    save_frame(routing_summary, suite / "routing_summary.csv")
    save_frame(utilization, suite / "expert_utilization.csv.gz")
    save_frame(pd.DataFrame(util_summary_rows), suite / "utilization_summary.csv")
    save_frame(pd.DataFrame(specialization), suite / "expert_specialization.csv")
    save_frame(coactivation, suite / "expert_coactivation.csv.gz")
    save_frame(pd.DataFrame(eer_rows), suite / "diagnostic_eer.csv")
    save_frame(pd.DataFrame(ood_rows), suite / "entropy_ood_auroc.csv")


def aggregate_counterfactual(root: Path, repeats: int, seed: int) -> None:
    suite = root / "counterfactual"
    shards = completed_shards(suite)
    predictions = read_many(
        [
            path
            for shard in shards
            for path in (shard / "details").glob("*__predictions.csv.gz")
        ]
    )
    active = read_many([shard / "active_count_by_layer.csv" for shard in shards])
    if predictions.empty:
        return
    predictions = predictions.drop_duplicates(
        ["source", "dataset", "weights", "policy", "utt_id"], keep="last"
    )
    summary = []
    for keys, group in predictions.groupby(["source", "dataset", "weights", "policy"]):
        eer, threshold = eer_and_threshold(group["label"], group["score"])
        active_group = active[
            (active["source"] == keys[0])
            & (active["dataset"] == keys[1])
            & (active["weights"] == keys[2])
            & (active["policy"] == keys[3])
        ]
        weighted_active = (
            (active_group["active_mean"] * active_group["tokens"]).sum()
            / max(active_group["tokens"].sum(), 1)
        )
        summary.append(
            {
                "source": keys[0],
                "dataset": keys[1],
                "weights": keys[2],
                "policy": keys[3],
                "diagnostic_eer": eer,
                "threshold": threshold,
                "items": len(group),
                "mean_active_experts": weighted_active,
            }
        )

    comparisons = []
    for (source, dataset), subset in predictions.groupby(["source", "dataset"]):
        variants = {
            (weights, policy): group[["utt_id", "label", "score"]]
            for (weights, policy), group in subset.groupby(["weights", "policy"])
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
            row = bootstrap_eer_delta(variants[left_key], variants[right_key], repeats, seed)
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
    save_frame(pd.DataFrame(summary), suite / "policy_summary.csv")
    save_frame(active, suite / "active_count_by_layer.csv")
    save_frame(pd.DataFrame(comparisons), suite / "counterfactual_bootstrap.csv")


def aggregate_attribution(root: Path, repeats: int, seed: int) -> None:
    suite = root / "attribution"
    shards = completed_shards(suite)
    native = read_many([shard / "native_predictions.csv.gz" for shard in shards])
    thresholds = read_many([shard / "source_dev_thresholds.csv" for shard in shards])
    shared = read_many([shard / "shared_path_predictions.csv.gz" for shard in shards])
    experts = read_many([shard / "expert_leave_one_out.csv" for shard in shards])
    if native.empty:
        return
    native = native.drop_duplicates(["source", "model", "dataset", "utt_id"], keep="last")
    thresholds = thresholds.drop_duplicates(["source", "model"], keep="last")
    threshold_map = {
        (row.source, row.model): float(row.dev_threshold)
        for row in thresholds.itertuples(index=False)
    }

    groups = []
    group_summary = []
    metadata_rows = []
    bootstrap_rows = []
    for (source, dataset), subset in native.groupby(["source", "dataset"]):
        models = {
            model: group.drop(columns=["source", "model", "dataset"])
            for model, group in subset.groupby("model")
        }
        for left_name, right_name in (("M0", "M2"), ("M2", "M3")):
            if left_name not in models or right_name not in models:
                continue
            grouped = decision_groups(
                models[left_name],
                models[right_name],
                threshold_map[(source, left_name)],
                threshold_map[(source, right_name)],
                left_name,
                right_name,
            )
            grouped.insert(0, "dataset", dataset)
            grouped.insert(0, "source", source)
            groups.append(grouped)
            for group_name, value in grouped.groupby("group"):
                group_summary.append(
                    {
                        "source": source,
                        "dataset": dataset,
                        "comparison": f"{left_name}->{right_name}",
                        "group": group_name,
                        "items": len(value),
                        "fraction": len(value) / len(grouped),
                        "right_entropy_mean": value.get(
                            "entropy_mean_right", pd.Series(dtype=float)
                        ).mean(),
                        "right_active_mean": value.get(
                            "active_mean_right", pd.Series(dtype=float)
                        ).mean(),
                        "right_margin_mean": value.get(
                            "margin_mean_right", pd.Series(dtype=float)
                        ).mean(),
                    }
                )
                for column in ("attack_right", "codec_right"):
                    if column not in value:
                        continue
                    counts = value[column].fillna("unknown").value_counts()
                    for metadata_value, count in counts.items():
                        metadata_rows.append(
                            {
                                "source": source,
                                "dataset": dataset,
                                "comparison": f"{left_name}->{right_name}",
                                "group": group_name,
                                "metadata": column.removesuffix("_right"),
                                "value": metadata_value,
                                "items": int(count),
                                "within_group_fraction": count / len(value),
                            }
                        )
            row = bootstrap_eer_delta(
                models[left_name][["utt_id", "label", "score"]],
                models[right_name][["utt_id", "label", "score"]],
                repeats,
                seed,
            )
            row.update(
                {
                    "source": source,
                    "dataset": dataset,
                    "comparison": f"{left_name}->{right_name}",
                }
            )
            bootstrap_rows.append(row)

    shared_summary = []
    if not shared.empty:
        shared = shared.drop_duplicates(
            ["source", "dataset", "intervention", "utt_id"], keep="last"
        )
        for keys, group in shared.groupby(["source", "dataset", "intervention"]):
            eer, threshold = eer_and_threshold(group["label"], group["score"])
            shared_summary.append(
                {
                    "source": keys[0],
                    "dataset": keys[1],
                    "intervention": keys[2],
                    "diagnostic_eer": eer,
                    "threshold": threshold,
                    "items": len(group),
                }
            )
    native_m3 = native[native["model"] == "M3"]
    for keys, group in native_m3.groupby(["source", "dataset"]):
        eer, threshold = eer_and_threshold(group["label"], group["score"])
        shared_summary.append(
            {
                "source": keys[0],
                "dataset": keys[1],
                "intervention": "native",
                "diagnostic_eer": eer,
                "threshold": threshold,
                "items": len(group),
            }
        )

    save_frame(native, suite / "native_predictions.csv.gz")
    save_frame(pd.concat(groups, ignore_index=True), suite / "rescued_harmed.csv.gz")
    save_frame(pd.DataFrame(group_summary), suite / "rescued_harmed_summary.csv")
    save_frame(pd.DataFrame(metadata_rows), suite / "rescued_harmed_metadata.csv")
    save_frame(pd.DataFrame(bootstrap_rows), suite / "native_bootstrap.csv")
    save_frame(thresholds, suite / "source_dev_thresholds.csv")
    save_frame(pd.DataFrame(shared_summary), suite / "shared_path_summary.csv")
    save_frame(experts, suite / "expert_leave_one_out.csv")


def main() -> None:
    args = parse_args()
    aggregate_profile(args.root)
    aggregate_counterfactual(args.root, args.bootstrap, args.seed)
    aggregate_attribution(args.root, args.bootstrap, args.seed)
    counts = {
        suite: len(completed_shards(args.root / suite))
        for suite in ("profile", "counterfactual", "attribution")
    }
    print("Aggregated completed shards:", counts)


if __name__ == "__main__":
    main()
