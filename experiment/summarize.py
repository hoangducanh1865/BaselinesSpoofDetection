#!/usr/bin/env python3
"""Build one human-readable summary from all explainability artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "experiment/results")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def markdown_table(frame: pd.DataFrame, decimals: int = 4) -> str:
    if frame.empty:
        return "_No results._"
    display = frame.copy()
    numeric = display.select_dtypes(include="number").columns
    display[numeric] = display[numeric].round(decimals)
    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def profile_section(root: Path) -> str:
    routing = read_csv(root / "profile/routing_summary.csv")
    utilization = read_csv(root / "profile/utilization_summary.csv")
    specialization = read_csv(root / "profile/expert_specialization.csv")
    ood = read_csv(root / "profile/entropy_ood_auroc.csv")
    if routing.empty:
        return "## Routing Profile\n\n_No profile output found._"

    aggregate = (
        routing.groupby(["source", "model", "dataset"], as_index=False)[
            [
                "entropy_mean",
                "active_mean",
                "total_active_mean",
                "pmax_mean",
                "margin_mean",
            ]
        ]
        .mean()
    )
    util = (
        utilization.groupby(["source", "model", "dataset"], as_index=False)[
            ["gini_selection", "cv_selection", "dead_expert_ratio"]
        ]
        .mean()
        if not utilization.empty
        else utilization
    )
    auroc = (
        ood.groupby(["source", "model", "target_dataset"], as_index=False)[
            "entropy_ood_auroc"
        ]
        .mean()
        .sort_values(["source", "model", "target_dataset"])
        if not ood.empty
        else ood
    )
    text = ["## Routing Profile", "", markdown_table(aggregate)]
    if not util.empty:
        text.extend(["", "### Utilization Balance", "", markdown_table(util)])
    if not specialization.empty:
        specialization_summary = (
            specialization.groupby(
                ["source", "model", "dataset", "metadata"], as_index=False
            )["selection_nmi"]
            .mean()
        )
        text.extend(
            [
                "",
                "### Expert Specialization (NMI)",
                "",
                markdown_table(specialization_summary),
            ]
        )
    if not auroc.empty:
        text.extend(["", "### Entropy As OOD Diagnostic", "", markdown_table(auroc)])
    return "\n".join(text)


def counterfactual_section(root: Path) -> str:
    summary = read_csv(root / "counterfactual/policy_summary.csv")
    bootstrap = read_csv(root / "counterfactual/counterfactual_bootstrap.csv")
    if summary.empty:
        return "## Counterfactuals\n\n_No counterfactual output found._"

    core = summary[
        ((summary["weights"] == "M0") & summary["policy"].isin(["fixed_k4", "entropy_adaptive"]))
        | ((summary["weights"] == "M2") & summary["policy"].isin(["fixed_k4", "entropy_adaptive"]))
    ][
        [
            "source",
            "dataset",
            "weights",
            "policy",
            "diagnostic_eer",
            "mean_active_experts",
        ]
    ]
    matched = summary[summary["weights"] == "M2"][
        [
            "source",
            "dataset",
            "policy",
            "diagnostic_eer",
            "mean_active_experts",
        ]
    ]
    text = [
        "## Counterfactuals",
        "",
        "### Core 2x2",
        "",
        markdown_table(core),
        "",
        "### Matched-Compute Controls",
        "",
        markdown_table(matched),
    ]
    if not bootstrap.empty:
        significant = bootstrap[
            (bootstrap["ci95_low"] > 0) | (bootstrap["ci95_high"] < 0)
        ][
            [
                "source",
                "dataset",
                "question",
                "delta_right_minus_left",
                "ci95_low",
                "ci95_high",
            ]
        ]
        text.extend(
            ["", "### Significant Paired Differences", "", markdown_table(significant)]
        )
    return "\n".join(text)


def attribution_section(root: Path) -> str:
    groups = read_csv(root / "attribution/rescued_harmed_summary.csv")
    metadata = read_csv(root / "attribution/rescued_harmed_metadata.csv")
    shared = read_csv(root / "attribution/shared_path_summary.csv")
    experts = read_csv(root / "attribution/expert_leave_one_out.csv")
    bootstrap = read_csv(root / "attribution/native_bootstrap.csv")
    text = ["## Attribution"]
    if groups.empty:
        text.extend(["", "_No attribution output found._"])
        return "\n".join(text)

    rescued_harmed = groups[groups["group"].isin(["rescued", "harmed"])][
        [
            "source",
            "dataset",
            "comparison",
            "group",
            "items",
            "fraction",
            "right_entropy_mean",
            "right_active_mean",
            "right_margin_mean",
        ]
    ]
    text.extend(["", "### Rescued Versus Harmed", "", markdown_table(rescued_harmed)])
    if not metadata.empty:
        top_metadata = (
            metadata.sort_values(
                ["source", "dataset", "comparison", "group", "items"],
                ascending=[True, True, True, True, False],
            )
            .groupby(
                ["source", "dataset", "comparison", "group", "metadata"],
                as_index=False,
            )
            .head(3)
        )
        text.extend(
            [
                "",
                "### Dominant Attack/Codec Groups",
                "",
                markdown_table(top_metadata),
            ]
        )

    if not bootstrap.empty:
        text.extend(
            [
                "",
                "### Native Model Paired Bootstrap",
                "",
                markdown_table(
                    bootstrap[
                        [
                            "source",
                            "dataset",
                            "comparison",
                            "delta_right_minus_left",
                            "ci95_low",
                            "ci95_high",
                        ]
                    ]
                ),
            ]
        )
    if not shared.empty:
        text.extend(
            [
                "",
                "### Shared-Path Interventions",
                "",
                markdown_table(
                    shared[
                        [
                            "source",
                            "dataset",
                            "intervention",
                            "diagnostic_eer",
                            "items",
                        ]
                    ]
                ),
            ]
        )
    if not experts.empty:
        top = (
            experts.sort_values(
                ["source", "dataset", "delta_drop_minus_base"], ascending=[True, True, False]
            )
            .groupby(["source", "dataset"], as_index=False)
            .head(3)
        )
        text.extend(
            [
                "",
                "### Most Causally Important Experts",
                "",
                markdown_table(
                    top[
                        [
                            "source",
                            "dataset",
                            "expert",
                            "base_eer",
                            "drop_eer",
                            "delta_drop_minus_base",
                        ]
                    ]
                ),
            ]
        )
    return "\n".join(text)


def main() -> None:
    args = parse_args()
    output = args.output or args.root / "summary.md"
    sections = [
        "# SEE-MoLEx Explainability Summary",
        "",
        "Diagnostic EERs below use the fixed stratified samples recorded in each run.",
        "They complement, but do not replace, the full-dataset EER table.",
        "",
        profile_section(args.root),
        "",
        counterfactual_section(args.root),
        "",
        attribution_section(args.root),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(sections) + "\n")
    print(output.read_text())
    print(f"\nSummary written to {output}")


if __name__ == "__main__":
    main()
