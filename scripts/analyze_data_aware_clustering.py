#!/usr/bin/env python3
"""Summarize the DRIFT resource-only versus data-aware clustering experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METHODS = ("drift", "drift_data_aware")


def _latest_file(root: Path, name: str) -> Path:
    matches = list(root.glob(f"**/{name}"))
    if not matches:
        raise FileNotFoundError(f"Missing {name} below {root}")
    return max(matches, key=lambda path: path.stat().st_mtime)


def _summarize_run(seed: int, method: str, csv_path: Path) -> dict:
    frame = pd.read_csv(csv_path)
    required = {
        "round_id", "num_members", "validation_samples", "train_samples",
        "mean_pairwise_label_jsd", "label_pair_count", "cluster_val_accuracy",
        "mean_tree_edge_label_jsd", "tree_edge_count",
        "cluster_global_accuracy", "final_global_accuracy",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")

    round_rows = []
    for round_id, group in frame.groupby("round_id"):
        val_weights = group["validation_samples"].astype(float)
        train_weights = group["train_samples"].astype(float)
        pair_weights = group["label_pair_count"].astype(float)
        edge_weights = group["tree_edge_count"].astype(float)
        round_rows.append({
            "round_id": int(round_id),
            "num_clusters": int(len(group)),
            "mean_cluster_size": float(group["num_members"].mean()),
            "singleton_cluster_ratio": float((group["num_members"] == 1).mean()),
            "pair_weighted_mean_label_jsd": (
                float((group["mean_pairwise_label_jsd"] * pair_weights).sum()
                      / pair_weights.sum())
                if pair_weights.sum() > 0 else None
            ),
            "edge_weighted_mean_label_jsd": (
                float((group["mean_tree_edge_label_jsd"] * edge_weights).sum()
                      / edge_weights.sum())
                if edge_weights.sum() > 0 else None
            ),
            "weighted_cluster_val_accuracy": float(
                (group["cluster_val_accuracy"] * val_weights).sum()
                / max(1.0, val_weights.sum())
            ),
            "sample_weighted_cluster_global_accuracy": float(
                (group["cluster_global_accuracy"] * train_weights).sum()
                / max(1.0, train_weights.sum())
            ),
            "final_global_accuracy": float(group["final_global_accuracy"].iloc[0]),
        })
    rounds = pd.DataFrame(round_rows).sort_values("round_id")
    return {
        "seed": seed,
        "method": method,
        "diagnostic_file": str(csv_path.resolve()),
        "evaluated_rounds": int(len(rounds)),
        "mean_num_clusters": float(rounds["num_clusters"].mean()),
        "mean_cluster_size": float(rounds["mean_cluster_size"].mean()),
        "mean_singleton_cluster_ratio": float(rounds["singleton_cluster_ratio"].mean()),
        "mean_pair_weighted_label_jsd": float(
            rounds["pair_weighted_mean_label_jsd"].dropna().mean()
        ),
        "mean_edge_weighted_label_jsd": float(
            rounds["edge_weighted_mean_label_jsd"].dropna().mean()
        ),
        "mean_weighted_cluster_val_accuracy": float(
            rounds["weighted_cluster_val_accuracy"].mean()
        ),
        "mean_sample_weighted_cluster_global_accuracy": float(
            rounds["sample_weighted_cluster_global_accuracy"].mean()
        ),
        "final_global_accuracy": float(rounds["final_global_accuracy"].iloc[-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.experiment_root.resolve()
    output_dir = (args.output_dir or root / "analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for seed_dir in sorted(root.glob("seed_*")):
        seed = int(seed_dir.name.removeprefix("seed_"))
        for method in METHODS:
            method_root = seed_dir / "runs" / method
            csv_path = _latest_file(method_root, "cluster_round_metrics.csv")
            records.append(_summarize_run(seed, method, csv_path))

    per_seed = pd.DataFrame(records).sort_values(["seed", "method"])
    per_seed_path = output_dir / "data_aware_per_seed.csv"
    per_seed.to_csv(per_seed_path, index=False)

    numeric = [
        column for column in per_seed.columns
        if column not in {"seed", "method", "diagnostic_file"}
    ]
    aggregate = per_seed.groupby("method")[numeric].agg(["mean", "std"])
    aggregate.columns = [f"{column}_{stat}" for column, stat in aggregate.columns]
    aggregate = aggregate.reset_index()
    aggregate_path = output_dir / "data_aware_aggregate.csv"
    aggregate.to_csv(aggregate_path, index=False)

    resource = per_seed[per_seed["method"] == "drift"].set_index("seed")
    aware = per_seed[per_seed["method"] == "drift_data_aware"].set_index("seed")
    shared_seeds = sorted(set(resource.index) & set(aware.index))
    comparisons = []
    for seed in shared_seeds:
        comparisons.append({
            "seed": int(seed),
            "cluster_val_gain_pp": 100.0 * (
                aware.loc[seed, "mean_weighted_cluster_val_accuracy"]
                - resource.loc[seed, "mean_weighted_cluster_val_accuracy"]
            ),
            "cluster_global_gain_pp": 100.0 * (
                aware.loc[seed, "mean_sample_weighted_cluster_global_accuracy"]
                - resource.loc[seed, "mean_sample_weighted_cluster_global_accuracy"]
            ),
            "final_global_gain_pp": 100.0 * (
                aware.loc[seed, "final_global_accuracy"]
                - resource.loc[seed, "final_global_accuracy"]
            ),
            "label_jsd_change": (
                aware.loc[seed, "mean_pair_weighted_label_jsd"]
                - resource.loc[seed, "mean_pair_weighted_label_jsd"]
            ),
            "tree_edge_jsd_change": (
                aware.loc[seed, "mean_edge_weighted_label_jsd"]
                - resource.loc[seed, "mean_edge_weighted_label_jsd"]
            ),
        })
    comparison_frame = pd.DataFrame(comparisons)
    comparison_path = output_dir / "data_aware_paired_comparison.csv"
    comparison_frame.to_csv(comparison_path, index=False)

    summary = {
        "experiment_root": str(root),
        "seeds": shared_seeds,
        "interpretation": {
            "cluster_val_gain_pp": "positive means data-aware clustering improves cluster-local validation",
            "cluster_global_gain_pp": "positive means cluster models generalize better to the global test set",
            "final_global_gain_pp": "positive means the final GEA model improves",
            "label_jsd_change": "negative means data-aware clusters are more label-homogeneous",
            "tree_edge_jsd_change": "negative means accepted parent-child edges are more label-similar",
        },
        "paired_mean": (
            comparison_frame.mean(numeric_only=True).to_dict()
            if not comparison_frame.empty else {}
        ),
    }
    summary_path = output_dir / "data_aware_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Per-seed: {per_seed_path}")
    print(f"Aggregate: {aggregate_path}")
    print(f"Paired comparison: {comparison_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
