#!/usr/bin/env python3
"""Summarize size-matched distributed, random, and Oracle cluster assignments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from analyze_data_aware_clustering import _latest_file, _summarize_run


METHODS = ("drift", "drift_random_matched", "drift_oracle_data_similar")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.experiment_root.resolve()
    output_dir = (args.output_dir or root / "oracle_analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for seed_dir in sorted(root.glob("seed_*")):
        seed = int(seed_dir.name.removeprefix("seed_"))
        for method in METHODS:
            diagnostic = _latest_file(
                seed_dir / "runs" / method, "cluster_round_metrics.csv"
            )
            records.append(_summarize_run(seed, method, diagnostic))

    per_seed = pd.DataFrame(records).sort_values(["seed", "method"])
    per_seed_path = output_dir / "oracle_per_seed.csv"
    per_seed.to_csv(per_seed_path, index=False)

    numeric = [
        column for column in per_seed.columns
        if column not in {"seed", "method", "diagnostic_file"}
    ]
    aggregate = per_seed.groupby("method")[numeric].agg(["mean", "std"])
    aggregate.columns = [f"{column}_{stat}" for column, stat in aggregate.columns]
    aggregate = aggregate.reset_index()
    aggregate_path = output_dir / "oracle_aggregate.csv"
    aggregate.to_csv(aggregate_path, index=False)

    indexed = {
        method: per_seed[per_seed["method"] == method].set_index("seed")
        for method in METHODS
    }
    shared = sorted(set.intersection(*(set(frame.index) for frame in indexed.values())))
    comparisons = []
    for seed in shared:
        oracle = indexed["drift_oracle_data_similar"].loc[seed]
        for control in ("drift", "drift_random_matched"):
            baseline = indexed[control].loc[seed]
            comparisons.append({
                "seed": int(seed),
                "control": control,
                "oracle_cluster_val_gain_pp": 100.0 * (
                    oracle["mean_weighted_cluster_val_accuracy"]
                    - baseline["mean_weighted_cluster_val_accuracy"]
                ),
                "oracle_cluster_global_gain_pp": 100.0 * (
                    oracle["mean_sample_weighted_cluster_global_accuracy"]
                    - baseline["mean_sample_weighted_cluster_global_accuracy"]
                ),
                "oracle_final_global_gain_pp": 100.0 * (
                    oracle["final_global_accuracy"] - baseline["final_global_accuracy"]
                ),
                "oracle_pairwise_jsd_change": (
                    oracle["mean_pair_weighted_label_jsd"]
                    - baseline["mean_pair_weighted_label_jsd"]
                ),
            })
    comparison = pd.DataFrame(comparisons)
    comparison_path = output_dir / "oracle_paired_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    summary = {
        "experiment_root": str(root),
        "seeds": shared,
        "methods": list(METHODS),
        "oracle_definition": (
            "train-label-only multi-start pair-swap minimization of within-cluster "
            "pairwise natural-log JSD under the distributed DRIFT cluster-size profile"
        ),
        "comparisons": comparisons,
    }
    summary_path = output_dir / "oracle_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Per-seed: {per_seed_path}")
    print(f"Aggregate: {aggregate_path}")
    print(f"Paired comparison: {comparison_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
