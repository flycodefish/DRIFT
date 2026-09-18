#!/usr/bin/env python3
"""Analyze the fixed-size within-cluster JSD response curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_data_aware_clustering import _latest_file, _summarize_run


METHODS = tuple(f"drift_jsd_q{value:02d}" for value in (0, 25, 50, 75, 100))


def _supplementary_metrics(csv_path: Path) -> dict[str, float]:
    frame = pd.read_csv(csv_path)
    weights = frame["train_samples"].astype(float)
    total = max(1.0, float(weights.sum()))
    result = {
        "mean_sample_weighted_label_coverage": float(
            (frame["pooled_label_coverage"] * weights).sum() / total
        ),
        "mean_sample_weighted_label_entropy": float(
            (frame["pooled_label_entropy"] * weights).sum() / total
        ),
        "mean_sample_weighted_largest_label_share": float(
            (frame["largest_pooled_label_share"] * weights).sum() / total
        ),
    }
    if "cluster_personalized_accuracy" in frame:
        personalized = []
        for _, group in frame.groupby("round_id"):
            valid = group.dropna(subset=["cluster_personalized_accuracy"])
            if valid.empty:
                continue
            validation_weights = valid["validation_samples"].astype(float)
            personalized.append(float(
                (valid["cluster_personalized_accuracy"] * validation_weights).sum()
                / max(1.0, float(validation_weights.sum()))
            ))
        result["mean_weighted_cluster_personalized_accuracy"] = (
            float(np.mean(personalized)) if personalized else float("nan")
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.experiment_root.resolve()
    output_dir = (args.output_dir or root / "jsd_curve_analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for seed_dir in sorted(root.glob("seed_*")):
        seed = int(seed_dir.name.removeprefix("seed_"))
        for method in METHODS:
            diagnostic = _latest_file(
                seed_dir / "runs" / method, "cluster_round_metrics.csv"
            )
            record = _summarize_run(seed, method, diagnostic)
            record.update(_supplementary_metrics(diagnostic))
            record["target_quantile"] = int(method.removeprefix("drift_jsd_q")) / 100.0
            records.append(record)

    per_seed = pd.DataFrame(records).sort_values(["seed", "target_quantile"])
    per_seed_path = output_dir / "jsd_curve_per_seed.csv"
    per_seed.to_csv(per_seed_path, index=False)

    fits = []
    for seed, frame in per_seed.groupby("seed"):
        x = frame["mean_pair_weighted_label_jsd"].to_numpy(dtype=float)
        for metric in (
            "mean_weighted_cluster_val_accuracy",
            "mean_sample_weighted_cluster_global_accuracy",
            "final_global_accuracy",
        ):
            a, b, c = np.polyfit(x, frame[metric].to_numpy(dtype=float), 2)
            vertex = -b / (2.0 * a) if abs(a) > 1e-12 else None
            fits.append({
                "seed": int(seed), "metric": metric,
                "quadratic_a": float(a), "linear_b": float(b), "intercept_c": float(c),
                "concave": bool(a < 0), "vertex_jsd": vertex,
                "vertex_in_observed_range": bool(
                    vertex is not None and float(x.min()) <= vertex <= float(x.max())
                ),
            })
    fits_path = output_dir / "jsd_curve_quadratic_fits.csv"
    pd.DataFrame(fits).to_csv(fits_path, index=False)
    summary_path = output_dir / "jsd_curve_summary.json"
    summary_path.write_text(json.dumps({
        "experiment_root": str(root),
        "methods": list(METHODS),
        "definition": "fixed cluster sizes with achieved within-cluster JSD spanning searched min to max",
        "fits": fits,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Per-seed curve: {per_seed_path}")
    print(f"Quadratic fits: {fits_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
