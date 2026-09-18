"""Validate and summarize Experiment 3 resource-awareness results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


H_LEVELS = (1, 5, 10)
METHODS = (
    "fixed_hierarchical", "resource_tree",
    "drift_no_resource_awareness", "drift",
)
LABELS = {
    "fixed_hierarchical": "Fixed HierFL",
    "resource_tree": "Resource-Tree (Oracle)",
    "drift_no_resource_awareness": "DRIFT-NoResource",
    "drift": "DRIFT",
}
METRICS = (
    "final_accuracy", "mean_round_latency_sim_ms",
    "p95_round_latency_sim_ms", "max_capacity_normalized_load",
    "capacity_normalized_load_imbalance", "max_peak_node_traffic_bytes",
    "total_bytes", "total_messages", "aggregation_load_imbalance",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_and_validate(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for heterogeneity in H_LEVELS:
        path = root / f"e3_resource_h{heterogeneity}" / "matrix_summary.csv"
        if not path.is_file():
            raise ValueError(f"Missing matrix summary: {path}")
        rows = _read_csv(path)
        for method in METHODS:
            selected = [row for row in rows if row.get("method") == method]
            if len(selected) != 3:
                raise ValueError(
                    f"H={heterogeneity} {method}: expected 3 runs, found {len(selected)}"
                )
            if {int(row["seed"]) for row in selected} != {42, 43, 44}:
                raise ValueError(f"H={heterogeneity} {method}: incorrect seeds")
            for row in selected:
                if row.get("status", "").upper() != "COMPLETE" or row.get("valid") != "True":
                    raise ValueError(f"Invalid run: {row.get('run')}")
                if int(row["rounds"]) != 10:
                    raise ValueError(f"Incorrect round count: {row.get('run')}")
                record: dict[str, Any] = {
                    "heterogeneity": heterogeneity,
                    "method": method,
                    "seed": int(row["seed"]),
                    "run": row["run"],
                }
                for metric in METRICS:
                    if row.get(metric) in (None, ""):
                        raise ValueError(f"Missing {metric}: {row.get('run')}")
                    record[metric] = float(row[metric])
                records.append(record)

    for heterogeneity in H_LEVELS:
        for seed in (42, 43, 44):
            accuracy = {
                record["final_accuracy"] for record in records
                if record["heterogeneity"] == heterogeneity and record["seed"] == seed
            }
            if len(accuracy) != 1:
                raise ValueError(f"Learning semantics differ at H={heterogeneity}, seed={seed}")
    return records


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for heterogeneity in H_LEVELS:
        for method in METHODS:
            selected = [
                record for record in records
                if record["heterogeneity"] == heterogeneity and record["method"] == method
            ]
            row: dict[str, Any] = {
                "heterogeneity": heterogeneity,
                "method": method,
                "label": LABELS[method],
                "valid_runs": len(selected),
            }
            for metric in METRICS:
                values = [record[metric] for record in selected]
                row[f"{metric}_mean"] = statistics.fmean(values)
                row[f"{metric}_std"] = statistics.stdev(values)
            result.append(row)
    return result


def write_csv(path: Path, summary: list[dict[str, Any]]) -> None:
    fields = list(summary[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)


def plot(summary: list[dict[str, Any]], path: Path) -> None:
    panels = (
        ("mean_round_latency_sim_ms", 1e-3, "Mean round latency (s)"),
        ("max_capacity_normalized_load", 1.0, "Max normalized aggregation load"),
        ("capacity_normalized_load_imbalance", 1.0, "Normalized load imbalance"),
    )
    colors = ("#f58518", "#54a24b", "#9d755d", "#e45756")
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
    for axis, (metric, scale, ylabel) in zip(axes, panels):
        for method, color in zip(METHODS, colors):
            selected = [row for row in summary if row["method"] == method]
            means = [row[f"{metric}_mean"] * scale for row in selected]
            stds = [row[f"{metric}_std"] * scale for row in selected]
            axis.errorbar(H_LEVELS, means, yerr=stds, marker="o", capsize=3,
                          linewidth=2, color=color, label=LABELS[method])
        axis.set_xticks(H_LEVELS)
        axis.set_xlabel("Compute heterogeneity H")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, default=Path("outputs/formal_v1"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output = args.output_dir or args.formal_root / "e3_resource_awareness" / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    records = load_and_validate(args.formal_root)
    summary = summarize(records)
    write_csv(output / "analysis_summary.csv", summary)
    (output / "analysis_summary.json").write_text(
        json.dumps({"schema_version": 1, "records": records, "aggregate": summary},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plot(summary, output / "resource_awareness_trends.png")
    print(f"Experiment 3 analysis written to: {output.resolve()}")


if __name__ == "__main__":
    main()
