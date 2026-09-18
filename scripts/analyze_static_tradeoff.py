"""Create reproducible tables and figures for the static trade-off matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_ORDER = (
    "fedavg", "fedavg_edge", "dpsgd", "fixed_hierarchical",
    "resource_tree", "epidemic_learning", "drift",
)
LABELS = {
    "fedavg": "FedAvg-Ideal",
    "fedavg_edge": "FedAvg-Edge",
    "dpsgd": "D-PSGD",
    "fixed_hierarchical": "Fixed HierFL",
    "resource_tree": "Resource-Tree",
    "epidemic_learning": "Epidemic Learning",
    "drift": "DRIFT",
}
TABLE_FIELDS = (
    "final_accuracy", "best_accuracy", "accuracy_gap_to_fedavg_pp",
    "mean_round_latency_sim_ms", "p95_round_latency_sim_ms",
    "total_simulated_training_time_ms", "max_peak_node_traffic_bytes",
    "total_bytes", "total_messages", "communication_load_imbalance",
    "aggregation_load_imbalance", "final_consensus_error",
    "mean_consensus_error",
)


def _latest_complete_run(output_root: Path) -> Path:
    statuses = list(output_root.glob("**/config/completion_status.json"))
    complete = []
    for path in statuses:
        status = json.loads(path.read_text(encoding="utf-8"))
        if str(status.get("status", "")).upper() == "COMPLETE":
            complete.append(path)
    if not complete:
        raise ValueError(f"No COMPLETE run below {output_root}")
    return max(complete, key=lambda path: path.stat().st_mtime).parent.parent


def _round_rows(run: Path) -> list[dict[str, str]]:
    files = list(run.glob("*_round_metrics.csv"))
    if len(files) != 1:
        raise ValueError(f"Expected one round metrics CSV below {run}, found {len(files)}")
    with files[0].open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _latency_summary(metrics: dict[str, Any]) -> tuple[float | None, float | None]:
    values = sorted(
        float(item["round_latency_sim_ms"])
        for item in metrics.get("simulated_rounds", [])
        if item.get("round_latency_sim_ms") is not None
    )
    if not values:
        return None, None
    position = 0.95 * (len(values) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    p95 = values[lower] + (values[upper] - values[lower]) * (position - lower)
    return statistics.fmean(values), p95


def _record(method: str, seed: int, run: Path, metrics_path: Path) -> dict[str, Any]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = _round_rows(run)
    accuracy = [float(row["val_acc"]) for row in rows]
    mean_latency, p95_latency = _latency_summary(metrics)
    consensus = [
        float(row["consensus_error"])
        for row in rows if row.get("consensus_error") not in (None, "")
    ]
    result = {
        "method": method,
        "label": LABELS[method],
        "seed": seed,
        "run": str(run.resolve()),
        "rounds": len(rows),
        "final_accuracy": accuracy[-1],
        "best_accuracy": max(accuracy),
        "accuracy_curve": accuracy,
        "mean_round_latency_sim_ms": mean_latency,
        "p95_round_latency_sim_ms": p95_latency,
        "final_consensus_error": consensus[-1] if consensus else None,
        "mean_consensus_error": statistics.fmean(consensus) if consensus else None,
    }
    for field in TABLE_FIELDS:
        if result.get(field) is None and metrics.get(field) is not None:
            result[field] = metrics[field]
    return result


def load_records(matrix_root: Path) -> list[dict[str, Any]]:
    records = []
    for manifest_path in sorted(matrix_root.glob("seed_*/manifest.json")):
        plan = json.loads(manifest_path.read_text(encoding="utf-8"))
        seed = int(plan["seed"])
        fedavg_run = None
        for method in plan["methods"]:
            name = method["method"]
            run = _latest_complete_run(Path(method["output_root"]))
            if name == "fedavg":
                fedavg_run = run
            record = _record(name, seed, run, run / "metrics" / "metrics_summary.json")
            if name == "fedavg":
                record["accuracy_gap_to_fedavg_pp"] = 0.0
            records.append(record)
        if fedavg_run is None:
            raise ValueError(f"Seed {seed} has no FedAvg run")
        edge_path = fedavg_run / "metrics" / "fedavg_edge_metrics.json"
        if not edge_path.is_file():
            raise ValueError(f"Missing FedAvg-Edge metrics: {edge_path}")
        edge_record = _record("fedavg_edge", seed, fedavg_run, edge_path)
        # FedAvg-Edge reuses exactly the FedAvg learning curve.
        fedavg_record = next(
            item for item in records if item["seed"] == seed and item["method"] == "fedavg"
        )
        edge_record["accuracy_gap_to_fedavg_pp"] = 0.0
        edge_record["accuracy_curve"] = fedavg_record["accuracy_curve"]
        records.append(edge_record)
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for method in METHOD_ORDER:
        selected = [item for item in records if item["method"] == method]
        summary: dict[str, Any] = {"valid_runs": len(selected)}
        for field in TABLE_FIELDS:
            values = [float(item[field]) for item in selected if item.get(field) is not None]
            if values:
                summary[f"{field}_mean"] = statistics.fmean(values)
                summary[f"{field}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        result[method] = summary
    return result


def write_table(output: Path, summary: dict[str, dict[str, Any]]) -> None:
    fields = ["method", "label", "valid_runs"]
    for field in TABLE_FIELDS:
        fields.extend((f"{field}_mean", f"{field}_std"))
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in METHOD_ORDER:
            writer.writerow({"method": method, "label": LABELS[method], **summary[method]})


def _mean_curve(records: list[dict[str, Any]], method: str) -> tuple[list[float], list[float]]:
    curves = [item["accuracy_curve"] for item in records if item["method"] == method]
    length = {len(curve) for curve in curves}
    if len(length) != 1:
        raise ValueError(f"Inconsistent curve lengths for {method}: {sorted(length)}")
    means, stds = [], []
    for values in zip(*curves):
        means.append(statistics.fmean(values))
        stds.append(statistics.stdev(values) if len(values) > 1 else 0.0)
    return means, stds


def plot_accuracy(records: list[dict[str, Any]], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.4, 5.0))
    curves = (
        ("fedavg", "FedAvg / Fixed HierFL / Resource-Tree / DRIFT"),
        ("dpsgd", LABELS["dpsgd"]),
        ("epidemic_learning", LABELS["epidemic_learning"]),
    )
    for method, label in curves:
        mean, std = _mean_curve(records, method)
        rounds = list(range(1, len(mean) + 1))
        axis.plot(rounds, mean, linewidth=2, label=label)
        axis.fill_between(rounds, [a - b for a, b in zip(mean, std)],
                          [a + b for a, b in zip(mean, std)], alpha=0.15)
    axis.set_xlabel("Communication round")
    axis.set_ylabel("Global test accuracy")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_system(summary: dict[str, dict[str, Any]], output: Path) -> None:
    methods = list(METHOD_ORDER)
    panels = (
        ("mean_round_latency_sim_ms", 1e-3, "Mean round latency (s)"),
        ("max_peak_node_traffic_bytes", 1e-9, "Peak node traffic (GB)"),
        ("total_bytes", 1e-9, "Total traffic (GB)"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
    colors = ["#7f7f7f", "#555555", "#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756"]
    for axis, (field, scale, ylabel) in zip(axes, panels):
        means = [summary[m].get(f"{field}_mean", 0.0) * scale for m in methods]
        stds = [summary[m].get(f"{field}_std", 0.0) * scale for m in methods]
        axis.bar(range(len(methods)), means, yerr=stds, capsize=3, color=colors)
        axis.set_xticks(range(len(methods)), [LABELS[m] for m in methods], rotation=42, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    matrix_root = args.matrix_root.resolve()
    output_dir = (args.output_dir or matrix_root / "analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(matrix_root)
    summary = summarize(records)
    payload = {
        "schema_version": 1,
        "matrix_root": str(matrix_root),
        "seeds": sorted({item["seed"] for item in records}),
        "records": [{k: v for k, v in item.items() if k != "accuracy_curve"} for item in records],
        "aggregate": summary,
        "rounds_to_target_status": "unavailable_no_prefrozen_target",
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_table(output_dir / "analysis_summary.csv", summary)
    plot_accuracy(records, output_dir / "accuracy_curves.png")
    plot_system(summary, output_dir / "system_tradeoff.png")
    print(f"Analysis written to: {output_dir}")


if __name__ == "__main__":
    main()
