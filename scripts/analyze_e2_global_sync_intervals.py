"""Consolidate E2 baselines and DRIFT global-sync interval results.

This script is read-only with respect to the source experiment directories.  It
writes paper-facing CSV/JSON/figures to a separate analysis directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
INTERVALS = (1, 3, 5)
LABELS = {1: "DRIFT (Interval-1)", 3: "DRIFT (Interval-3)", 5: "DRIFT (Interval-5)"}


def latest_complete_run(root: Path) -> Path:
    candidates = []
    for status_path in root.glob("*/config/completion_status.json"):
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if str(status.get("status", "")).upper() == "COMPLETE":
            candidates.append(status_path.parent.parent)
    if not candidates:
        raise ValueError(f"No COMPLETE run below {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def run_for(seed: int, interval: int) -> Path:
    if interval == 1:
        base = ROOT / f"outputs/formal_v1/e2_static_tradeoff/seed_{seed}/runs/drift"
    elif seed == 42:
        base = ROOT / f"outputs/formal/e2_global_sync_100r_seed42_v1/interval_{interval}"
    else:
        base = ROOT / f"outputs/formal/e2_global_sync_100r_remaining_v1/seed_{seed}/interval_{interval}"
    return latest_complete_run(base)


def only_file(run: Path, pattern: str) -> Path:
    matches = list(run.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one {pattern} below {run}, found {len(matches)}")
    return matches[0]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def record(seed: int, interval: int) -> dict:
    run = run_for(seed, interval)
    rounds = read_csv(only_file(run, "*_round_metrics.csv"))
    comm = read_csv(only_file(run, "*_communication_stats.csv"))
    evaluated = [
        (int(row["round_id"]) + 1, float(row["val_acc"]))
        for row in rounds
        if float(row.get("val_acc") or 0.0) > 0.0
    ]
    if len(rounds) != 100 or not evaluated or evaluated[-1][0] != 100:
        raise ValueError(f"Incomplete 100-round evaluation in {run}")
    total_messages = sum(int(row["total_msgs"]) for row in comm)
    total_bytes = sum(int(row["total_bytes"]) for row in comm)
    # Legacy Interval-1 CSVs predate this explicit column; the original
    # protocol synchronizes globally in every round.
    sync_rounds = sum(
        str(row.get("global_sync_performed", "true")).lower() == "true"
        for row in comm
    )
    wall_seconds = sum(float(row.get("round_wall_clock_seconds") or 0.0) for row in rounds)
    return {
        "method": f"drift_interval_{interval}",
        "label": LABELS[interval],
        "seed": seed,
        "global_sync_interval": interval,
        "rounds": len(rounds),
        "global_sync_rounds": sync_rounds,
        "final_accuracy": evaluated[-1][1],
        "best_accuracy": max(value for _, value in evaluated),
        "total_messages": total_messages,
        "total_bytes": total_bytes,
        "total_traffic_gb": total_bytes / 1e9,
        "mean_messages_per_round": total_messages / len(rounds),
        "mean_traffic_gb_per_round": total_bytes / len(rounds) / 1e9,
        "training_wall_seconds": wall_seconds,
        "evaluated_curve": evaluated,
        "run": str(run.relative_to(ROOT)),
    }


def mean_std(records: list[dict], key: str) -> tuple[float, float]:
    values = [float(row[key]) for row in records]
    return statistics.fmean(values), statistics.stdev(values)


def summaries(records: list[dict]) -> list[dict]:
    result = []
    baseline = [row for row in records if row["global_sync_interval"] == 1]
    baseline_accuracy = statistics.fmean(row["final_accuracy"] for row in baseline)
    baseline_bytes = statistics.fmean(row["total_bytes"] for row in baseline)
    baseline_messages = statistics.fmean(row["total_messages"] for row in baseline)
    fields = (
        "final_accuracy", "best_accuracy", "total_messages", "total_bytes",
        "total_traffic_gb", "mean_messages_per_round", "mean_traffic_gb_per_round",
        "training_wall_seconds", "global_sync_rounds",
    )
    for interval in INTERVALS:
        selected = [row for row in records if row["global_sync_interval"] == interval]
        item = {
            "method": f"drift_interval_{interval}",
            "label": LABELS[interval],
            "global_sync_interval": interval,
            "valid_runs": len(selected),
        }
        for field in fields:
            item[f"{field}_mean"], item[f"{field}_std"] = mean_std(selected, field)
        item["accuracy_change_vs_interval1_pp"] = (
            item["final_accuracy_mean"] - baseline_accuracy
        ) * 100.0
        item["message_reduction_vs_interval1_pct"] = (
            1.0 - item["total_messages_mean"] / baseline_messages
        ) * 100.0
        item["traffic_reduction_vs_interval1_pct"] = (
            1.0 - item["total_bytes_mean"] / baseline_bytes
        ) * 100.0
        result.append(item)
    return result


def write_csv(path: Path, rows: list[dict], excluded: set[str] | None = None) -> None:
    excluded = excluded or set()
    fields = [key for key in rows[0] if key not in excluded]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def write_paper_comparison(output: Path, interval_summary: list[dict]) -> None:
    source = ROOT / "outputs/paper_results_archive_v1/E2/analysis/analysis_summary.csv"
    old = read_csv(source)
    selected_fields = (
        "method", "label", "valid_runs", "final_accuracy_mean", "final_accuracy_std",
        "best_accuracy_mean", "best_accuracy_std", "total_messages_mean",
        "total_messages_std", "total_bytes_mean", "total_bytes_std",
    )
    rows = [{key: row.get(key, "") for key in selected_fields} for row in old]
    # The existing DRIFT row is the original Interval-1 protocol; make that explicit.
    for row in rows:
        if row["method"] == "drift":
            row["method"], row["label"] = "drift_interval_1", LABELS[1]
    for summary in interval_summary[1:]:
        rows.append({key: summary.get(key, "") for key in selected_fields})
    write_csv(output, rows)


def plot_learning(records: list[dict], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.8, 4.8))
    colors = {1: "#2F5597", 3: "#E07A1F", 5: "#3A8F5B"}
    markers = {1: None, 3: "o", 5: "s"}
    for interval in INTERVALS:
        selected = [row for row in records if row["global_sync_interval"] == interval]
        round_sets = {tuple(round_id for round_id, _ in row["evaluated_curve"]) for row in selected}
        if len(round_sets) != 1:
            raise ValueError(f"Mismatched evaluation rounds for interval {interval}")
        x_values = list(next(iter(round_sets)))
        means = [
            statistics.fmean(row["evaluated_curve"][idx][1] for row in selected)
            for idx in range(len(x_values))
        ]
        axis.plot(
            x_values, means, label=LABELS[interval], color=colors[interval], linewidth=2,
            marker=markers[interval], markersize=3.2, markevery=max(1, len(x_values) // 12),
        )
    axis.set_xlabel("Training round")
    axis.set_ylabel("Global test accuracy")
    axis.set_xlim(1, 100)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=240)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def plot_tradeoff(summary: list[dict], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.6, 4.8))
    colors = {1: "#2F5597", 3: "#E07A1F", 5: "#3A8F5B"}
    for row in summary:
        interval = int(row["global_sync_interval"])
        x_value = row["total_traffic_gb_mean"]
        y_value = row["final_accuracy_mean"] * 100.0
        axis.scatter(x_value, y_value, s=75, color=colors[interval], zorder=3)
        offset = (-70, 7) if interval == 1 else (7, 5)
        axis.annotate(
            f"Interval-{interval}", (x_value, y_value), xytext=offset,
            textcoords="offset points", fontsize=9,
        )
    axis.set_xlabel("Total traffic over 100 rounds (GB, mean of 3 seeds)")
    axis.set_ylabel("Final global accuracy (%, mean of 3 seeds)")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=240)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "outputs/paper_results_archive_v1/E2/global_sync_interval_analysis",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    records = [record(seed, interval) for seed in SEEDS for interval in INTERVALS]
    summary = summaries(records)
    write_csv(output / "per_seed_results.csv", records, {"evaluated_curve"})
    write_csv(output / "interval_summary.csv", summary)
    write_paper_comparison(output / "e2_paper_comparison.csv", summary)
    plot_learning(records, output / "interval_learning_curves")
    plot_tradeoff(summary, output / "interval_accuracy_traffic_tradeoff")

    manifest = {
        "analysis": "E2 DRIFT global synchronization interval comparison",
        "seeds": list(SEEDS),
        "rounds": 100,
        "source_runs": [row["run"] for row in records],
        "source_baseline_summary": "outputs/paper_results_archive_v1/E2/analysis/analysis_summary.csv",
        "notes": [
            "No training was rerun.",
            "Traffic and message counts use the same runtime communication CSV schema for Interval-1/3/5.",
            "Learning curves show three-seed means without variance shading.",
        ],
    }
    (output / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    by_interval = {int(row["global_sync_interval"]): row for row in summary}
    readme = f"""# E2 global synchronization interval analysis

This folder consolidates the existing E2 baselines with the completed 100-round,
three-seed DRIFT Interval-1/3/5 experiments. No experiment was rerun and no source
run was modified.

## Main result

| Setting | Final accuracy | Total traffic | Traffic reduction vs I1 | Accuracy change vs I1 |
|---|---:|---:|---:|---:|
"""
    for interval in INTERVALS:
        row = by_interval[interval]
        readme += (
            f"| Interval-{interval} | {row['final_accuracy_mean']*100:.2f}% "
            f"± {row['final_accuracy_std']*100:.2f} | {row['total_traffic_gb_mean']:.2f} GB "
            f"± {row['total_traffic_gb_std']:.2f} | "
            f"{row['traffic_reduction_vs_interval1_pct']:.2f}% | "
            f"{row['accuracy_change_vs_interval1_pp']:+.2f} pp |\n"
        )
    readme += """

## Files

- `per_seed_results.csv`: nine source runs and their measured results.
- `interval_summary.csv`: mean/std and changes relative to Interval-1.
- `e2_paper_comparison.csv`: focused E2 baseline table plus Interval-3/5.
- `interval_learning_curves.{png,pdf}`: mean learning curves, without variance shading.
- `interval_accuracy_traffic_tradeoff.{png,pdf}`: final-accuracy/traffic trade-off.
- `source_manifest.json`: source paths and analysis assumptions.

Interpretation: Interval-1 remains the accuracy-oriented choice. Interval-3 is a
moderate communication-saving option, while Interval-5 is the communication-oriented
choice for stable networks. This E2 result alone does not establish which interval
should be used in dynamic scenarios.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(f"Wrote E2 interval analysis to {output}")


if __name__ == "__main__":
    main()
