#!/usr/bin/env python3
"""Plot the completed personalized fixed-size JSD experiment.

The script is read-only with respect to training artifacts.  It selects only
COMPLETE runs, aggregates cluster metrics with their appropriate sample
weights, and writes PDF, 300-DPI PNG, and the exact plotted CSV values.
"""

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


QUANTILE_METHODS = tuple(f"drift_jsd_q{value:02d}" for value in (0, 25, 50, 75, 100))
METHODS = ("drift_random_matched", *QUANTILE_METHODS)
LABELS = {
    "drift_random_matched": "Random (size-matched)",
    **{method: method.removeprefix("drift_jsd_").upper() for method in QUANTILE_METHODS},
}
COLORS = {
    "drift_random_matched": "#555555",
    "drift_jsd_q00": "#2166AC",
    "drift_jsd_q25": "#67A9CF",
    "drift_jsd_q50": "#7F7F7F",
    "drift_jsd_q75": "#EF8A62",
    "drift_jsd_q100": "#B2182B",
}


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.2,
        "lines.linewidth": 1.7,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#4A4A4A",
        "axes.facecolor": "#F7F8FA",
        "figure.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def polish(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(color="white", linewidth=0.9)
    axis.set_axisbelow(True)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def completed_metric_file(method_root: Path) -> Path:
    matches = []
    for metric_path in method_root.glob("*/cluster_round_metrics.csv"):
        status_path = metric_path.parent / "config" / "completion_status.json"
        if not status_path.is_file():
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "COMPLETE" and status.get("completed_rounds") == 30:
            matches.append(metric_path)
    if len(matches) != 1:
        raise ValueError(f"Expected one complete metric file below {method_root}, found {matches}")
    return matches[0]


def weighted(rows: list[dict[str, str]], value: str, weight: str) -> float:
    pairs = [(float(row[value]), float(row[weight])) for row in rows if row.get(value)]
    return sum(item * mass for item, mass in pairs) / sum(mass for _, mass in pairs)


def load_run(seed: int, method: str, path: Path) -> dict[str, Any]:
    rows = read_rows(path)
    by_round: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        by_round.setdefault(int(row["round_id"]), []).append(row)
    rounds = sorted(by_round)
    if rounds != list(range(30)):
        raise ValueError(f"{method} seed={seed} does not contain rounds 0..29")

    personalized = [
        weighted(by_round[index], "cluster_personalized_accuracy", "validation_samples")
        for index in rounds
    ]
    cluster_global = [
        weighted(by_round[index], "cluster_global_accuracy", "train_samples")
        for index in rounds
    ]
    jsd_pairs = [
        (float(row["mean_pairwise_label_jsd"]), int(row["label_pair_count"]))
        for row in rows if int(row["label_pair_count"]) > 0
    ]
    achieved_jsd = sum(value * count for value, count in jsd_pairs) / sum(
        count for _, count in jsd_pairs
    )
    final_global = float(by_round[29][0]["final_global_accuracy"])
    return {
        "seed": seed,
        "method": method,
        "achieved_jsd": achieved_jsd,
        "personalized": personalized,
        "cluster_global": cluster_global,
        "final_global": final_global,
        "num_clusters": len(by_round[29]),
        "singleton_ratio": sum(
            int(row["num_members"]) == 1 for row in by_round[29]
        ) / len(by_round[29]),
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def save(fig: plt.Figure, stem: Path, rows: list[dict[str, Any]]) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    plt.close(fig)


def plot_steady_state(records: dict[str, list[dict[str, Any]]], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.65))
    specs = (
        ("personalized", "Personalized accuracy (%)", "(a) Cluster-personalized test"),
        ("cluster_global", "Accuracy (%)", "(b) Global test per cluster model"),
        ("final_global", "Accuracy (%)", "(c) Final global model"),
    )
    plotted = []
    for axis, (metric, ylabel, title) in zip(axes, specs):
        qx, qy = [], []
        for method in QUANTILE_METHODS:
            values = records[method]
            xmean, xstd = mean_std([row["achieved_jsd"] for row in values])
            raw = [
                statistics.fmean(row[metric][-5:]) if isinstance(row[metric], list)
                else row[metric]
                for row in values
            ]
            ymean, ystd = mean_std(raw)
            qx.append(xmean)
            qy.append(100 * ymean)
            plotted.append({
                "panel_metric": metric, "method": method,
                "achieved_jsd_mean": xmean, "achieved_jsd_std": xstd,
                "accuracy_mean_percent": 100 * ymean,
                "accuracy_std_percent": 100 * ystd,
            })
        axis.plot(
            qx, qy, color="#3B6FB6", marker="o",
            markersize=4, label="JSD-controlled clusters",
        )

        random_values = records["drift_random_matched"]
        rx, rxerr = mean_std([row["achieved_jsd"] for row in random_values])
        random_raw = [
            statistics.fmean(row[metric][-5:]) if isinstance(row[metric], list)
            else row[metric]
            for row in random_values
        ]
        ry, ryerr = mean_std(random_raw)
        axis.plot(
            [rx], [100 * ry], color=COLORS["drift_random_matched"],
            marker="D", markersize=4.5, linestyle="none",
            label="Random (size-matched)", zorder=4,
        )
        plotted.append({
            "panel_metric": metric, "method": "drift_random_matched",
            "achieved_jsd_mean": rx, "achieved_jsd_std": rxerr,
            "accuracy_mean_percent": 100 * ry,
            "accuracy_std_percent": 100 * ryerr,
        })
        axis.set(xlabel="Achieved within-cluster JSD", ylabel=ylabel, title=title)
        polish(axis)
    axes[0].legend(frameon=False, loc="lower left", fontsize=6.8)
    fig.tight_layout(w_pad=1.15)
    save(fig, output / "personalized_jsd_steady_state", plotted)


def plot_learning_curves(records: dict[str, list[dict[str, Any]]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))
    plotted = []
    for axis, metric, ylabel, title in (
        (axes[0], "personalized", "Personalized accuracy (%)", "(a) Personalized evaluation"),
        (axes[1], "cluster_global", "Global-test accuracy (%)", "(b) Cluster models on global test"),
    ):
        for method in METHODS:
            curves = [row[metric] for row in records[method]]
            means = [statistics.fmean(values) for values in zip(*curves)]
            stds = [statistics.stdev(values) for values in zip(*curves)]
            rounds = list(range(1, 31))
            color = COLORS[method]
            linestyle = "--" if method == "drift_random_matched" else "-"
            marker = "D" if method == "drift_random_matched" else "o"
            markevery = 5 if method == "drift_random_matched" else 6
            axis.plot(
                rounds, [100 * value for value in means], label=LABELS[method],
                color=color, linestyle=linestyle, marker=marker,
                markevery=markevery, markersize=2.8,
            )
            plotted.extend({
                "panel_metric": metric, "method": method, "round": round_id,
                "accuracy_mean_percent": 100 * mean,
                "accuracy_std_percent": 100 * std,
            } for round_id, mean, std in zip(rounds, means, stds))
        axis.set(xlabel="Global round", ylabel=ylabel, title=title, xlim=(1, 30))
        polish(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, 1.03), columnspacing=1.2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=1.2)
    save(fig, output / "personalized_jsd_learning_curves", plotted)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.experiment_root.resolve()
    output = (args.output_dir or root / "jsd_curve_analysis" / "figures").resolve()
    records: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    for seed_dir in sorted(root.glob("seed_*")):
        seed = int(seed_dir.name.removeprefix("seed_"))
        for method in METHODS:
            metric_path = completed_metric_file(seed_dir / "runs" / method)
            records[method].append(load_run(seed, method, metric_path))
    seed_counts = {method: len(values) for method, values in records.items()}
    if set(seed_counts.values()) != {3}:
        raise ValueError(f"Expected three complete seeds per method, found {seed_counts}")

    configure_style()
    plot_steady_state(records, output)
    plot_learning_curves(records, output)
    print(f"Figures and source CSV files: {output}")


if __name__ == "__main__":
    main()
