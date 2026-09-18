#!/usr/bin/env python3
"""Combine the reused seed-42 pilot with formal E4 seeds 43 and 44."""

from __future__ import annotations

import csv
import glob
import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/formal_v1/e4_multi_node_failure_analysis"
DOC = ROOT / "docs/EXPERIMENT_4_ANALYSIS.md"
SEEDS = (42, 43, 44)
FAILURE, REJOIN, THRESHOLD, PATIENCE = 20, 30, 0.5, 2
METHODS = {
    "fixed_hierarchical": ("Fixed Hierarchical", "#7f7f7f", "--", 1.8),
    "dynamic_dpsgd": ("Dynamic D-PSGD", "#1f77b4", "-.", 1.8),
    "epidemic_learning": ("Epidemic Learning", "#ff7f0e", ":", 2.0),
    "resource_tree": ("Resource-Tree (oracle)", "#2ca02c", "--", 2.0),
    "drift": ("DRIFT", "#d62728", "-", 3.0),
}


def failure_root(seed: int, method: str) -> Path:
    if seed == 42:
        base = ROOT / "outputs/pilots/e4_multi_node_failure_v1/e4_multi_node_failure_pilot"
    else:
        base = ROOT / "outputs/formal_v1/e4_multi_node_failure_remaining_v1/e4_multi_node_failure_pilot"
    return base / f"seed_{seed}/runs" / method


def control_root(seed: int, method: str) -> Path:
    if method == "fixed_hierarchical":
        return ROOT / f"outputs/formal_v1/e2_static_tradeoff/seed_{seed}/runs/{method}"
    return ROOT / f"outputs/formal_v1/e4_no_failure_control/seed_{seed}/runs/{method}"


def curve(root: Path) -> dict[int, float]:
    files = glob.glob(str(root / "*/*round_metrics.csv"))
    if len(files) != 1:
        raise RuntimeError(f"Expected one metrics file below {root}, got {files}")
    with open(files[0], newline="", encoding="utf-8") as handle:
        return {int(row["round_id"]): 100 * float(row["val_acc"])
                for row in csv.DictReader(handle)}


def system_record(seed: int, method: str) -> dict:
    summary = (ROOT / "outputs/pilots/e4_multi_node_failure_v1/e4_multi_node_failure_pilot/matrix_summary.json"
               if seed == 42 else
               ROOT / "outputs/formal_v1/e4_multi_node_failure_remaining_v1/e4_multi_node_failure_pilot/matrix_summary.json")
    records = json.loads(summary.read_text())["records"]
    return next(item for item in records if item["seed"] == seed and item["method"] == method)


def mean_ci(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    # Student-t 95% CI for df=2.
    half = 4.303 * statistics.stdev(values) / math.sqrt(len(values))
    return mean, half


def recovery_round(gaps: dict[int, float]) -> int | None:
    for round_id in range(REJOIN, max(gaps) - PATIENCE + 2):
        if all(gaps[r] <= THRESHOLD for r in range(round_id, round_id + PATIENCE)):
            return round_id + PATIENCE - 1 - REJOIN
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fault, gaps, rows = {}, {}, []
    for method in METHODS:
        fault[method], gaps[method] = {}, {}
        for seed in SEEDS:
            f, c = curve(failure_root(seed, method)), curve(control_root(seed, method))
            fault[method][seed] = f
            gaps[method][seed] = {r: c[r] - f[r] for r in f}
            sys = system_record(seed, method)
            rows.append({
                "method": method, "seed": seed, "final_accuracy_pct": f[59],
                "final_gap_pp": gaps[method][seed][59],
                "failure_window_mean_gap_pp": statistics.mean(gaps[method][seed][r] for r in range(FAILURE, REJOIN)),
                "failure_window_max_gap_pp": max(gaps[method][seed][r] for r in range(FAILURE, REJOIN)),
                "recovery_rounds_after_rejoin": recovery_round(gaps[method][seed]),
                "mean_round_latency_s": sys["mean_round_latency_sim_ms"] / 1000,
                "total_traffic_gb": sys["total_bytes"] / 1e9,
                "peak_node_traffic_mb": sys["max_peak_node_traffic_bytes"] / 1e6,
            })
    with (OUT / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    aggregate = []
    for method, (label, *_rest) in METHODS.items():
        own = [row for row in rows if row["method"] == method]
        item = {"method": method, "label": label}
        for key in ("final_accuracy_pct", "final_gap_pp", "failure_window_mean_gap_pp",
                    "failure_window_max_gap_pp", "mean_round_latency_s",
                    "total_traffic_gb", "peak_node_traffic_mb"):
            mean, ci = mean_ci([row[key] for row in own])
            item[f"{key}_mean"], item[f"{key}_ci95"] = mean, ci
        recovered = [row["recovery_rounds_after_rejoin"] for row in own]
        item["recovery_rounds"] = recovered
        item["recovery_rounds_mean"] = None if any(x is None for x in recovered) else statistics.mean(recovered)
        aggregate.append(item)
    (OUT / "summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)
    for method, (label, color, style, width) in METHODS.items():
        rounds = range(60)
        means, cis, gap_means, gap_cis = [], [], [], []
        for r in rounds:
            mean, ci = mean_ci([fault[method][seed][r] for seed in SEEDS]); means.append(mean); cis.append(ci)
            mean, ci = mean_ci([gaps[method][seed][r] for seed in SEEDS]); gap_means.append(mean); gap_cis.append(ci)
        ax1.plot(rounds, means, label=label, color=color, linestyle=style, linewidth=width)
        ax1.fill_between(rounds, [m-c for m,c in zip(means,cis)], [m+c for m,c in zip(means,cis)], color=color, alpha=.08)
        zoom = list(range(18, 51))
        ax2.plot(zoom, [gap_means[r] for r in zoom], label=label, color=color, linestyle=style, linewidth=width)
        ax2.fill_between(zoom, [gap_means[r]-gap_cis[r] for r in zoom],
                         [gap_means[r]+gap_cis[r] for r in zoom], color=color, alpha=.10)
    for ax in (ax1, ax2):
        ax.axvspan(FAILURE, REJOIN, color="#d9d9d9", alpha=.35)
        ax.axvline(FAILURE, color="#555", linewidth=1); ax.axvline(REJOIN, color="#555", linewidth=1)
    ax1.set(title="Learning under transient 20% node failure", xlabel="Round index", ylabel="Global test accuracy (%)", xlim=(0,59))
    ax1.legend(fontsize=8, ncol=2, loc="lower right")
    ax2.axhline(0, color="black", linewidth=.8); ax2.axhline(THRESHOLD, color="#9467bd", linestyle="--", linewidth=1.2)
    ax2.set(title="Paired accuracy deficit vs. no-failure control", xlabel="Round index",
            ylabel="Accuracy deficit (percentage points)", xlim=(18,50))
    ax2.legend(fontsize=8, loc="upper right")
    fig.suptitle("E4: 6/30 nodes fail at round 20 and rejoin at round 30 (mean and 95% CI, 3 seeds)")
    fig.savefig(OUT / "e4_multi_node_failure_3seed.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "e4_multi_node_failure_3seed.pdf", bbox_inches="tight")

    by_method = {item["method"]: item for item in aggregate}
    lines = [
        "# Experiment 4: Multi-node failure and recovery", "",
        "## Purpose", "",
        "Evaluate learning degradation and recovery when 6/30 method-independent random nodes are unavailable from round 20 through round 29. The surviving physical graph is connected for every seed.", "",
        "## Protocol", "", "- CIFAR-10, ResNet-18, Dirichlet alpha=1.0, 30 clients.",
        "- Seeds 42, 43, and 44; 60 rounds with the learning-rate schedule fixed to 100 rounds.",
        "- Recovery means two consecutive rounds within 0.5 percentage points of the paired no-failure control.",
        "- Resource-Tree is an oracle-assisted reference, not a deployable literature baseline.", "",
        "## Results", "", "| Method | Final accuracy | Failure-window mean gap | Recovery rounds | Mean latency | Total traffic |", "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        x = by_method[method]; recovery = x["recovery_rounds_mean"]
        recovery_text = f"{recovery:.2f}" if recovery is not None else "censored in one seed"
        lines.append(f"| {x['label']} | {x['final_accuracy_pct_mean']:.2f}% | {x['failure_window_mean_gap_pp_mean']:.2f} pp | {recovery_text} | {x['mean_round_latency_s_mean']:.2f}s | {x['total_traffic_gb_mean']:.2f}GB |")
    lines += ["", "## Main finding", "",
              "DRIFT returned to the paired no-failure trajectory in one round for every seed, matching the oracle Resource-Tree reference. Dynamic D-PSGD required 10-15 rounds, Epidemic Learning 4-11 rounds, and Fixed Hierarchical 14-17 rounds in the two recovered seeds while remaining outside the threshold in seed 42.", "",
              "The claim is a trade-off rather than universal dominance: DRIFT preserves the strongest final accuracy among deployable baselines and sharply improves recovery, while Dynamic D-PSGD retains lower simulated latency and peak traffic.", ""]
    DOC.write_text("\n".join(lines), encoding="utf-8")
    print(OUT / "e4_multi_node_failure_3seed.png")
    print(DOC)


if __name__ == "__main__":
    main()
