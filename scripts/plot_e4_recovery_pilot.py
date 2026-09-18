#!/usr/bin/env python3
"""Plot the paired E4 multi-node-failure pilot and its recovery gap."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "outputs/pilots/e4_multi_node_failure_v1/e4_multi_node_failure_pilot"
CONTROL = ROOT / "outputs/formal_v1/e4_no_failure_control/seed_42/runs"
FIXED_CONTROL = ROOT / "outputs/formal_v1/e2_static_tradeoff/seed_42/runs/fixed_hierarchical"
OUTPUT = PILOT / "analysis/e4_multi_node_failure_recovery_pilot.png"

METHODS = {
    "fixed_hierarchical": ("Fixed Hierarchical", "#7f7f7f", "--", 1.8),
    "dynamic_dpsgd": ("Dynamic D-PSGD", "#1f77b4", "-.", 1.8),
    "epidemic_learning": ("Epidemic Learning", "#ff7f0e", ":", 2.0),
    "resource_tree": ("Resource-Tree (oracle)", "#2ca02c", "--", 2.0),
    "drift": ("DRIFT", "#d62728", "-", 3.0),
}

FAILURE_ROUND = 20
RECOVERY_ROUND = 30
RECOVERY_THRESHOLD_PP = 0.5
RECOVERY_PATIENCE = 2


def curve(run_root: Path) -> dict[int, float]:
    files = sorted(run_root.glob("*/*round_metrics.csv"))
    if len(files) != 1:
        raise RuntimeError(f"Expected one round-metrics file under {run_root}, found {files}")
    with files[0].open(newline="", encoding="utf-8") as handle:
        return {
            int(row["round_id"]): float(row["val_acc"])
            for row in csv.DictReader(handle)
        }


def confirmation_round(gaps: dict[int, float]) -> int | None:
    last = max(gaps)
    for round_id in range(RECOVERY_ROUND, last - RECOVERY_PATIENCE + 2):
        window = range(round_id, round_id + RECOVERY_PATIENCE)
        if all(gaps[item] <= RECOVERY_THRESHOLD_PP for item in window):
            return round_id + RECOVERY_PATIENCE - 1
    return None


def main() -> None:
    pilot_curves = {
        method: curve(PILOT / "seed_42/runs" / method) for method in METHODS
    }
    control_curves = {
        method: curve(FIXED_CONTROL if method == "fixed_hierarchical" else CONTROL / method)
        for method in METHODS
    }

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)

    for method, (label, color, linestyle, width) in METHODS.items():
        rounds = sorted(pilot_curves[method])
        accuracy = [100.0 * pilot_curves[method][item] for item in rounds]
        ax1.plot(rounds, accuracy, label=label, color=color,
                 linestyle=linestyle, linewidth=width)

        gaps = {
            item: 100.0 * (control_curves[method][item] - pilot_curves[method][item])
            for item in rounds
        }
        zoom_rounds = [item for item in rounds if 18 <= item <= 50]
        ax2.plot(zoom_rounds, [gaps[item] for item in zoom_rounds], label=label,
                 color=color, linestyle=linestyle, linewidth=width)
        recovered = confirmation_round(gaps)
        if recovered is not None and method in {"dynamic_dpsgd", "epidemic_learning", "drift"}:
            ax2.scatter([recovered], [gaps[recovered]], color=color, s=34, zorder=5)
            ax2.annotate(
                f"{recovered - RECOVERY_ROUND} rounds",
                (recovered, gaps[recovered]), xytext=(4, 8),
                textcoords="offset points", fontsize=8, color=color,
            )

    for axis in (ax1, ax2):
        axis.axvspan(FAILURE_ROUND, RECOVERY_ROUND, color="#d9d9d9", alpha=0.42,
                    label="20% nodes unavailable" if axis is ax1 else None)
        axis.axvline(FAILURE_ROUND, color="#555555", linewidth=1.0)
        axis.axvline(RECOVERY_ROUND, color="#555555", linewidth=1.0)
        axis.text(FAILURE_ROUND + 0.35, axis.get_ylim()[1], "Failure", va="top", fontsize=8)
        axis.text(RECOVERY_ROUND + 0.35, axis.get_ylim()[1], "Rejoin", va="top", fontsize=8)

    ax1.set_title("Learning trajectories under transient multi-node failure")
    ax1.set_xlabel("Round index")
    ax1.set_ylabel("Global test accuracy (%)")
    ax1.set_xlim(0, 59)
    ax1.legend(loc="lower right", fontsize=8, ncol=2, frameon=True)

    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.axhline(RECOVERY_THRESHOLD_PP, color="#9467bd", linestyle="--", linewidth=1.2,
                label="Recovery threshold (0.5 pp)")
    ax2.set_title("Recovery gap relative to matched no-failure control")
    ax2.set_xlabel("Round index")
    ax2.set_ylabel("Accuracy deficit (percentage points)")
    ax2.set_xlim(18, 50)
    ax2.set_ylim(-0.8, 8.0)
    ax2.legend(loc="upper right", fontsize=8, frameon=True)

    fig.suptitle(
        "E4 pilot: 6/30 nodes fail at round 20 and rejoin at round 30 (seed 42)",
        fontsize=13,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    print(OUTPUT)


if __name__ == "__main__":
    main()
