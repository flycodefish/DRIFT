"""Analyze E4 DRIFT Interval-1/3/5 with matched no-failure controls."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/paper_results_archive_v1/E4/global_sync_interval_analysis"
SEEDS = (42, 43, 44)
INTERVALS = (1, 3, 5)
FAILURE, REJOIN = 20, 30


def latest_complete(base: Path) -> Path:
    runs = []
    for status in base.glob("*/config/completion_status.json"):
        if json.loads(status.read_text())["status"] == "COMPLETE":
            runs.append(status.parent.parent)
    if not runs:
        raise RuntimeError(f"No COMPLETE run below {base}")
    return max(runs, key=lambda path: path.stat().st_mtime)


def failure_run(seed: int, interval: int) -> Path:
    if interval == 1:
        base = (ROOT / f"outputs/pilots/e4_multi_node_failure_v1/e4_multi_node_failure_pilot/seed_{seed}/runs/drift"
                if seed == 42 else
                ROOT / f"outputs/formal_v1/e4_multi_node_failure_remaining_v1/e4_multi_node_failure_pilot/seed_{seed}/runs/drift")
    else:
        base = ROOT / f"outputs/formal/e4_global_sync_intervals_60r_3seed_v1/seed_{seed}/interval_{interval}"
    return latest_complete(base)


def control_run(seed: int, interval: int) -> Path:
    if interval == 1:
        base = ROOT / f"outputs/formal_v1/e4_no_failure_control/seed_{seed}/runs/drift"
    elif seed == 42:
        base = ROOT / f"outputs/formal/e2_global_sync_100r_seed42_v1/interval_{interval}"
    else:
        base = ROOT / f"outputs/formal/e2_global_sync_100r_remaining_v1/seed_{seed}/interval_{interval}"
    return latest_complete(base)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def only(run: Path, pattern: str) -> Path:
    found = list(run.glob(pattern))
    if len(found) != 1:
        raise RuntimeError(f"Expected one {pattern} below {run}, got {found}")
    return found[0]


def curve(run: Path) -> dict[int, float]:
    return {int(row["round_id"]): 100 * float(row["val_acc"])
            for row in rows(only(run, "*_round_metrics.csv"))}


def comm(run: Path) -> tuple[list[int], int, int]:
    data = rows(only(run, "*_communication_stats.csv"))
    # Legacy Interval-1 files have one row per round but no explicit round id
    # or sync flag; the original protocol synchronizes every round.
    syncs = ([int(row.get("round_id", index)) for index, row in enumerate(data)
              if row.get("global_sync_performed", "true").lower() == "true"])
    return syncs, sum(int(row["total_msgs"]) for row in data), sum(int(row["total_bytes"]) for row in data)


def recovery(syncs: list[int], gaps: dict[int, float]) -> int | None:
    post = [round_id for round_id in syncs if round_id >= REJOIN]
    for first, second in zip(post, post[1:]):
        if gaps[first] <= 0.5 and gaps[second] <= 0.5:
            return second - REJOIN
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records, curves = [], {}
    for interval in INTERVALS:
        curves[interval] = {}
        for seed in SEEDS:
            failed, control = failure_run(seed, interval), control_run(seed, interval)
            fcurve, ccurve = curve(failed), curve(control)
            syncs, messages, total_bytes = comm(failed)
            gaps = {round_id: ccurve[round_id] - fcurve[round_id] for round_id in syncs}
            window = [round_id for round_id in syncs if FAILURE <= round_id < REJOIN]
            records.append({
                "interval": interval, "seed": seed, "sync_rounds": len(syncs),
                "final_accuracy_pct": fcurve[59], "final_gap_to_matched_control_pp": gaps[59],
                "failure_window_sync_checkpoints": len(window),
                "failure_window_mean_gap_pp": statistics.fmean(gaps[r] for r in window),
                "failure_window_max_gap_pp": max(gaps[r] for r in window),
                "recovery_rounds_after_rejoin": recovery(syncs, gaps),
                "total_messages": messages, "total_bytes": total_bytes,
                "total_traffic_gb": total_bytes / 1e9,
                "failure_run": str(failed.relative_to(ROOT)),
                "control_run": str(control.relative_to(ROOT)),
            })
            curves[interval][seed] = fcurve

    fields = list(records[0])
    with (OUT / "per_seed_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(records)

    metrics = ("final_accuracy_pct", "final_gap_to_matched_control_pp",
               "failure_window_mean_gap_pp", "failure_window_max_gap_pp",
               "recovery_rounds_after_rejoin", "total_messages", "total_traffic_gb")
    summary = []
    baseline = [row for row in records if row["interval"] == 1]
    base_acc = statistics.fmean(row["final_accuracy_pct"] for row in baseline)
    base_bytes = statistics.fmean(row["total_bytes"] for row in baseline)
    for interval in INTERVALS:
        selected = [row for row in records if row["interval"] == interval]
        item = {"interval": interval, "valid_runs": len(selected)}
        for metric in metrics:
            values = [float(row[metric]) for row in selected if row[metric] is not None]
            item[f"{metric}_mean"] = statistics.fmean(values)
            item[f"{metric}_std"] = statistics.stdev(values)
        item["accuracy_change_vs_interval1_pp"] = item["final_accuracy_pct_mean"] - base_acc
        item["traffic_reduction_vs_interval1_pct"] = 100 * (
            1 - statistics.fmean(row["total_bytes"] for row in selected) / base_bytes)
        summary.append(item)
    with (OUT / "interval_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)

    colors = {1: "#2F5597", 3: "#E07A1F", 5: "#3A8F5B"}
    fig, axis = plt.subplots(figsize=(7.8, 4.8))
    for interval in INTERVALS:
        means = [statistics.fmean(curves[interval][seed][r] for seed in SEEDS) for r in range(60)]
        means = [value if value > 0 else float("nan") for value in means]
        axis.plot(range(1, 61), means, color=colors[interval], linewidth=2,
                  label=f"Interval-{interval}")
    axis.axvspan(FAILURE + 1, REJOIN, color="#bbbbbb", alpha=.25, label="6 nodes unavailable")
    axis.set(xlabel="Training round", ylabel="Global test accuracy (%)", xlim=(1, 60))
    axis.grid(alpha=.25); axis.legend(frameon=False); fig.tight_layout()
    fig.savefig(OUT / "e4_interval_learning_curves.png", dpi=240)
    fig.savefig(OUT / "e4_interval_learning_curves.pdf"); plt.close(fig)

    by_interval = {row["interval"]: row for row in summary}
    lines = [
        "# E4 global synchronization interval analysis", "",
        "三种子、60轮；第20轮后6/30节点离线，第30轮后恢复。每个 interval 使用同 interval、同冻结输入的无故障运行作为配对对照。", "",
        "| Interval | 最终准确率 | 故障窗口配对差值 | 最终配对差值 | 通信量 | 相对 I1 通信降低 |", "|---|---:|---:|---:|---:|---:|",
    ]
    for interval in INTERVALS:
        row = by_interval[interval]
        lines.append(
            f"| {interval} | {row['final_accuracy_pct_mean']:.2f}% ± {row['final_accuracy_pct_std']:.2f} "
            f"| {row['failure_window_mean_gap_pp_mean']:.2f} pp "
            f"| {row['final_gap_to_matched_control_pp_mean']:.2f} pp "
            f"| {row['total_traffic_gb_mean']:.2f} GB "
            f"| {row['traffic_reduction_vs_interval1_pct']:.2f}% |"
        )
    lines += [
        "", "## Interpretation", "",
        "- Interval-3/5 substantially reduce communication, with a modest final-accuracy cost relative to Interval-1.",
        "- The paired failure deficits remain small for all settings; this experiment does not show that node failure disproportionately damages Interval-5.",
        "- Recovery-round values for sparse intervals are bounded by evaluation cadence: two qualifying global-sync checkpoints require at least 5 rounds for I3 and 9 rounds for I5. They must not be interpreted as pure optimization recovery time.",
        "- The result supports a tunable communication/accuracy trade-off. It motivates, but does not by itself validate, an event-triggered adaptive interval.",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
