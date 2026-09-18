"""Validate and aggregate completed scalability trace runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


METRICS = (
    "total_messages", "total_bytes", "mean_peak_node_traffic_bytes",
    "max_peak_node_traffic_bytes", "max_capacity_normalized_load",
    "capacity_normalized_load_imbalance", "mean_round_latency_sim_ms",
    "p95_round_latency_sim_ms",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--clients", default="10,30,50")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--methods", default="fedavg_edge,resource_tree,drift")
    args = parser.parse_args()
    clients = [int(value) for value in args.clients.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    methods = [value.strip() for value in args.methods.split(",")]
    records = []
    for count in clients:
        for seed in seeds:
            for method in methods:
                run = args.root / f"n{count}" / f"seed_{seed}" / method
                status_path = run / "config" / "completion_status.json"
                metrics_path = run / "metrics" / "metrics_summary.json"
                if not status_path.is_file() or not metrics_path.is_file():
                    raise FileNotFoundError(f"incomplete scalability run: {run}")
                status = json.loads(status_path.read_text())
                metrics = json.loads(metrics_path.read_text())
                if status.get("status") != "COMPLETE" or not metrics.get("trace_only"):
                    raise ValueError(f"invalid scalability run: {run}")
                if int(metrics.get("num_clients", -1)) != count:
                    raise ValueError(f"client-count mismatch: {run}")
                record = {"num_clients": count, "seed": seed, "method": method,
                          **{key: metrics.get(key) for key in METRICS}}
                records.append(record)
    aggregate = []
    for count in clients:
        for method in methods:
            selected = [row for row in records
                        if row["num_clients"] == count and row["method"] == method]
            row = {"num_clients": count, "method": method, "valid_seeds": len(selected)}
            for key in METRICS:
                values = [float(item[key]) for item in selected if item[key] is not None]
                row[f"{key}_mean"] = statistics.fmean(values)
                row[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            aggregate.append(row)
    args.root.mkdir(parents=True, exist_ok=True)
    payload = {"status": "COMPLETE", "records": records, "aggregate": aggregate}
    (args.root / "scalability_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, values in (("scalability_records.csv", records),
                         ("scalability_aggregate.csv", aggregate)):
        with (args.root / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader(); writer.writerows(values)
    print(f"COMPLETE: {len(records)} runs -> {args.root / 'scalability_summary.json'}")


if __name__ == "__main__":
    main()
