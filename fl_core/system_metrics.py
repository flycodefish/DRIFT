"""Algorithm-neutral raw system logs and offline-derived metrics."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


COMM_FIELDS = [
    "experiment_id", "method", "seed", "round", "timestamp_start",
    "timestamp_end", "src", "dst", "message_type", "message_id",
    "payload_bytes", "bandwidth_mbps", "latency_ms", "drop_rate", "phase",
    "flow_id", "logical_parent_message_id",
    "link_type", "success", "physical_hop_index", "physical_hop_count",
    "attempt_count", "backoff_ms",
]

NODE_FIELDS = [
    "experiment_id", "method", "seed", "round", "node_id", "role",
    "train_time_sim", "aggregation_time_sim", "rx_bytes", "tx_bytes",
    "rx_messages", "tx_messages", "rx_models", "tx_models",
    "aggregation_inputs", "intra_aggregation_inputs",
    "inter_aggregation_inputs", "model_upload_aggregation_inputs",
    "parent", "children_count", "active", "failed",
]

TOPOLOGY_FIELDS = [
    "experiment_id", "method", "seed", "round", "event_time", "event_type",
    "target_type", "target_id", "detected_time", "repair_start_time",
    "repair_end_time", "repair_time_wall_clock_seconds", "affected_nodes",
    "control_messages", "control_bytes", "old_state", "new_state", "source",
]


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _node_id(value: Any) -> str:
    text = str(value)
    return text.removeprefix("client_")


def build_node_round_stats(
    communication_events: list[dict[str, Any]],
    round_contexts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Derive node traffic/load rows from immutable communication events."""
    contexts = {int(item["round"]): item for item in round_contexts}
    counters: dict[tuple[int, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for event in communication_events:
        if not event.get("success", False):
            continue
        round_id = int(event["round"])
        payload = int(event.get("payload_bytes", 0) or 0)
        src, dst = _node_id(event["src"]), _node_id(event["dst"])
        counters[(round_id, src)]["tx_bytes"] += payload
        counters[(round_id, src)]["tx_messages"] += 1
        counters[(round_id, src)]["tx_models"] += int(payload > 0)
        counters[(round_id, dst)]["rx_bytes"] += payload
        counters[(round_id, dst)]["rx_messages"] += 1
        counters[(round_id, dst)]["rx_models"] += int(payload > 0)
        if str(event.get("message_type", "")).upper() in {
            "UPLOAD", "MODEL_UPLOAD", "PEER_MODEL", "AGGREGATED_MODEL"
        }:
            counters[(round_id, dst)]["aggregation_inputs"] += 1

    rows = []
    for round_id, context in sorted(contexts.items()):
        nodes = [str(node) for node in context.get("node_ids", [])]
        roles = {str(k): v for k, v in context.get("roles", {}).items()}
        parents = {str(k): v for k, v in context.get("parents", {}).items()}
        children = {str(k): v for k, v in context.get("children_counts", {}).items()}
        aggregation = {
            str(k): v for k, v in context.get("aggregation_inputs", {}).items()
        }
        aggregation_by_phase = {
            str(phase): {str(k): int(v) for k, v in values.items()}
            for phase, values in context.get(
                "aggregation_inputs_by_phase", {}
            ).items()
        }
        active = {str(node) for node in context.get("active_nodes", nodes)}
        failed = {str(node) for node in context.get("failed_nodes", [])}
        for node in nodes:
            values = counters[(round_id, node)]
            phase_counts = {
                phase: int(aggregation_by_phase.get(phase, {}).get(node, 0))
                for phase in ("INTRA_CLUSTER_UPLOAD", "INTER_CLUSTER_UPLOAD", "MODEL_UPLOAD")
            }
            total_aggregation = aggregation.get(
                node, sum(phase_counts.values()) or values["aggregation_inputs"]
            )
            rows.append({
                "experiment_id": context.get("experiment_id"),
                "method": context.get("method"), "seed": context.get("seed"),
                "round": round_id, "node_id": node,
                "role": roles.get(node, "CLIENT"),
                "train_time_sim": context.get("train_time_sim", {}).get(node),
                "aggregation_time_sim": context.get("aggregation_time_sim", {}).get(node),
                "rx_bytes": values["rx_bytes"], "tx_bytes": values["tx_bytes"],
                "rx_messages": values["rx_messages"], "tx_messages": values["tx_messages"],
                "rx_models": values["rx_models"], "tx_models": values["tx_models"],
                "aggregation_inputs": total_aggregation,
                "intra_aggregation_inputs": phase_counts["INTRA_CLUSTER_UPLOAD"],
                "inter_aggregation_inputs": phase_counts["INTER_CLUSTER_UPLOAD"],
                "model_upload_aggregation_inputs": phase_counts["MODEL_UPLOAD"],
                "parent": parents.get(node), "children_count": children.get(node, 0),
                "active": node in active, "failed": node in failed,
            })
    return rows


def _imbalance(values: list[float]) -> float | None:
    if not values or sum(values) == 0:
        return None
    return max(values) / (sum(values) / len(values))


def capacity_normalized_load_metrics(
    node_rows: list[dict[str, Any]],
    compute_capacity_by_node: dict[str, float],
) -> dict[str, Any]:
    """Summarize aggregation work after normalizing by compute capacity.

    ``assigned_work`` is the number of model inputs aggregated by a node. Local
    training is intentionally excluded because every participating client trains
    its own samples; this metric measures placement of the aggregation workload.
    """
    work_by_node: dict[str, float] = defaultdict(float)
    for row in node_rows:
        work_by_node[_node_id(row["node_id"])] += float(
            row.get("aggregation_inputs", 0) or 0
        )

    normalized: dict[str, float] = {}
    for node, capacity in compute_capacity_by_node.items():
        value = float(capacity)
        if value <= 0:
            raise ValueError(f"compute capacity must be positive for node {node}")
        normalized[_node_id(node)] = work_by_node.get(_node_id(node), 0.0) / value

    values = list(normalized.values())
    return {
        "max_capacity_normalized_load": max(values, default=0.0),
        "capacity_normalized_load_imbalance": _imbalance(values),
        "capacity_normalized_load_by_node": normalized,
        "capacity_normalized_load_definition": (
            "cumulative aggregation_inputs / relative compute_capacity"
        ),
    }


def _sum_known(rows: list[dict[str, Any]], key: str) -> int | float | None:
    if not rows:
        return 0
    values = [row.get(key) for row in rows]
    if any(value is None for value in values):
        return None
    return sum(values)


def _repair_control_cost(
    communication_events: list[dict[str, Any]],
    topology_events: list[dict[str, Any]],
    round_contexts: list[dict[str, Any]] | None = None,
) -> tuple[int | None, int | None, str]:
    """Return raw control-plane cost without mixing it with data traffic.

    A control-capable implementation with no CONTROL events is explicitly
    marked uninstrumented.  Legacy topology summaries are accepted only when
    they carry a non-zero value, avoiding double counting raw events.
    """
    control_events = [
        event for event in communication_events
        if str(event.get("phase", "")).strip().upper() == "CONTROL"
    ]
    if control_events:
        return (
            len(control_events),
            sum(int(event.get("payload_bytes", 0) or 0) for event in control_events),
            "instrumented",
        )
    legacy_messages = _sum_known(topology_events, "control_messages")
    legacy_bytes = _sum_known(topology_events, "control_bytes")
    if (legacy_messages not in (None, 0)) or (legacy_bytes not in (None, 0)):
        return legacy_messages, legacy_bytes, "legacy_summary"
    method = " ".join(str(event.get("method", "")) for event in communication_events).lower()
    method += " " + " ".join(str(event.get("method", "")) for event in topology_events).lower()
    method += " " + " ".join(str(context.get("method", "")) for context in (round_contexts or [])).lower()
    if "drift" in method or "de_asfl" in method or "deasfl" in method:
        return None, None, "uninstrumented"
    return 0, 0, "not_applicable"


def build_metrics_summary(
    communication_events: list[dict[str, Any]],
    node_rows: list[dict[str, Any]],
    round_contexts: list[dict[str, Any]],
    topology_events: list[dict[str, Any]],
) -> dict[str, Any]:
    successful = [item for item in communication_events if item.get("success", False)]
    rounds = sorted({int(item["round"]) for item in round_contexts})
    per_round = []
    for round_id in rounds:
        events = [item for item in successful if int(item["round"]) == round_id]
        nodes = [item for item in node_rows if int(item["round"]) == round_id]
        traffic = [int(item["rx_bytes"]) + int(item["tx_bytes"]) for item in nodes]
        agg = [int(item["aggregation_inputs"]) for item in nodes]
        context = next(item for item in round_contexts if int(item["round"]) == round_id)
        per_round.append({
            "round": round_id,
            "total_messages": len(events),
            "total_bytes": sum(int(item.get("payload_bytes", 0) or 0) for item in events),
            "peak_node_traffic_bytes": max(traffic, default=0),
            "communication_load_imbalance": _imbalance(traffic),
            "aggregation_load_imbalance": _imbalance(agg),
            "eligible_node_ratio": context.get("eligible_node_ratio"),
            "reachable_node_ratio": context.get("reachable_node_ratio"),
            "eligible_sample_ratio": context.get("eligible_sample_ratio"),
            "aggregated_sample_ratio": context.get("aggregated_sample_ratio"),
            "round_completion_ratio": context.get("round_completion_ratio"),
            "round_wall_clock_seconds": context.get("round_wall_clock_seconds"),
            "round_latency_sim_ms": context.get("round_latency_sim_ms"),
        })
    total_by_node: dict[str, int] = defaultdict(int)
    aggregate_by_node: dict[str, int] = defaultdict(int)
    for row in node_rows:
        total_by_node[str(row["node_id"])] += int(row["rx_bytes"]) + int(row["tx_bytes"])
        aggregate_by_node[str(row["node_id"])] += int(row["aggregation_inputs"])
    repair_messages, repair_bytes, repair_control_status = _repair_control_cost(
        communication_events, topology_events, round_contexts
    )
    repair_wall_clock = _sum_known(topology_events, "repair_time_wall_clock_seconds")
    if repair_control_status == "uninstrumented":
        repair_wall_clock = None
    return {
        "schema_version": 1,
        "definitions": {
            "traffic": "successful payload bytes counted once per recorded hop",
            "event_scope": "link_type distinguishes direct runtime events from DRIFT logical accounting",
            "round_latency_sim_ms": "null until a shared compute and dependency-DAG model is frozen",
            "round_wall_clock_seconds": "diagnostic only; not a paper system-performance metric",
            "energy": "unsupported in Protocol V1; no power model is assumed",
            "accuracy_recovery": "requires a matched no-failure control run",
            "repair_control_cost": "CONTROL communication events counted once; null means control-plane instrumentation is absent",
        },
        "total_messages": len(successful),
        "failed_messages": len(communication_events) - len(successful),
        "total_bytes": sum(int(item.get("payload_bytes", 0) or 0) for item in successful),
        "wan_bytes": sum(int(item.get("payload_bytes", 0) or 0) for item in successful
                         if str(item.get("link_type", "")).upper() == "WAN"),
        "cross_cluster_bytes": sum(
            int(item.get("payload_bytes", 0) or 0) for item in successful
            if str(item.get("link_type", "")).upper() in {"CROSS_CLUSTER", "INTER_CLUSTER"}
        ),
        "mean_peak_node_traffic_bytes": (
            sum(item["peak_node_traffic_bytes"] for item in per_round) / len(per_round)
            if per_round else 0
        ),
        "max_peak_node_traffic_bytes": max(
            (item["peak_node_traffic_bytes"] for item in per_round), default=0
        ),
        "communication_load_imbalance": _imbalance(list(total_by_node.values())),
        "aggregation_load_imbalance": _imbalance(list(aggregate_by_node.values())),
        "repair_messages": repair_messages,
        "repair_bytes": repair_bytes,
        "repair_control_status": repair_control_status,
        "repair_time_wall_clock_seconds": repair_wall_clock,
        "topology_recovery_time_sim_ms": None,
        "accuracy_drop_pp": None,
        "accuracy_recovery_rounds": None,
        "accuracy_recovery_status": "requires_control_run",
        "topology_event_count": len(topology_events),
        "energy_status": "unsupported",
        "per_round": per_round,
    }


def dump_system_metrics(
    output_dir: str,
    communication_events: list[dict[str, Any]],
    round_contexts: list[dict[str, Any]],
    topology_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write the protocol V1 raw tables and offline-derived summary."""
    topology_events = topology_events or []
    node_rows = build_node_round_stats(communication_events, round_contexts)
    summary = build_metrics_summary(
        communication_events, node_rows, round_contexts, topology_events
    )
    root = Path(output_dir)
    _write_csv(root / "raw" / "communication_events.csv", COMM_FIELDS, communication_events)
    _write_csv(root / "raw" / "node_round_stats.csv", NODE_FIELDS, node_rows)
    _write_csv(root / "raw" / "topology_events.csv", TOPOLOGY_FIELDS, topology_events)
    (root / "raw" / "round_contexts.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "raw" / "round_contexts.json").write_text(
        json.dumps(round_contexts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics_path = root / "metrics" / "metrics_summary.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
