"""Recompute comparable learning and system metrics from run artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fl_core.simulation import simulate_round
from fl_core.artifacts import normalize_physical_graph
from fl_core.system_metrics import (
    build_metrics_summary,
    build_node_round_stats,
    capacity_normalized_load_metrics,
)


def rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_one(run: Path, pattern: str) -> Path:
    matches = sorted(run.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one {pattern} under {run}, found {len(matches)}")
    return matches[0]


def accuracy_curve(run: Path) -> list[float]:
    values = rows(find_one(run, "*_round_metrics.csv"))
    if not values:
        raise ValueError(f"No round metrics found under {run}")
    round_key = "round" if "round" in values[0] else "round_id"
    round_ids = [int(item[round_key]) for item in values]
    if round_ids != list(range(len(values))):
        raise ValueError(f"Non-contiguous round ids under {run}: {round_ids}")
    return [float(item["val_acc"]) for item in values]


def failure_event_rounds(topology_events: list[dict]) -> list[int]:
    result = []
    for item in topology_events:
        event_type = str(item.get("event_type", "")).strip().lower()
        if event_type not in {"node_failure", "link_failure"}:
            continue
        round_value = item.get("round")
        if round_value not in (None, ""):
            result.append(int(round_value))
    return sorted(set(result))


def node_failure_rounds(topology_events: list[dict]) -> list[int]:
    """Backward-compatible node-only failure helper."""
    return sorted({int(item["round"]) for item in topology_events
                   if str(item.get("event_type", "")).strip().lower() == "node_failure"
                   and item.get("round") not in (None, "")})


def _completion_status(run: Path) -> dict[str, Any]:
    path = run / "config" / "completion_status.json"
    if not path.is_file():
        raise ValueError(f"Missing completion status for {run}: {path}")
    status = json.loads(path.read_text(encoding="utf-8"))
    if str(status.get("status", "")).upper() != "COMPLETE":
        raise ValueError(f"Run {run} is not COMPLETE: {status.get('status')!r}")
    if status.get("requested_rounds") is not None and status.get("completed_rounds") is not None:
        if int(status["requested_rounds"]) != int(status["completed_rounds"]):
            raise ValueError(f"Run {run} has incomplete rounds: {status['completed_rounds']}/{status['requested_rounds']}")
        metrics_files = sorted(run.glob("*_round_metrics.csv"))
        if len(metrics_files) == 1:
            actual_rounds = len(rows(metrics_files[0]))
            if actual_rounds != int(status["completed_rounds"]):
                raise ValueError(f"Run {run} status/curve mismatch: {actual_rounds}/{status['completed_rounds']}")
    return status


_IMMUTABLE_INPUTS = (
    "data_partition.json", "initial_model.pt", "physical_graph.json",
    "node_resources.json", "link_profiles.json",
)
_FAILURE_INPUT = "failure_trace.json"
_CONFIG_KEYS = (
    "dataset", "dataset_split_policy", "har_local_val_seed", "model",
    "input_dim", "num_classes", "num_clients", "rounds", "local_epochs",
    "batch_size", "learning_rate", "weight_decay", "learning_rate_min",
    "warmup_rounds", "cosine_annealing", "learning_rate_schedule_rounds",
    "data_partition_mode", "dirichlet_alpha", "classes_per_client", "seed",
    "deterministic_training", "client_sample_ratio", "client_drop_rate",
    "use_gpu",
)


def _canonical_config_value(config: dict[str, Any], key: str) -> Any:
    """Normalize equivalent config names used by FedAvg and DRIFT."""
    if key == "har_local_val_seed":
        dataset = str(config.get("dataset", "")).strip().lower()
        return config.get(key) if dataset == "har" else None
    if key == "data_partition_mode":
        value = (config.get("data_partition_mode") or config.get("partition")
                 or config.get("partition_method"))
        if value is None and config.get("alpha") is not None:
            value = "dirichlet"
        return str(value).lower() if value is not None else None
    if key == "dirichlet_alpha":
        value = config.get("dirichlet_alpha")
        if value is None:
            value = config.get("alpha")
        return float(value) if value is not None else None
    value = config.get(key)
    # DRIFT historically omitted these fields because all clients participate
    # and client sampling/dropout is not part of its protocol.
    if key == "client_sample_ratio" and value is None:
        return 1.0
    if key == "client_drop_rate" and value is None:
        return 0.0
    if key == "dataset_split_policy" and value is None:
        return "default"
    if key == "classes_per_client":
        # This setting is irrelevant for Dirichlet partitioning.
        mode = _canonical_config_value(config, "data_partition_mode")
        return value if mode not in {"dirichlet", "iid", "uniform"} else None
    if key in {"learning_rate", "learning_rate_min", "weight_decay",
               "dirichlet_alpha", "client_sample_ratio", "client_drop_rate"} and value is not None:
        return float(value)
    return value


def immutable_input_hashes(run: Path, include_failure_trace: bool = False) -> dict[str, str]:
    hashes = {}
    names = _IMMUTABLE_INPUTS + ((_FAILURE_INPUT,) if include_failure_trace else ())
    for name in names:
        path = run / "config" / name
        if not path.is_file():
            raise ValueError(f"Missing immutable input for comparison: {path}")
        if name == "data_partition.json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid data partition artifact: {path}") from exc
            # local_val_seed is meaningful for HAR only.  CIFAR runtimes may
            # either emit or omit the default without changing any sample.
            if str(payload.get("dataset", "")).strip().lower() != "har":
                payload.pop("local_val_seed", None)
            encoded = json.dumps(
                payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            hashes[name] = hashlib.sha256(encoded).hexdigest()
        else:
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def validate_comparable_runs(
    run: Path,
    control: Path,
    label: str,
    allow_failure_trace_difference: bool = False,
) -> None:
    _completion_status(run)
    _completion_status(control)
    current, reference = accuracy_curve(run), accuracy_curve(control)
    if len(current) != len(reference):
        raise ValueError(f"{label} curve length mismatch: {len(current)} != {len(reference)}")
    include_failure_trace = not allow_failure_trace_difference
    left, right = immutable_input_hashes(run, include_failure_trace), immutable_input_hashes(control, include_failure_trace)
    compared_inputs = _IMMUTABLE_INPUTS + ((_FAILURE_INPUT,) if include_failure_trace else ())
    mismatched = [name for name in compared_inputs if left[name] != right[name]]
    if mismatched:
        raise ValueError(f"{label} immutable inputs differ: {', '.join(mismatched)}")
    left_path, right_path = run / "config" / "experiment.json", control / "config" / "experiment.json"
    if not left_path.is_file() or not right_path.is_file():
        raise ValueError("Both comparison runs must contain config/experiment.json")
    left_cfg, right_cfg = json.loads(left_path.read_text()), json.loads(right_path.read_text())
    mismatched = [key for key in _CONFIG_KEYS
                  if _canonical_config_value(left_cfg, key) != _canonical_config_value(right_cfg, key)]
    if mismatched:
        raise ValueError(f"{label} config differs in: {', '.join(mismatched)}")


def target_metrics(accuracy: list[float], target: float, simulated: list[dict] | None = None) -> dict[str, Any]:
    hit = next((i for i, value in enumerate(accuracy) if value >= target), None)
    result: dict[str, Any] = {"target_accuracy": target, "rounds_to_target": hit + 1 if hit is not None else None, "target_status": "reached" if hit is not None else "not_reached", "time_to_target_sim_ms": None}
    if hit is not None and simulated is not None:
        result["time_to_target_sim_ms"] = sum(float(item.get("round_latency_sim_ms", 0) or 0) for item in simulated[:hit + 1])
    return result


def round_latency_metrics(simulated: list[dict]) -> dict[str, float | None]:
    """Summarize the simulated per-round critical-path latency."""
    values = sorted(
        float(item["round_latency_sim_ms"])
        for item in simulated
        if item.get("round_latency_sim_ms") is not None
    )
    if not values:
        return {
            "mean_round_latency_sim_ms": None,
            "p95_round_latency_sim_ms": None,
        }
    position = 0.95 * (len(values) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    p95 = values[lower] + (values[upper] - values[lower]) * (position - lower)
    return {
        "mean_round_latency_sim_ms": sum(values) / len(values),
        "p95_round_latency_sim_ms": p95,
    }


def recovery_metrics(accuracy: list[float], control: list[float] | None, topology_events: list[dict], gap_pp: float, patience: int) -> dict[str, Any]:
    failures = failure_event_rounds(topology_events)
    result: dict[str, Any] = {"failure_rounds": failures, "accuracy_drop_pp": None, "max_drop_round": None, "accuracy_recovery_rounds": None, "accuracy_recovery_round": None, "accuracy_recovery_status": "no_failure_event" if not failures else "requires_control_run", "recovery_gap_pp": gap_pp, "recovery_patience": patience}
    if not failures or control is None:
        return result
    if len(accuracy) != len(control):
        raise ValueError("Recovery control curve length must equal the run curve")
    first = min(failures)
    if first >= len(accuracy):
        result["accuracy_recovery_status"] = "insufficient_post_failure_rounds"
        return result
    gaps = [(control[i] - accuracy[i]) * 100 for i in range(len(accuracy))]
    drop_index = max(range(first, len(gaps)), key=lambda i: gaps[i])
    result["accuracy_drop_pp"] = max(0.0, gaps[drop_index])
    result["max_drop_round"] = drop_index if gaps[drop_index] > 0 else None
    if len(accuracy) - first < patience:
        result["accuracy_recovery_status"] = "insufficient_post_failure_rounds"
        return result
    for index in range(first, len(gaps) - patience + 1):
        if all(gaps[j] <= max(0.0, float(gap_pp)) for j in range(index, index + patience)):
            confirmation_round = index + patience - 1
            result["accuracy_recovery_round"] = confirmation_round
            result["accuracy_recovery_rounds"] = confirmation_round - first
            result["accuracy_recovery_status"] = "recovered"
            return result
    result["accuracy_recovery_status"] = "not_recovered"
    return result


def topology_recovery_metrics(topology_events: list[dict], simulated: list[dict] | None = None) -> dict[str, Any]:
    events = sorted(topology_events, key=lambda item: int(item.get("round", 0) or 0))
    pairs = {"node_failure": "node_recovery", "link_failure": "link_recovery"}
    failure_events = [item for item in events if str(item.get("event_type", "")).strip().lower() in pairs]
    consumed_recoveries: set[int] = set()
    matched = []
    for failure in events:
        kind = str(failure.get("event_type", "")).strip().lower()
        if kind not in pairs:
            continue
        failure_round = int(failure.get("round", 0))
        target = (str(failure.get("target_type", "")).lower(), str(failure.get("target_id", "")))
        for recovery_index, recovery in enumerate(events):
            if recovery_index in consumed_recoveries:
                continue
            recovery_kind = str(recovery.get("event_type", "")).strip().lower()
            if recovery_kind != pairs[kind] or int(recovery.get("round", 0) or 0) < failure_round:
                continue
            recovery_target = (str(recovery.get("target_type", "")).lower(), str(recovery.get("target_id", "")))
            if target == ("", "") or recovery_target == target or recovery_target[1] == target[1]:
                matched.append((failure_round, int(recovery.get("round", 0))))
                consumed_recoveries.add(recovery_index)
                break
    if not matched:
        return {"availability_recovery_rounds": None, "topology_recovery_time_sim_ms": None, "availability_recovery_status": "no_failure_event" if not failure_events else "not_recovered", "availability_recovery_unmatched_failures": len(failure_events)}
    durations = [recovery - failure for failure, recovery in matched]
    fully_recovered = len(matched) == len(failure_events)
    sim_time = None
    if simulated is not None and fully_recovered:
        sim_time = max(sum(float(item.get("round_latency_sim_ms", 0) or 0) for item in simulated[failure:recovery]) for failure, recovery in matched)
    return {"availability_recovery_rounds": max(durations) if fully_recovered else None, "topology_recovery_time_sim_ms": sim_time, "availability_recovery_status": "recovered" if fully_recovered else "not_recovered", "availability_recovery_unmatched_failures": len(failure_events) - len(matched)}


def _validate_calibration_artifacts(assumptions: dict[str, Any], config_path: Path) -> None:
    artifacts = assumptions.get("calibration_artifacts")
    if artifacts is None and assumptions.get("calibration_artifact") is not None:
        artifacts = [assumptions["calibration_artifact"]]
    hashes = assumptions.get("calibration_sha256")
    if isinstance(hashes, str):
        hashes = [hashes]
    if not isinstance(artifacts, list) or not isinstance(hashes, list) or len(artifacts) != len(hashes) or not artifacts:
        raise ValueError(
            "Frozen simulation configuration must list calibration_artifact(s) "
            "and matching calibration_sha256 value(s)"
        )
    for artifact, expected in zip(artifacts, hashes):
        path = Path(str(artifact))
        if not path.is_absolute():
            # Calibration paths are recorded relative to the repository root;
            # fall back to the config directory for self-contained bundles.
            repo_path = REPO_ROOT / path
            path = repo_path if repo_path.is_file() else config_path.parent / path
        if not path.is_file():
            raise ValueError(f"Missing simulation calibration artifact: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual.lower() != str(expected).lower():
            raise ValueError(f"Calibration artifact SHA256 mismatch: {path}")


def _shortest_physical_path(graph: dict[int, list[int]], start: int, end: int) -> list[int]:
    if start == end:
        return [start]
    queue = [start]
    parent = {start: None}
    while queue:
        node = queue.pop(0)
        for neighbor in sorted(graph.get(node, [])):
            if neighbor in parent:
                continue
            parent[neighbor] = node
            if neighbor == end:
                path = [end]
                while path[-1] != start:
                    path.append(parent[path[-1]])
                return list(reversed(path))
            queue.append(neighbor)
    raise ValueError(f"FedAvg-Edge server node {start} cannot reach client {end}")


def _reinterpret_fedavg_edge(
    events: list[dict[str, Any]], contexts: dict[int, dict[str, Any]],
    run: Path, server_node: int, sample_count: int,
) -> list[dict[str, Any]]:
    """Route the logical FedAvg star through one frozen physical edge node."""
    raw_graph = json.loads((run / "config" / "physical_graph.json").read_text())
    # Frozen artifacts have existed in both adjacency-list and mapping forms.
    # Normalize both forms and close one-sided edges before routing.
    graph = normalize_physical_graph(raw_graph, sample_count)
    if server_node not in graph:
        raise ValueError(f"FedAvg-Edge server node {server_node} is outside the physical graph")
    links = json.loads((run / "config" / "link_profiles.json").read_text())
    default = links.get("default", links)
    overrides = links.get("edge_overrides", {})
    expanded: list[dict[str, Any]] = []
    for event in events:
        src, dst = str(event["src"]), str(event["dst"])
        if src == "server" and dst.startswith("client_"):
            path = _shortest_physical_path(graph, server_node, int(dst.removeprefix("client_")))
        elif dst == "server" and src.startswith("client_"):
            path = _shortest_physical_path(graph, int(src.removeprefix("client_")), server_node)
        else:
            raise ValueError("FedAvg-Edge requires pure server/client FedAvg communication events")
        hop_count = len(path) - 1
        for hop_index, (left, right) in enumerate(zip(path[:-1], path[1:]), start=1):
            edge_key = f"{min(left, right)}->{max(left, right)}"
            profile = {**default, **overrides.get(edge_key, {})}
            item = dict(event)
            item.update({
                "src": f"client_{left}", "dst": f"client_{right}",
                "message_id": f"{event.get('message_id', 'message')}:edge:{hop_index}",
                "flow_id": event.get("flow_id") or f"fedavg-edge:{event.get('message_id')}",
                "bandwidth_mbps": float(profile.get(
                    "bandwidth_mbps", profile.get("default_bandwidth_mbps", 100.0)
                )),
                "latency_ms": float(profile.get(
                    "latency_ms", profile.get("default_latency_ms", 5.0)
                )),
                "drop_rate": float(profile.get("drop_rate", 0.0)),
                "link_type": "PHYSICAL", "physical_hop_index": hop_index,
                "physical_hop_count": hop_count,
            })
            expanded.append(item)
    for context in contexts.values():
        node_keys = {str(node) for node in context.get("node_ids", [])}
        aggregation_node = (
            str(server_node) if str(server_node) in node_keys
            else f"client_{server_node}"
        )
        by_phase = context.setdefault("aggregation_inputs_by_phase", {})
        by_phase["MODEL_UPLOAD"] = {aggregation_node: int(sample_count)}
    return expanded


def _fedavg_edge_system_summary(
    run: Path, server_node: int, sample_count: int,
) -> dict[str, Any]:
    """Recompute traffic metrics after expanding FedAvg over physical hops."""
    events = rows(run / "raw" / "communication_events.csv")
    for event in events:
        event["success"] = str(event.get("success", "")).lower() == "true"
        event["payload_bytes"] = int(event.get("payload_bytes") or 0)
    contexts_path = run / "raw" / "round_contexts.json"
    contexts = (
        {int(item["round"]): item for item in json.loads(contexts_path.read_text(encoding="utf-8"))}
        if contexts_path.is_file() else {}
    )
    if not contexts:
        # Older FedAvg artifacts predate round_contexts.json.  Reconstruct the
        # minimal context needed for traffic/peak-load metrics from the event
        # rounds so those metrics are not silently reported as zero.
        rounds = sorted({int(item["round"]) for item in events})
        nodes = [str(node_id) for node_id in range(sample_count)]
        contexts = {
            round_id: {
                "round": round_id, "node_ids": nodes,
                "active_nodes": nodes, "failed_nodes": [],
                "roles": {node: "CLIENT" for node in nodes},
                "parents": {}, "children_counts": {},
                "aggregation_inputs_by_phase": {},
            }
            for round_id in rounds
        }
    expanded = _reinterpret_fedavg_edge(events, contexts, run, server_node, sample_count)
    node_rows = build_node_round_stats(expanded, list(contexts.values()))
    topology_path = run / "raw" / "topology_events.csv"
    topology = rows(topology_path) if topology_path.is_file() else []
    summary = build_metrics_summary(expanded, node_rows, list(contexts.values()), topology)
    resources = json.loads((run / "config" / "node_resources.json").read_text())
    capacities = {}
    for position, item in enumerate(resources if isinstance(resources, list) else []):
        params = item.get("resource_params", item)
        capacity = params.get("compute_capacity")
        if capacity is not None:
            capacities[str(item.get("node_id", position))] = float(capacity)
    summary.update(capacity_normalized_load_metrics(node_rows, capacities))
    return summary


def _load_simulation(args: argparse.Namespace, accuracy: list[float]) -> tuple[list[dict], dict[str, Any]]:
    assumptions = json.loads(args.simulation_config.read_text(encoding="utf-8"))
    if str(assumptions.get("status", "")).lower() != "frozen":
        raise ValueError(
            "Simulation configuration is not frozen; use a reviewed frozen "
            "configuration for formal metrics"
        )
    _validate_calibration_artifacts(assumptions, args.simulation_config)
    experiment = json.loads((args.run / "config" / "experiment.json").read_text(encoding="utf-8"))
    reference_model = assumptions.get("reference_model")
    if reference_model and str(reference_model).lower() != str(experiment.get("model", "")).lower():
        raise ValueError(
            f"Simulation reference_model {reference_model!r} does not match "
            f"run model {experiment.get('model')!r}"
        )
    reference_batch = assumptions.get("reference_batch_size")
    if reference_batch is not None and int(reference_batch) != int(experiment.get("batch_size", reference_batch)):
        raise ValueError(
            f"Simulation reference_batch_size {reference_batch} does not match "
            f"run batch_size {experiment.get('batch_size')!r}"
        )
    for field in ("reference_input_dim", "reference_num_classes"):
        expected = assumptions.get(field)
        config_key = field.removeprefix("reference_")
        if expected is not None and config_key in experiment and int(expected) != int(experiment[config_key]):
            raise ValueError(
                f"Simulation {field} {expected!r} does not match "
                f"run {config_key} {experiment[config_key]!r}"
            )
    resources = json.loads((args.run / "config" / "node_resources.json").read_text(encoding="utf-8"))
    reference = float(assumptions.get("resource_compute_capacity_reference", 1.0))
    capacities = {str(k): float(v) for k, v in assumptions.get("compute_capacity_by_node", {}).items()}
    aggregation = {str(k): float(v) for k, v in assumptions.get("aggregation_capacity_by_node", {}).items()}
    network = {str(k): float(v) for k, v in assumptions.get("network_capacity_by_node", {}).items()}
    for position, item in enumerate(resources if isinstance(resources, list) else []):
        params = item.get("resource_params", item)
        node_id = item.get("node_id", position)
        capacity = params.get("compute_capacity", params.get("cpu_cores"))
        if capacity is not None:
            value = float(capacity)
            normalized = params.get("compute_capacity_semantics") == "relative_throughput"
            if params.get("compute_capacity_semantics") is None and params.get("cpu_cores") is not None:
                normalized = abs(value - float(params["cpu_cores"]) / reference) < 1e-12
            capacities.setdefault(str(node_id), value if normalized else value / reference)
            aggregation.setdefault(str(node_id), capacities[str(node_id)])
        bandwidth = params.get("network_capacity_mbps", params.get("bandwidth_mbps"))
        if bandwidth is not None:
            network.setdefault(str(node_id), float(bandwidth))
    assumptions.update({"compute_capacity_by_node": capacities, "aggregation_capacity_by_node": aggregation, "network_capacity_by_node": network})
    edge_server = getattr(args, "fedavg_edge_server_node", None)
    if edge_server is not None:
        if str(experiment.get("algorithm", "")).lower() != "fedavg":
            raise ValueError("--fedavg-edge-server-node is only valid for a FedAvg run")
        edge_key = str(int(edge_server))
        if edge_key not in capacities or edge_key not in network:
            raise ValueError(f"FedAvg-Edge server node {edge_server} has no frozen resource profile")
        assumptions["compute_capacity_by_node"]["server"] = capacities[edge_key]
        assumptions["aggregation_capacity_by_node"]["server"] = aggregation[edge_key]
        assumptions["network_capacity_by_node"]["server"] = network[edge_key]
        assumptions["server_compute_capacity"] = capacities[edge_key]
        assumptions["server_network_capacity_mbps"] = network[edge_key]
        assumptions["system_interpretation"] = "fedavg_edge"
        assumptions["fedavg_edge_server_node"] = int(edge_server)
    events = rows(args.run / "raw" / "communication_events.csv")
    for event in events:
        event["success"] = str(event.get("success", "")).lower() == "true"
        event["payload_bytes"] = int(event.get("payload_bytes") or 0)
        event["bandwidth_mbps"] = float(event["bandwidth_mbps"]) if event.get("bandwidth_mbps") else None
        event["latency_ms"] = float(event["latency_ms"]) if event.get("latency_ms") else None
        event["attempt_count"] = int(event.get("attempt_count") or 1)
        event["backoff_ms"] = float(event.get("backoff_ms") or 0)
    node_rows = rows(args.run / "raw" / "node_round_stats.csv")
    partition = json.loads((args.run / "config" / "data_partition.json").read_text())
    contexts_path = args.run / "raw" / "round_contexts.json"
    contexts = {int(item["round"]): item for item in json.loads(contexts_path.read_text(encoding="utf-8"))} if contexts_path.is_file() else {}
    if edge_server is not None:
        events = _reinterpret_fedavg_edge(
            events, contexts, args.run, int(edge_server), len(partition["sample_counts"])
        )
    simulated = [simulate_round([item for item in events if int(item["round"]) == r], [item for item in node_rows if int(item["round"]) == r], partition["sample_counts"], int(experiment.get("local_epochs", 1)), assumptions, contexts.get(r)) for r in range(len(accuracy))]
    return simulated, assumptions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--fedavg-control", type=Path, help="FedAvg accuracy reference only")
    parser.add_argument("--no-failure-control", type=Path, help="Matched no-failure run for recovery")
    parser.add_argument("--target-accuracy", type=float)
    parser.add_argument("--recovery-gap-pp", type=float, default=1.0)
    parser.add_argument("--recovery-patience", type=int, default=3)
    parser.add_argument("--simulation-config", type=Path)
    parser.add_argument("--fedavg-edge-server-node", type=int,
                        help="Reuse a FedAvg run with the server constrained by this frozen edge node")
    parser.add_argument("--recovery-threshold-ratio", type=float, help="Deprecated compatibility option")
    args = parser.parse_args()
    if args.recovery_patience < 1:
        raise ValueError("--recovery-patience must be >= 1")
    if args.recovery_gap_pp < 0:
        raise ValueError("--recovery-gap-pp must be >= 0")
    if args.target_accuracy is not None and not 0 <= args.target_accuracy <= 1:
        raise ValueError("--target-accuracy must be between 0 and 1")
    if args.recovery_threshold_ratio is not None:
        raise ValueError("--recovery-threshold-ratio is obsolete; use --recovery-gap-pp")
    if args.target_accuracy is not None or args.fedavg_control or args.no_failure_control:
        _completion_status(args.run)
    native_metrics_path = args.run / "metrics" / "metrics_summary.json"
    metrics_path = (args.run / "metrics" / "fedavg_edge_metrics.json"
                    if args.fedavg_edge_server_node is not None else native_metrics_path)
    metrics = json.loads(native_metrics_path.read_text(encoding="utf-8"))
    if args.fedavg_edge_server_node is not None and args.simulation_config is None:
        raise ValueError("--fedavg-edge-server-node requires --simulation-config")
    for key in (
        "accuracy_gap_to_fedavg_pp", "accuracy_gap_curve_pp",
        "target_accuracy", "rounds_to_target", "target_status",
        "time_to_target_sim_ms", "simulated_rounds",
        "simulation_assumptions", "simulation_config", "total_simulated_training_time_ms",
        "mean_round_latency_sim_ms", "p95_round_latency_sim_ms",
        "failure_rounds", "accuracy_drop_pp", "max_drop_round",
        "accuracy_recovery_rounds", "accuracy_recovery_round",
        "accuracy_recovery_status", "recovery_gap_pp", "recovery_patience",
        "availability_recovery_rounds", "topology_recovery_time_sim_ms",
        "availability_recovery_status", "availability_recovery_unmatched_failures",
        "max_capacity_normalized_load", "capacity_normalized_load_imbalance",
        "capacity_normalized_load_by_node", "capacity_normalized_load_definition",
    ):
        metrics.pop(key, None)
    resources = json.loads(
        (args.run / "config" / "node_resources.json").read_text(encoding="utf-8")
    )
    compute_capacities = {}
    for position, item in enumerate(resources if isinstance(resources, list) else []):
        params = item.get("resource_params", item)
        node_id = str(item.get("node_id", position))
        capacity = params.get("compute_capacity")
        if capacity is not None:
            compute_capacities[node_id] = float(capacity)
    node_stats = rows(args.run / "raw" / "node_round_stats.csv")
    metrics.update(capacity_normalized_load_metrics(node_stats, compute_capacities))
    accuracy = accuracy_curve(args.run)
    metrics["final_accuracy"], metrics["best_accuracy"] = accuracy[-1], max(accuracy)
    if args.fedavg_control:
        validate_comparable_runs(args.run, args.fedavg_control, "FedAvg control")
        control = accuracy_curve(args.fedavg_control)
        gaps = [(control[i] - accuracy[i]) * 100 for i in range(len(accuracy))]
        metrics["accuracy_gap_to_fedavg_pp"], metrics["accuracy_gap_curve_pp"] = gaps[-1], gaps
    topology = rows(args.run / "raw" / "topology_events.csv")
    simulation_assumptions = None
    edge_system_summary = None
    if args.simulation_config:
        simulated, simulation_assumptions = _load_simulation(args, accuracy)
        if args.fedavg_edge_server_node is not None:
            edge_system_summary = _fedavg_edge_system_summary(
                args.run, int(args.fedavg_edge_server_node),
                len(json.loads((args.run / "config" / "data_partition.json").read_text())["sample_counts"]),
            )
    else:
        simulated = None
    if simulated is not None:
        metrics["simulated_rounds"] = simulated
        metrics["simulation_assumptions"] = simulation_assumptions
        metrics["simulation_config"] = str(args.simulation_config)
        metrics["total_simulated_training_time_ms"] = sum(item["round_latency_sim_ms"] for item in simulated)
        metrics.update(round_latency_metrics(simulated))
        if args.fedavg_edge_server_node is not None:
            metrics["system_interpretation"] = "fedavg_edge"
            metrics["fedavg_edge_server_node"] = args.fedavg_edge_server_node
            if edge_system_summary is not None:
                for key in (
                    "definitions", "total_messages", "failed_messages", "total_bytes",
                    "wan_bytes", "cross_cluster_bytes", "mean_peak_node_traffic_bytes",
                    "max_peak_node_traffic_bytes", "communication_load_imbalance",
                    "aggregation_load_imbalance", "per_round",
                    "max_capacity_normalized_load",
                    "capacity_normalized_load_imbalance",
                    "capacity_normalized_load_by_node",
                    "capacity_normalized_load_definition",
                ):
                    metrics[key] = edge_system_summary[key]
    if args.target_accuracy is not None:
        metrics.update(target_metrics(accuracy, args.target_accuracy, simulated))
    recovery_control = None
    if args.no_failure_control:
        validate_comparable_runs(
            args.run, args.no_failure_control, "no-failure control",
            allow_failure_trace_difference=True,
        )
        control_topology = rows(args.no_failure_control / "raw" / "topology_events.csv")
        if failure_event_rounds(control_topology):
            raise ValueError("--no-failure-control must contain no node_failure or link_failure events")
        recovery_control = accuracy_curve(args.no_failure_control)
    metrics.update(recovery_metrics(accuracy, recovery_control, topology, args.recovery_gap_pp, args.recovery_patience))
    metrics.update(topology_recovery_metrics(topology, simulated))
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Updated: {metrics_path}")


if __name__ == "__main__":
    main()
