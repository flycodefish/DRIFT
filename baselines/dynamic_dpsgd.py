"""Dynamic synchronous D-PSGD with persistent node/link events."""

from __future__ import annotations

import json
import time
from typing import Any

import ray

from .common import connected_components, record_path, set_all_models, undirected_neighbors
from .dpsgd import metropolis_weights


def _trace(runtime: Any, cfg: Any) -> list[dict[str, Any]]:
    if runtime.frozen_inputs:
        raw = runtime.frozen_inputs.get("failure_trace.json", {})
    elif cfg.dynamic_failure_trace_json:
        raw = json.loads(cfg.dynamic_failure_trace_json)
    else:
        raw = []
    if isinstance(raw, list):
        events = raw
    elif isinstance(raw, dict):
        events = raw.get("events", [])
    else:
        raise ValueError("dynamic failure trace must be a list or an object with events")
    if not isinstance(events, list):
        raise ValueError("dynamic failure trace must be a list or an object with events")
    return events


def _raw_trace(runtime: Any, cfg: Any) -> Any:
    if runtime.frozen_inputs:
        return runtime.frozen_inputs.get("failure_trace.json", {})
    if cfg.dynamic_failure_trace_json:
        return json.loads(cfg.dynamic_failure_trace_json)
    return []


def _targets(event: dict[str, Any]) -> list[int]:
    if not any(key in event for key in ("node_ids", "targets", "target")):
        raise ValueError(f"node event has no target: {event!r}")
    raw = event.get("node_ids", event.get("targets", event.get("target")))
    if not isinstance(raw, list):
        raw = [raw]
    return [int(value) for value in raw]


def _edges(event: dict[str, Any]) -> list[tuple[int, int]]:
    if not any(key in event for key in ("edges", "targets", "target")):
        raise ValueError(f"link event has no target: {event!r}")
    raw = event.get("edges", event.get("targets", event.get("target")))
    if isinstance(raw, dict):
        raw = list(raw.items())
    if not isinstance(raw, list):
        raw = [raw]
    if raw and isinstance(raw[0], int):
        raw = [raw]
    result = []
    for edge in raw:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            raise ValueError(f"invalid link edge: {edge!r}")
        left, right = int(edge[0]), int(edge[1])
        if left == right:
            raise ValueError("link failure cannot target a self-loop")
        result.append(tuple(sorted((left, right))))
    return result


def _validate_trace(events: list[dict[str, Any]], num_clients: int,
                    base_graph: dict[int, list[int]], rounds: int,
                    initial_inactive: set[int]) -> None:
    valid_types = {"node_failure", "node_recovery", "node_join",
                   "link_failure", "link_recovery"}
    physical_edges = {
        tuple(sorted((src, dst)))
        for src, values in base_graph.items() for dst in values
        if src != dst
    }
    available = set(base_graph) - initial_inactive
    disabled_edges: set[tuple[int, int]] = set()
    for event in events:
        if not isinstance(event, dict):
            raise ValueError(f"dynamic event must be an object: {event!r}")
    try:
        ordered_events = sorted(events, key=lambda item: int(item.get("round", -1)))
    except (TypeError, ValueError) as exc:
        raise ValueError("dynamic event has an invalid round") from exc
    for event in ordered_events:
        try:
            round_id = int(event["round"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"dynamic event has invalid round: {event!r}") from exc
        if not 0 <= round_id < rounds:
            raise ValueError(f"dynamic event round outside run: {round_id}")
        event_type = str(event.get("event_type", "")).lower()
        if event_type not in valid_types:
            raise ValueError(f"unsupported dynamic event type: {event_type!r}")
        if event_type.startswith("node_"):
            ids = _targets(event)
            if any(node < 0 or node >= num_clients for node in ids):
                raise ValueError(f"invalid node in dynamic event: {ids}")
            for node in ids:
                if event_type == "node_failure":
                    if node not in available:
                        raise ValueError(f"node {node} is already inactive")
                    available.remove(node)
                elif node in available:
                    raise ValueError(f"node {node} is already active")
                else:
                    available.add(node)
        else:
            edges = _edges(event)
            if any(edge not in physical_edges for edge in edges):
                raise ValueError(f"dynamic event targets a non-physical edge: {edges}")
            for edge in edges:
                if event_type == "link_failure":
                    if edge in disabled_edges:
                        raise ValueError(f"edge {edge} is already failed")
                    disabled_edges.add(edge)
                else:
                    if edge not in disabled_edges:
                        raise ValueError(f"edge {edge} is not failed")
                    disabled_edges.remove(edge)


def _apply_events(
    events: list[dict[str, Any]], round_id: int, num_clients: int,
    available: set[int], disabled_edges: set[tuple[int, int]], monitor: Any,
) -> None:
    for event in events:
        if int(event.get("round", -1)) != round_id:
            continue
        event_type = str(event.get("event_type", "")).lower()
        if event_type in {"node_failure", "node_recovery", "node_join"}:
            ids = _targets(event)
            if any(node < 0 or node >= num_clients for node in ids):
                raise ValueError(f"invalid node in dynamic event: {ids}")
            failed = event_type == "node_failure"
            for node in ids:
                old = node in available
                if failed:
                    available.discard(node)
                else:
                    available.add(node)
                monitor.record_topology_event(
                    experiment_id=getattr(monitor, "experiment_id", ""),
                    method=getattr(monitor, "method", ""),
                    seed=getattr(monitor, "seed", 0),
                    round=round_id, event_type=event_type,
                    target_type="node", target_id=str(node), affected_nodes=[node],
                    old_state="active" if old else "inactive",
                    new_state="active" if node in available else "inactive",
                    control_messages=0, control_bytes=0,
                    repair_time_wall_clock_seconds=0.0, source="failure_trace",
                )
        elif event_type in {"link_failure", "link_recovery"}:
            edges = _edges(event)
            if any(node < 0 or node >= num_clients for edge in edges for node in edge):
                raise ValueError(f"invalid edge in dynamic event: {edges}")
            failed = event_type == "link_failure"
            for edge in edges:
                old = edge in disabled_edges
                if failed:
                    disabled_edges.add(edge)
                else:
                    disabled_edges.discard(edge)
                monitor.record_topology_event(
                    experiment_id=getattr(monitor, "experiment_id", ""),
                    method=getattr(monitor, "method", ""),
                    seed=getattr(monitor, "seed", 0),
                    round=round_id, event_type=event_type,
                    target_type="link", target_id=f"{edge[0]}-{edge[1]}",
                    affected_nodes=list(edge),
                    old_state="failed" if old else "active",
                    new_state="failed" if edge in disabled_edges else "active",
                    control_messages=0, control_bytes=0,
                    repair_time_wall_clock_seconds=0.0, source="failure_trace",
                )
        else:
            raise ValueError(f"unsupported dynamic event type: {event_type!r}")


def _record_context(runtime: Any, cfg: Any, round_id: int, selected: list[int],
                    active: list[int], unavailable: list[int], graph: dict[int, list[int]],
                    wall_clock: float) -> None:
    components = connected_components(graph)
    runtime.monitor.record_round_context(
        experiment_id=runtime.run_name, method=cfg.algorithm, seed=cfg.seed,
        round=round_id, node_ids=[str(node) for node in range(cfg.num_clients)],
        roles={str(node): "PEER" for node in range(cfg.num_clients)},
        parents={}, children_counts={}, active_nodes=[str(node) for node in active],
        failed_nodes=[str(node) for node in unavailable],
        eligible_node_ratio=len(active) / max(1, len(selected)),
        reachable_node_ratio=(max((len(part) for part in components), default=0) /
                              max(1, len(active))),
        eligible_sample_ratio=(sum(len(runtime.client_indices[node]) for node in active) /
                               max(1, sum(len(runtime.client_indices[node]) for node in selected))),
        aggregated_sample_ratio=None,
        round_completion_ratio=1.0 if active else 0.0,
        aggregation_inputs={}, aggregation_inputs_by_phase={},
        effective_edge_count=sum(len(values) for values in graph.values()) // 2,
        round_wall_clock_seconds=wall_clock, round_latency_sim_ms=None,
    )


def run(runtime: Any, cfg: Any) -> None:
    params = runtime.init_model_params()
    runtime.save_experiment_inputs(params)
    set_all_models(runtime, params)
    events = _trace(runtime, cfg)
    raw_trace = _raw_trace(runtime, cfg)
    base_graph = undirected_neighbors(runtime.neighbors)
    trace_initial = (raw_trace.get("initial_inactive_nodes", [])
                     if isinstance(raw_trace, dict) else [])
    if cfg.dynamic_initial_inactive_nodes:
        initial_inactive = json.loads(cfg.dynamic_initial_inactive_nodes)
        if not isinstance(initial_inactive, list):
            raise ValueError("dynamic_initial_inactive_nodes must be a JSON list")
        initial_inactive = {int(node) for node in initial_inactive}
        if any(node < 0 or node >= cfg.num_clients for node in initial_inactive):
            raise ValueError("dynamic_initial_inactive_nodes contains invalid node")
    else:
        initial_inactive = {int(node) for node in trace_initial}
    if any(node < 0 or node >= cfg.num_clients for node in initial_inactive):
        raise ValueError("initial_inactive_nodes contains invalid node")
    _validate_trace(events, cfg.num_clients, base_graph, cfg.rounds, initial_inactive)
    available = set(base_graph) - initial_inactive
    disabled_edges: set[tuple[int, int]] = set()
    start_time = time.perf_counter()
    print(f"Running Dynamic D-PSGD on {cfg.topology} topology")

    for round_id in range(cfg.rounds):
        round_start = time.perf_counter()
        _apply_events(events, round_id, cfg.num_clients, available, disabled_edges,
                      runtime.monitor)
        selected = runtime.select_clients()
        active = [node for node in selected if node in available]
        selected_unavailable = [node for node in selected if node not in available]
        unavailable = [node for node in range(cfg.num_clients) if node not in available]
        if not active:
            raise RuntimeError("Dynamic D-PSGD has no active selected clients")
        graph = undirected_neighbors(base_graph, set(active))
        graph = {node: [neighbor for neighbor in values
                        if tuple(sorted((node, neighbor))) not in disabled_edges]
                 for node, values in graph.items()}
        train_results = runtime.train_selected_clients(active, round_id)
        snapshots = ray.get([runtime.clients[node].upload_model.remote() for node in active])
        snapshot_map = {item["client_id"]: item["params"] for item in snapshots}
        payload_bytes = sum(t.numel() * t.element_size()
                            for t in snapshots[0]["params"].values())
        refs = []
        successful_edges = 0
        for receiver in active:
            weights = metropolis_weights(graph, receiver)
            sender_params, sender_ids, sender_weights = [], [], [weights[receiver]]
            for sender in sorted(weights):
                if sender == receiver:
                    continue
                if record_path(runtime, round_id, [sender, receiver], payload_bytes, "gossip",
                               "PEER_MODEL", "PEER_EXCHANGE",
                               f"dynamic-dpsgd:{round_id}:{sender}:{receiver}"):
                    sender_params.append(snapshot_map[sender])
                    sender_ids.append(sender)
                    sender_weights.append(weights[sender])
                    successful_edges += 1
            refs.append(runtime.clients[receiver].receive_weighted_push.remote(
                sender_params, sender_ids, sender_weights))
        ray.get(refs)
        interval = max(1, int(cfg.decentralized_diagnostic_interval))
        params, metrics = runtime.evaluate_decentralized(
            active, include_local=(round_id % interval == 0 or round_id == cfg.rounds - 1)
        )
        summary = runtime.summarize_round_comm(round_id)
        elapsed = time.perf_counter() - round_start
        _record_context(runtime, cfg, round_id, selected, active, unavailable, graph, elapsed)
        runtime.monitor.record_round(
            round_id=round_id,
            avg_train_loss=sum(item["train_loss"] for item in train_results) / len(train_results),
            val_loss=metrics["consensus_loss"], val_acc=metrics["consensus_accuracy"],
            selected_count=len(selected), dropped_count=len(selected_unavailable),
            comm_success_count=summary["comm_success_count"],
            comm_failure_count=summary["comm_failure_count"],
            comm_timeout_count=summary["comm_timeout_count"],
            broadcast_success_count=summary["broadcast_success_count"],
            broadcast_failure_count=summary["broadcast_failure_count"],
            upload_success_count=summary["upload_success_count"],
            upload_failure_count=summary["upload_failure_count"],
            upload_storage_blocked_count=summary["upload_storage_blocked_count"],
            gossip_success_count=successful_edges,
            gossip_failure_count=summary["gossip_failure_count"],
            round_wall_clock_seconds=elapsed,
        )
        runtime.monitor.record_decentralized_metrics(round_id, **metrics)
        print(f"Round {round_id + 1}/{cfg.rounds}: acc={metrics['consensus_accuracy']:.4f}, "
              f"active={len(active)}, edges={sum(map(len, graph.values())) // 2}, "
              f"messages={summary['comm_success_count']}")
    runtime.save_final_model(params)
    print(f"Dynamic D-PSGD completed in {time.perf_counter() - start_time:.2f}s")
