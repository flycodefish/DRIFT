"""Synchronous EL-Local adapted to a routed physical network.

The original Epidemic Learning protocol assumes that a sampled peer can be
reached directly.  This baseline keeps the randomized peer sampling and
plain averaging semantics, but routes every overlay transfer over the frozen
physical graph so that communication cost is observable in this project.
"""

from __future__ import annotations

import math
import random
import time
from typing import Any

import ray

from .common import connected_components, record_path, set_all_models, shortest_path, undirected_neighbors
from .dynamic_dpsgd import _apply_events, _raw_trace, _trace, _validate_trace


def default_epidemic_sample_size(num_clients: int) -> int:
    """Return the paper-inspired ``ceil(log2(n))`` peer count."""
    if num_clients <= 1:
        return 0
    return max(1, math.ceil(math.log2(num_clients)))


def sample_epidemic_targets(
    sender: int,
    candidates: list[int] | tuple[int, ...] | set[int],
    rng: random.Random,
    sample_size: int,
) -> list[int]:
    """Sample distinct peers deterministically without ever selecting self."""
    pool = sorted({int(node) for node in candidates if int(node) != int(sender)})
    count = min(max(0, int(sample_size)), len(pool))
    return rng.sample(pool, count) if count else []


def resolve_epidemic_sample_size(num_clients: int, configured_size: int) -> int:
    if configured_size < 0:
        raise ValueError("epidemic_sample_size must be non-negative")
    if num_clients <= 1:
        if configured_size > 0:
            raise ValueError("epidemic_sample_size must be zero for a single client")
        return 0
    if configured_size > num_clients - 1:
        raise ValueError("epidemic_sample_size cannot exceed num_clients - 1")
    return configured_size or default_epidemic_sample_size(num_clients)


def _record_unreachable(
    runtime: Any, round_id: int, sender: int, target: int, model_bytes: int
) -> None:
    """Record a sampled overlay transfer for which no physical route exists."""
    runtime.monitor.record_comm(
        round_id=round_id,
        src=f"client_{sender}",
        dst=f"client_{target}",
        payload_bytes=model_bytes,
        kind="gossip",
        simulated_latency_ms=0.0,
        bandwidth_mbps=float(runtime.cfg.link_bandwidth_mbps),
        drop_rate=1.0,
        delivered=False,
        retries_used=0,
        timeout_occurred=False,
        total_backoff_ms=0.0,
        link_latency_ms=None,
        message_type="PEER_MODEL",
        phase="PEER_EXCHANGE",
        flow_id=f"epidemic:{round_id}:{sender}:{target}",
        link_type="UNREACHABLE",
        physical_hop_index=1,
        physical_hop_count=1,
    )


def _record_context(
    runtime: Any,
    cfg: Any,
    round_id: int,
    selected: list[int],
    active: list[int],
    dropped: list[int],
    graph: dict[int, list[int]],
    aggregation_inputs: dict[int, int],
    logical_successes: int,
    logical_failures: int,
    wall_clock: float,
) -> None:
    components = connected_components(graph)
    largest_component = max((len(component) for component in components), default=0)
    sample_denominator = sum(len(runtime.client_indices[node]) for node in selected)
    runtime.monitor.record_round_context(
        experiment_id=runtime.run_name,
        method=cfg.algorithm,
        seed=cfg.seed,
        round=round_id,
        node_ids=[str(node) for node in range(cfg.num_clients)],
        roles={str(node): "PEER" for node in range(cfg.num_clients)},
        parents={},
        children_counts={},
        active_nodes=[str(node) for node in active],
        failed_nodes=[str(node) for node in dropped],
        eligible_node_ratio=len(active) / max(1, len(selected)),
        reachable_node_ratio=largest_component / max(1, len(active)),
        eligible_sample_ratio=(
            sum(len(runtime.client_indices[node]) for node in active)
            / max(1, sample_denominator)
        ),
        aggregated_sample_ratio=None,
        round_completion_ratio=1.0,
        aggregation_inputs={str(node): int(count) for node, count in aggregation_inputs.items()},
        aggregation_inputs_by_phase={
            "PEER_EXCHANGE": {
                str(node): int(count) for node, count in aggregation_inputs.items()
            }
        },
        effective_edge_count=sum(len(values) for values in graph.values()) // 2,
        logical_peer_success_count=logical_successes,
        logical_peer_failure_count=logical_failures,
        round_wall_clock_seconds=wall_clock,
        round_latency_sim_ms=None,
    )


def run(runtime: Any, cfg: Any) -> None:
    """Run EL-Local-Routed with synchronous local training and plain averaging."""
    params = runtime.init_model_params()
    runtime.save_experiment_inputs(params)
    set_all_models(runtime, params)
    if abs(float(cfg.client_sample_ratio) - 1.0) > 1e-12:
        raise ValueError("EL-Local-Routed requires client_sample_ratio=1.0")
    rng_by_node = {
        node: random.Random(int(cfg.seed) + node)
        for node in range(cfg.num_clients)
    }
    configured_size = int(getattr(cfg, "epidemic_sample_size", 0))
    sample_size = resolve_epidemic_sample_size(cfg.num_clients, configured_size)
    events = _trace(runtime, cfg)
    raw_trace = _raw_trace(runtime, cfg)
    base_graph = undirected_neighbors(runtime.neighbors)
    trace_initial = (
        raw_trace.get("initial_inactive_nodes", [])
        if isinstance(raw_trace, dict)
        else []
    )
    if cfg.dynamic_initial_inactive_nodes:
        import json

        configured_initial = json.loads(cfg.dynamic_initial_inactive_nodes)
        if not isinstance(configured_initial, list):
            raise ValueError("dynamic_initial_inactive_nodes must be a JSON list")
        initial_inactive = {int(node) for node in configured_initial}
    else:
        initial_inactive = {int(node) for node in trace_initial}
    if any(node < 0 or node >= cfg.num_clients for node in initial_inactive):
        raise ValueError("initial_inactive_nodes contains invalid node")
    _validate_trace(
        events, cfg.num_clients, base_graph, cfg.rounds, initial_inactive
    )
    available = set(base_graph) - initial_inactive
    disabled_edges: set[tuple[int, int]] = set()
    print(
        "Running EL-Local-Routed on %s topology "
        "(sample_size=%d; physical-topology constrained)"
        % (cfg.topology, sample_size)
    )
    start_time = time.perf_counter()

    for round_id in range(cfg.rounds):
        round_start = time.perf_counter()
        _apply_events(
            events,
            round_id,
            cfg.num_clients,
            available,
            disabled_edges,
            runtime.monitor,
        )
        selected = runtime.select_clients()
        selected_available = [node for node in selected if node in available]
        active, stochastic_dropped = runtime.inject_dropouts(
            selected_available, round_id
        )
        selected_dropped = sorted(
            (set(selected) - set(selected_available)) | set(stochastic_dropped)
        )
        unavailable = sorted((set(base_graph) - available) | set(stochastic_dropped))
        if not active:
            raise RuntimeError("EL-Local-Routed has no active clients for this round")
        graph = undirected_neighbors(base_graph, set(active))
        graph = {
            node: [
                neighbor
                for neighbor in values
                if tuple(sorted((node, neighbor))) not in disabled_edges
            ]
            for node, values in graph.items()
        }
        train_results = runtime.train_selected_clients(active, round_id)
        snapshots = ray.get(
            [runtime.clients[node].upload_model.remote() for node in active]
        )
        snapshot_map = {item["client_id"]: item["params"] for item in snapshots}
        model_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in snapshots[0]["params"].values()
        )

        # Freeze all local models before any peer mixing, matching the
        # synchronous EL round semantics.
        incoming_params: dict[int, list[dict[str, Any]]] = {
            node: [] for node in active
        }
        incoming_ids: dict[int, list[int]] = {node: [] for node in active}
        logical_successes = 0
        logical_failures = 0
        for sender in active:
            targets = sample_epidemic_targets(
                sender,
                set(active) - {sender},
                rng_by_node[sender],
                sample_size,
            )
            for target in targets:
                try:
                    path = shortest_path(graph, sender, target)
                except ValueError:
                    _record_unreachable(
                        runtime, round_id, sender, target, model_bytes
                    )
                    logical_failures += 1
                    continue
                delivered = record_path(
                    runtime,
                    round_id,
                    path,
                    model_bytes,
                    "gossip",
                    "PEER_MODEL",
                    "PEER_EXCHANGE",
                    f"epidemic:{round_id}:{sender}:{target}",
                    link_type="OVERLAY",
                )
                if delivered:
                    logical_successes += 1
                    incoming_params[target].append(snapshot_map[sender])
                    incoming_ids[target].append(sender)
                else:
                    logical_failures += 1

        # PlainAverageSharing: each receiver averages its current model and
        # every successfully received peer model with equal weights.
        refs = []
        for receiver in active:
            received = incoming_params[receiver]
            refs.append(
                runtime.clients[receiver].receive_weighted_push.remote(
                    received,
                    incoming_ids[receiver],
                    [1.0] * (len(received) + 1),
                )
            )
        ray.get(refs)

        diagnostic_interval = max(1, int(cfg.decentralized_diagnostic_interval))
        include_local = (
            round_id % diagnostic_interval == 0 or round_id == cfg.rounds - 1
        )
        params, metrics = runtime.evaluate_decentralized(
            active, include_local=include_local
        )
        summary = runtime.summarize_round_comm(round_id)
        elapsed = time.perf_counter() - round_start
        _record_context(
            runtime,
            cfg,
            round_id,
            selected,
            active,
            unavailable,
            graph,
            {node: len(incoming_params[node]) for node in active},
            logical_successes,
            logical_failures,
            elapsed,
        )
        runtime.monitor.record_round(
            round_id=round_id,
            avg_train_loss=sum(item["train_loss"] for item in train_results)
            / len(train_results),
            val_loss=metrics["consensus_loss"],
            val_acc=metrics["consensus_accuracy"],
            selected_count=len(selected),
            dropped_count=len(selected_dropped),
            comm_success_count=summary["comm_success_count"],
            comm_failure_count=summary["comm_failure_count"],
            comm_timeout_count=summary["comm_timeout_count"],
            broadcast_success_count=summary["broadcast_success_count"],
            broadcast_failure_count=summary["broadcast_failure_count"],
            upload_success_count=summary["upload_success_count"],
            upload_failure_count=summary["upload_failure_count"],
            upload_storage_blocked_count=summary["upload_storage_blocked_count"],
            gossip_success_count=logical_successes,
            gossip_failure_count=logical_failures,
            round_wall_clock_seconds=elapsed,
        )
        runtime.monitor.record_decentralized_metrics(round_id, **metrics)
        print(
            f"Round {round_id + 1}/{cfg.rounds}: "
            f"acc={metrics['consensus_accuracy']:.4f}, "
            f"logical_messages={logical_successes}, "
            f"physical_hops={summary['comm_success_count']}"
        )

    runtime.save_final_model(params)
    print(f"EL-Local-Routed completed in {time.perf_counter() - start_time:.2f}s")
