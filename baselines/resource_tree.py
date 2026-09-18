from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import ray
from deasfl.aggregation import finalize_weighted_state
from fl_core.artifacts import failure_state_at_round

from .common import (
    aggregate_tree,
    build_resource_tree,
    connected_components,
    distribute_tree,
    record_path,
    set_all_models,
    shortest_path,
    undirected_neighbors,
)


def _edge_costs(runtime: Any, graph: dict[int, list[int]], model_bytes: int) -> dict[tuple[int, int], float]:
    """Use the slower direction because a tree edge carries up and down traffic."""
    result: dict[tuple[int, int], float] = {}
    for src, values in graph.items():
        for dst in values:
            if (src, dst) in result:
                continue
            costs = []
            for left, right in ((src, dst), (dst, src)):
                profile = runtime._resolve_link_profile("upload")
                profile = runtime._apply_client_network_adjustments(
                    profile, f"client_{left}", f"client_{right}"
                )
                profile = runtime._apply_edge_overrides(
                    profile, f"client_{left}", f"client_{right}"
                )
                costs.append(runtime._estimate_comm_latency_ms(
                    model_bytes, profile["bandwidth_mbps"], profile["latency_ms"]
                ))
            result[(src, dst)] = result[(dst, src)] = max(costs)
    return result


def _record_context(runtime: Any, cfg: Any, round_id: int, root: int,
                    parents: dict[int, int | None], children: dict[int, list[int]],
                    selected: list[int], active: list[int], dropped: list[int], aggregated_nodes: set[int],
                    wall_clock: float) -> None:
    inputs: dict[str, int] = defaultdict(int)
    for event in runtime.monitor.comm_events:
        if (event["round_id"] == round_id and event["delivered"] and
                event.get("phase") == "MODEL_UPLOAD"):
            inputs[event["dst"].removeprefix("client_")] += 1
    sample_counts = {node: len(runtime.client_indices[node]) for node in active}
    runtime.monitor.record_round_context(
        experiment_id=runtime.run_name, method=cfg.algorithm, seed=cfg.seed,
        round=round_id, node_ids=[str(node) for node in range(cfg.num_clients)],
        roles={str(node): ("RESOURCE_ROOT" if node == root else "CLIENT")
               for node in range(cfg.num_clients)},
        parents={str(node): parent for node, parent in parents.items() if parent is not None},
        children_counts={str(node): len(items) for node, items in children.items()},
        active_nodes=[str(node) for node in active],
        failed_nodes=[str(node) for node in dropped],
        eligible_node_ratio=len(active) / max(1, len(active) + len(dropped)),
        reachable_node_ratio=len(aggregated_nodes) / max(1, len(active)),
        eligible_sample_ratio=(sum(sample_counts.values()) / max(
            1, sum(len(runtime.client_indices[node]) for node in selected)
        )),
        aggregated_sample_ratio=(sum(sample_counts[node] for node in aggregated_nodes)
                                 / max(1, sum(sample_counts.values()))),
        round_completion_ratio=1.0,
        aggregation_inputs=dict(inputs),
        aggregation_inputs_by_phase={"MODEL_UPLOAD": dict(inputs)},
        round_wall_clock_seconds=wall_clock,
        round_latency_sim_ms=None,
    )


def _synchronize_rejoining_clients(
    runtime: Any,
    params: dict[str, Any],
    active: list[int],
    previously_active: set[int],
    graph: dict[int, list[int]],
    round_id: int,
) -> None:
    """Give rejoining clients the current global model before local training."""
    rejoining = sorted(set(active) - previously_active)
    if not rejoining:
        return
    synchronized = set(active) & previously_active
    if not synchronized:
        raise RuntimeError("Resource-Tree has no synchronized source for rejoining clients")
    model_bytes = sum(value.numel() * value.element_size() for value in params.values())
    for node in rejoining:
        source = min(synchronized)
        path = shortest_path(graph, source, node)
        delivered = record_path(
            runtime, round_id, path, model_bytes, "broadcast",
            "GLOBAL_MODEL", "MODEL_DISTRIBUTION",
            f"resource-tree:rejoin:{source}:{node}",
        )
        if not delivered:
            raise RuntimeError(
                f"Resource-Tree could not synchronize rejoining client {node}"
            )
        ray.get(runtime.clients[node].receive_model.remote(params))
        synchronized.add(node)


def run(runtime: Any, cfg: Any) -> None:
    params = runtime.init_model_params()
    runtime.save_experiment_inputs(params)
    set_all_models(runtime, params)
    previously_active = set(range(cfg.num_clients))
    start_time = time.perf_counter()

    for round_id in range(cfg.rounds):
        round_start = time.perf_counter()
        runtime.record_frozen_topology_events(round_id)
        selected = runtime.select_clients()
        active, dropped = runtime.inject_dropouts(selected, round_id)
        if not active:
            raise RuntimeError("Resource-Tree has no active clients for this round")
        graph = undirected_neighbors(runtime.neighbors, set(active))
        _, failed_edges = failure_state_at_round(runtime.frozen_inputs, round_id)
        graph = {
            node: [neighbor for neighbor in values
                   if tuple(sorted((node, neighbor))) not in failed_edges]
            for node, values in graph.items()
        }
        _synchronize_rejoining_clients(
            runtime, params, active, previously_active, graph, round_id
        )
        train_results = runtime.train_selected_clients(active, round_id)
        payloads = ray.get([runtime.clients[node].upload_model.remote() for node in active])
        model_states = {item["client_id"]: item["params"] for item in payloads}
        sample_counts = {node: len(runtime.client_indices[node]) for node in active}
        first_params = payloads[0]["params"]
        model_bytes = sum(value.numel() * value.element_size()
                          for value in first_params.values())
        component = connected_components(graph)[0]
        reachable_graph = {node: [neighbor for neighbor in graph[node] if neighbor in component]
                           for node in sorted(component)}
        root, parents, children = build_resource_tree(
            reachable_graph, runtime.node_resources_input,
            _edge_costs(runtime, reachable_graph, model_bytes)
        )
        print(f"Resource-Tree round {round_id + 1}: root client {root}")
        weighted_params, aggregated_nodes = aggregate_tree(
            root, children, model_states, sample_counts, runtime, round_id,
            model_bytes, phase="MODEL_UPLOAD", flow_prefix="resource-tree",
        )
        params = finalize_weighted_state(weighted_params)
        # Keep the root synchronized for the next training round.
        ray.get(runtime.clients[root].receive_model.remote(params))
        distribute_tree(
            root, children, params, runtime, round_id, model_bytes,
            phase="MODEL_DISTRIBUTION", flow_prefix="resource-tree",
        )
        val_loss, val_acc = runtime.evaluate_global(params)
        summary = runtime.summarize_round_comm(round_id)
        elapsed = time.perf_counter() - round_start
        _record_context(runtime, cfg, round_id, root, parents, children, selected, active,
                        dropped, aggregated_nodes, elapsed)
        runtime.monitor.record_round(
            round_id=round_id,
            avg_train_loss=sum(item["train_loss"] for item in train_results) / len(train_results),
            val_loss=val_loss, val_acc=val_acc, selected_count=len(selected),
            dropped_count=len(dropped), comm_success_count=summary["comm_success_count"],
            comm_failure_count=summary["comm_failure_count"],
            comm_timeout_count=summary["comm_timeout_count"],
            broadcast_success_count=summary["broadcast_success_count"],
            broadcast_failure_count=summary["broadcast_failure_count"],
            upload_success_count=summary["upload_success_count"],
            upload_failure_count=summary["upload_failure_count"],
            upload_storage_blocked_count=summary["upload_storage_blocked_count"],
            gossip_success_count=0, gossip_failure_count=0,
            round_wall_clock_seconds=elapsed,
        )
        print(f"Round {round_id + 1}/{cfg.rounds}: acc={val_acc:.4f}, "
              f"messages={summary['comm_success_count']}")
        previously_active = set(active)

    runtime.save_final_model(params)
    print(f"Resource-Tree completed in {time.perf_counter() - start_time:.2f}s")
