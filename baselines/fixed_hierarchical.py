from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import ray

from deasfl.aggregation import finalize_weighted_state, merge_weighted_states

from .common import (
    aggregate_tree,
    build_fixed_hierarchy,
    distribute_tree,
    prune_fixed_tree,
    record_path,
    set_all_models,
    shortest_path,
    undirected_neighbors,
)


def _record_context(runtime: Any, cfg: Any, round_id: int, root: int,
                    heads: list[int], owner: dict[int, int],
                    parents: dict[int, int | None], children: dict[int, list[int]],
                    selected: list[int], active: list[int], dropped: list[int],
                    aggregated_nodes: set[int], wall_clock: float,
                    round_completed: bool) -> None:
    by_phase: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for event in runtime.monitor.comm_events:
        if event["round_id"] == round_id and event["delivered"]:
            phase = event.get("phase")
            if phase in {"INTRA_CLUSTER_UPLOAD", "INTER_CLUSTER_UPLOAD"}:
                if (phase == "INTER_CLUSTER_UPLOAD" and
                        event.get("physical_hop_index", 1) !=
                        event.get("physical_hop_count", 1)):
                    continue
                by_phase[phase][event["dst"].removeprefix("client_")] += 1
    total_inputs: dict[str, int] = defaultdict(int)
    for values in by_phase.values():
        for node, value in values.items():
            total_inputs[node] += value
    roles = {}
    for node in owner:
        if node == root:
            roles[str(node)] = "GLOBAL_AGGREGATOR_AND_CLUSTER_HEAD"
        elif node in heads:
            roles[str(node)] = "CLUSTER_HEAD"
        else:
            roles[str(node)] = "CLIENT"
    sample_counts = {node: len(runtime.client_indices[node]) for node in owner}
    active_set = set(active)
    runtime.monitor.record_round_context(
        experiment_id=runtime.run_name, method=cfg.algorithm, seed=cfg.seed,
        round=round_id, node_ids=[str(node) for node in sorted(owner)], roles=roles,
        parents={str(node): parent for node, parent in parents.items() if parent is not None},
        children_counts={str(node): len(values) for node, values in children.items()},
        cluster_heads=[str(head) for head in heads],
        cluster_members={str(head): [str(node) for node, assigned in owner.items() if assigned == head]
                         for head in heads},
        active_nodes=[str(node) for node in sorted(active)],
        failed_nodes=[str(node) for node in sorted(dropped)],
        eligible_node_ratio=len(active) / max(1, len(selected)),
        reachable_node_ratio=len(aggregated_nodes) / max(1, len(active)),
        eligible_sample_ratio=(sum(sample_counts[node] for node in active_set) /
                               max(1, sum(sample_counts[node] for node in selected))),
        aggregated_sample_ratio=(sum(sample_counts[node] for node in aggregated_nodes)
                                 / max(1, sum(sample_counts[node] for node in active_set))),
        round_completion_ratio=1.0 if round_completed else 0.0,
        aggregation_inputs=dict(total_inputs),
        aggregation_inputs_by_phase={phase: dict(values) for phase, values in by_phase.items()},
        round_wall_clock_seconds=wall_clock,
        round_latency_sim_ms=None,
    )


def run(runtime: Any, cfg: Any) -> None:
    """Run standard fixed two-level hierarchical federated learning."""
    params = runtime.init_model_params()
    runtime.save_experiment_inputs(params)
    set_all_models(runtime, params)
    graph = undirected_neighbors(runtime.neighbors)
    root = int(cfg.fixed_hierarchical_root_id)
    heads, owner, parents, children = build_fixed_hierarchy(
        graph, root, int(cfg.fixed_hierarchical_num_clusters)
    )
    print(f"Running Fixed Hierarchical FL: root={root}, cluster_heads={heads}")
    start_time = time.perf_counter()
    sample_counts = {node: len(runtime.client_indices[node]) for node in owner}
    inter_paths = {head: shortest_path(graph, head, root)
                   for head in heads if head != root}

    for round_id in range(cfg.rounds):
        round_start = time.perf_counter()
        runtime.record_frozen_topology_events(round_id)
        selected = runtime.select_clients()
        active, dropped = runtime.inject_dropouts(selected, round_id)
        if not active:
            raise RuntimeError("Fixed Hierarchical FL has no active clients for this round")
        train_results = runtime.train_selected_clients(active, round_id)
        payloads = ray.get([runtime.clients[node].upload_model.remote() for node in active])
        model_states = {item["client_id"]: item["params"] for item in payloads}
        model_bytes = sum(value.numel() * value.element_size()
                          for value in payloads[0]["params"].values())

        active_set = set(active)
        cluster_states = {}
        cluster_nodes: dict[int, set[int]] = {}
        for head in heads:
            pruned_children, reachable = prune_fixed_tree(children, active_set, head)
            if not reachable:
                continue
            state, included = aggregate_tree(
                head, pruned_children, model_states, sample_counts, runtime, round_id,
                model_bytes, phase="INTRA_CLUSTER_UPLOAD",
                flow_prefix=f"fixed:cluster:{head}",
            )
            cluster_states[head] = state
            cluster_nodes[head] = included

        aggregated_nodes: set[int] = set()
        round_completed = root in cluster_states
        if round_completed:
            global_parts = [cluster_states[root]]
            aggregated_nodes.update(cluster_nodes[root])
            for head in heads:
                if head == root or head not in cluster_states:
                    continue
                path = inter_paths[head]
                if (all(node in active_set for node in path) and record_path(
                    runtime, round_id, path, model_bytes, "upload", "AGGREGATED_MODEL",
                    "INTER_CLUSTER_UPLOAD", f"fixed:cluster:{head}:inter-up",
                    link_type="INTER_CLUSTER",
                )):
                    global_parts.append(cluster_states[head])
                    aggregated_nodes.update(cluster_nodes[head])
            params = finalize_weighted_state(merge_weighted_states(global_parts))
            ray.get(runtime.clients[root].receive_model.remote(params))

            # No re-election or rerouting: a failed fixed node cuts that path.
            root_children, _ = prune_fixed_tree(children, active_set, root)
            distribute_tree(root, root_children, params, runtime, round_id, model_bytes,
                            phase="INTRA_CLUSTER_DOWNLOAD", flow_prefix=f"fixed:cluster:{root}")
            for head in heads:
                if head == root or head not in cluster_states:
                    continue
                path = list(reversed(inter_paths[head]))
                if (all(node in active_set for node in path) and record_path(
                    runtime, round_id, path, model_bytes, "broadcast", "GLOBAL_MODEL",
                    "INTER_CLUSTER_DOWNLOAD", f"fixed:cluster:{head}:inter-down",
                    link_type="INTER_CLUSTER",
                )):
                    ray.get(runtime.clients[head].receive_model.remote(params))
                    head_children, _ = prune_fixed_tree(children, active_set, head)
                    distribute_tree(head, head_children, params, runtime, round_id, model_bytes,
                                    phase="INTRA_CLUSTER_DOWNLOAD",
                                    flow_prefix=f"fixed:cluster:{head}")

        val_loss, val_acc = runtime.evaluate_global(params)
        summary = runtime.summarize_round_comm(round_id)
        elapsed = time.perf_counter() - round_start
        _record_context(runtime, cfg, round_id, root, heads, owner, parents,
                        children, selected, active, dropped, aggregated_nodes,
                        elapsed, round_completed)
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

    runtime.save_final_model(params)
    print(f"Fixed Hierarchical FL completed in {time.perf_counter() - start_time:.2f}s")
