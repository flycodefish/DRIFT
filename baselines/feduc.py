"""Independent FedUC-CenSyn-I5 baseline integration.

The runtime is deliberately opt-in.  Until the paper's complete CenSyn bound
estimator is recovered, only the explicitly named label-distribution proxy is
available and the run identifies itself as FedUC-inspired.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import time
from typing import Any

import ray

from deasfl.aggregation import (
    finalize_weighted_state,
    leaf_weighted_state,
    merge_weighted_states,
)

from .common import record_path, set_all_models, undirected_neighbors
from .feduc_objective import (
    build_reference_assignment,
    frozen_deadline_ms,
    greedy_assignment,
    label_distribution_proxy,
    local_training_times_ms,
    select_aggregators,
    shortest_latency_path,
)


def _label_vectors(runtime: Any, num_classes: int) -> dict[int, list[int]]:
    return {
        node: [int(runtime.client_label_stats[node].get(label, 0))
               for label in range(num_classes)]
        for node in range(runtime.cfg.num_clients)
    }


def _parse_aggregators(cfg: Any, resources: list[dict]) -> list[int]:
    raw = str(getattr(cfg, "feduc_aggregator_ids", "")).strip()
    if raw:
        values = [int(value.strip()) for value in raw.split(",") if value.strip()]
        if len(values) != len(set(values)):
            raise ValueError("feduc_aggregator_ids must be unique")
        return sorted(values)
    count = int(getattr(cfg, "feduc_num_aggregators", 0))
    if count <= 0:
        raise ValueError("FedUC requires feduc_num_aggregators or feduc_aggregator_ids")
    return sorted(select_aggregators(resources, count))


def _edge_times(runtime: Any, graph: dict[int, list[int]], model_bytes: int):
    costs = {}
    for source, neighbors in graph.items():
        for destination in neighbors:
            profile = runtime._resolve_link_profile("upload")
            profile = runtime._apply_client_network_adjustments(
                profile, f"client_{source}", f"client_{destination}"
            )
            profile = runtime._apply_edge_overrides(
                profile, f"client_{source}", f"client_{destination}"
            )
            costs[(source, destination)] = runtime._estimate_comm_latency_ms(
                model_bytes, profile["bandwidth_mbps"], profile["latency_ms"]
            )
    return costs


def _prepare_assignment(runtime: Any, cfg: Any, model_bytes: int):
    mode = str(getattr(cfg, "feduc_objective_mode", "unavailable"))
    if mode == "unavailable":
        raise RuntimeError(
            "The faithful FedUC CenSyn objective is not implemented. "
            "Use label_distribution_proxy only for an explicitly FedUC-inspired run."
        )
    if mode != "label_distribution_proxy":
        raise ValueError(f"Unsupported FedUC objective mode: {mode}")
    graph = undirected_neighbors(runtime.neighbors)
    workers = list(range(cfg.num_clients))
    aggregators = _parse_aggregators(cfg, runtime.node_resources_input)
    if not set(aggregators) <= set(workers):
        raise ValueError("FedUC aggregator ids must identify client nodes")
    edge_times = _edge_times(runtime, graph, model_bytes)
    paths, upload_ms = {}, {}
    for worker in workers:
        for aggregator in aggregators:
            path, latency = shortest_latency_path(
                graph, worker, aggregator, edge_times
            )
            paths[(worker, aggregator)] = path
            upload_ms[(worker, aggregator)] = latency
    sample_counts = {node: len(runtime.client_indices[node]) for node in workers}
    train_ms = local_training_times_ms(
        sample_counts, runtime.node_resources_input,
        local_epochs=cfg.local_epochs,
        reference_ms_per_sample_epoch=cfg.feduc_reference_train_ms_per_sample_epoch,
    )
    reference = build_reference_assignment(workers, aggregators, upload_ms)
    deadline = frozen_deadline_ms(
        reference, train_ms, upload_ms, slack=cfg.feduc_deadline_slack
    )
    labels = _label_vectors(runtime, cfg.num_classes)
    objective = lambda groups: label_distribution_proxy(groups, labels)
    groups, decisions = greedy_assignment(
        workers, aggregators, sample_counts, train_ms, upload_ms,
        deadline, objective,
    )
    audit = {
        "method": "FedUC-inspired-CenSyn-I5",
        "objective_mode": mode,
        "aggregator_ids": aggregators,
        "num_aggregators": len(aggregators),
        "global_sync_interval": int(cfg.feduc_global_sync_interval),
        "deadline_slack": float(cfg.feduc_deadline_slack),
        "deadline_ms": deadline,
        "reference_assignment": reference,
        "final_assignment": groups,
        "sample_counts": sample_counts,
        "label_counts": labels,
        "local_training_ms": train_ms,
        "upload_ms": {f"{worker}->{head}": value
                      for (worker, head), value in upload_ms.items()},
        "paths": {f"{worker}->{head}": path
                  for (worker, head), path in paths.items()},
        "decisions": [decision.__dict__ for decision in decisions],
    }
    return groups, paths, sample_counts, audit


def _record_context(runtime, cfg, round_id, groups, active, dropped,
                    aggregated_nodes, elapsed, global_sync):
    heads = sorted(groups)
    roles = {
        str(node): ("AGGREGATOR" if node in heads else "WORKER")
        for node in range(cfg.num_clients)
    }
    by_phase = defaultdict(lambda: defaultdict(int))
    for event in runtime.monitor.comm_events:
        if event["round_id"] == round_id and event["delivered"]:
            phase = event.get("phase")
            if phase:
                by_phase[phase][event["dst"].removeprefix("client_")] += 1
    runtime.monitor.record_round_context(
        experiment_id=runtime.run_name, method=cfg.algorithm, seed=cfg.seed,
        round=round_id, node_ids=[str(node) for node in range(cfg.num_clients)],
        roles=roles, parents={}, children_counts={},
        cluster_heads=[str(head) for head in heads],
        cluster_members={str(head): [str(node) for node in members]
                         for head, members in groups.items()},
        active_nodes=[str(node) for node in active],
        failed_nodes=[str(node) for node in dropped],
        eligible_node_ratio=len(active) / max(1, cfg.num_clients),
        reachable_node_ratio=len(aggregated_nodes) / max(1, len(active)),
        eligible_sample_ratio=1.0,
        aggregated_sample_ratio=len(aggregated_nodes) / max(1, len(active)),
        round_completion_ratio=1.0,
        aggregation_inputs={},
        aggregation_inputs_by_phase={key: dict(value) for key, value in by_phase.items()},
        round_wall_clock_seconds=elapsed, round_latency_sim_ms=None,
        feduc_global_sync=bool(global_sync),
    )


def run(runtime: Any, cfg: Any) -> None:
    params = runtime.init_model_params()
    runtime.save_experiment_inputs(params)
    set_all_models(runtime, params)
    model_bytes = sum(value.numel() * value.element_size() for value in params.values())
    groups, paths, sample_counts, audit = _prepare_assignment(
        runtime, cfg, model_bytes
    )
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "feduc_assignment.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    interval = int(cfg.feduc_global_sync_interval)
    if interval <= 0:
        raise ValueError("feduc_global_sync_interval must be positive")
    print(f"Running FedUC-inspired CenSyn: heads={sorted(groups)}, I={interval}")
    start_time = time.perf_counter()

    for round_id in range(cfg.rounds):
        round_start = time.perf_counter()
        selected = runtime.select_clients()
        active, dropped = runtime.inject_dropouts(selected, round_id)
        active_set = set(active)
        if not active:
            raise RuntimeError("FedUC has no active clients for this round")
        train_results = runtime.train_selected_clients(active, round_id)
        payloads = ray.get([runtime.clients[node].upload_model.remote() for node in active])
        states = {item["client_id"]: item["params"] for item in payloads}
        cluster_states = {}
        aggregated_nodes = set()
        for head, members in groups.items():
            live = [node for node in members if node in active_set]
            if head not in active_set or not live:
                continue
            parts = []
            for node in live:
                delivered = node == head or record_path(
                    runtime, round_id, paths[(node, head)], model_bytes, "upload",
                    "LOCAL_MODEL", "INTRA_CLUSTER_UPLOAD",
                    f"feduc:{round_id}:cluster:{head}:up:{node}",
                )
                if delivered:
                    parts.append(leaf_weighted_state(states[node], sample_counts[node]))
                    aggregated_nodes.add(node)
            if parts:
                cluster_states[head] = merge_weighted_states(parts)

        if not cluster_states:
            raise RuntimeError("FedUC produced no cluster aggregate")
        global_sync = ((round_id + 1) % interval == 0 or round_id == cfg.rounds - 1)
        if global_sync:
            delivered_parts = []
            for head, state in cluster_states.items():
                if runtime._record_comm(
                    round_id, f"client_{head}", "server", model_bytes, "upload",
                    message_type="CLUSTER_MODEL", phase="INTER_CLUSTER_UPLOAD",
                    flow_id=f"feduc:{round_id}:inter-up:{head}",
                    link_type="INTER_CLUSTER",
                ):
                    delivered_parts.append(state)
            if not delivered_parts:
                raise RuntimeError("FedUC global synchronization received no cluster models")
            params = finalize_weighted_state(merge_weighted_states(delivered_parts))
            for head, members in groups.items():
                if head not in active_set or head not in cluster_states:
                    continue
                if runtime._record_comm(
                    round_id, "server", f"client_{head}", model_bytes, "broadcast",
                    message_type="GLOBAL_MODEL", phase="INTER_CLUSTER_DOWNLOAD",
                    flow_id=f"feduc:{round_id}:inter-down:{head}",
                    link_type="INTER_CLUSTER",
                ):
                    ray.get(runtime.clients[head].receive_model.remote(params))
                    for node in members:
                        if node != head and node in active_set:
                            if record_path(
                                runtime, round_id, list(reversed(paths[(node, head)])),
                                model_bytes, "broadcast", "GLOBAL_MODEL",
                                "INTRA_CLUSTER_DOWNLOAD",
                                f"feduc:{round_id}:cluster:{head}:global-down:{node}",
                            ):
                                ray.get(runtime.clients[node].receive_model.remote(params))
        else:
            for head, state in cluster_states.items():
                cluster_params = finalize_weighted_state(state)
                ray.get(runtime.clients[head].receive_model.remote(cluster_params))
                for node in groups[head]:
                    if node != head and node in active_set:
                        if record_path(
                            runtime, round_id, list(reversed(paths[(node, head)])),
                            model_bytes, "broadcast", "CLUSTER_MODEL",
                            "INTRA_CLUSTER_DOWNLOAD",
                            f"feduc:{round_id}:cluster:{head}:down:{node}",
                        ):
                            ray.get(runtime.clients[node].receive_model.remote(cluster_params))
            # Evaluation-only weighted consensus; it is not distributed.
            params = finalize_weighted_state(merge_weighted_states(list(cluster_states.values())))

        val_loss, val_acc = runtime.evaluate_global(params)
        summary = runtime.summarize_round_comm(round_id)
        elapsed = time.perf_counter() - round_start
        _record_context(runtime, cfg, round_id, groups, active, dropped,
                        aggregated_nodes, elapsed, global_sync)
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
              f"global_sync={global_sync}, messages={summary['comm_success_count']}")

    runtime.save_final_model(params)
    print(f"FedUC-inspired CenSyn completed in {time.perf_counter() - start_time:.2f}s")
