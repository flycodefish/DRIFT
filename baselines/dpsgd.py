from __future__ import annotations

import time
from typing import Any

import ray

from .common import connected_components, record_path, set_all_models, undirected_neighbors


def metropolis_weights(neighbors: dict[int, list[int]], receiver: int) -> dict[int, float]:
    degree = len(neighbors.get(receiver, []))
    weights = {}
    for sender in sorted(neighbors.get(receiver, [])):
        if receiver not in neighbors.get(sender, []):
            raise ValueError("Metropolis mixing requires an undirected physical graph")
        weights[sender] = 1.0 / (1.0 + max(degree, len(neighbors.get(sender, []))))
    weights[receiver] = 1.0 - sum(weights.values())
    if weights[receiver] < -1e-12:
        raise ValueError(f"invalid Metropolis self weight for node {receiver}")
    weights[receiver] = max(0.0, weights[receiver])
    return weights


def _record_context(runtime: Any, cfg: Any, round_id: int, selected: list[int], active: list[int],
                    dropped: list[int], graph: dict[int, list[int]], wall_clock: float) -> None:
    largest_component = max((len(component) for component in connected_components(graph)),
                            default=0)
    runtime.monitor.record_round_context(
        experiment_id=runtime.run_name, method=cfg.algorithm, seed=cfg.seed,
        round=round_id, node_ids=[str(node) for node in range(cfg.num_clients)],
        roles={str(node): "PEER" for node in range(cfg.num_clients)},
        parents={}, children_counts={},
        active_nodes=[str(node) for node in active],
        failed_nodes=[str(node) for node in dropped],
        eligible_node_ratio=len(active) / max(1, len(active) + len(dropped)),
        reachable_node_ratio=largest_component / max(1, len(active)),
        eligible_sample_ratio=(sum(len(runtime.client_indices[node]) for node in active) /
                               max(1, sum(len(runtime.client_indices[node]) for node in selected))),
        aggregated_sample_ratio=None, round_completion_ratio=1.0,
        aggregation_inputs={}, aggregation_inputs_by_phase={},
        round_wall_clock_seconds=wall_clock, round_latency_sim_ms=None,
    )


def run(runtime: Any, cfg: Any) -> None:
    params = runtime.init_model_params()
    runtime.save_experiment_inputs(params)
    set_all_models(runtime, params)
    print(f"Running synchronous D-PSGD on {cfg.topology} topology")
    start_time = time.perf_counter()

    for round_id in range(cfg.rounds):
        round_start = time.perf_counter()
        selected = runtime.select_clients()
        active, dropped = runtime.inject_dropouts(selected, round_id)
        if not active:
            raise RuntimeError("D-PSGD has no active clients for this round")
        active_graph = undirected_neighbors(runtime.neighbors, set(active))
        train_results = runtime.train_selected_clients(active, round_id)
        snapshots = ray.get([runtime.clients[node].upload_model.remote() for node in active])
        snapshot_map = {item["client_id"]: item["params"] for item in snapshots}
        payload_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in snapshots[0]["params"].values()
        )
        refs = []
        successful_edges = 0
        for receiver in active:
            weights = metropolis_weights(active_graph, receiver)
            sender_ids = [node for node in sorted(weights) if node != receiver]
            successful_sender_ids = []
            sender_params = []
            sender_weights = [weights[receiver]]
            for sender in sender_ids:
                if record_path(
                    runtime, round_id, [sender, receiver], payload_bytes, "gossip",
                    "PEER_MODEL", "PEER_EXCHANGE",
                    f"dpsgd:{round_id}:{sender}:{receiver}",
                ):
                    sender_params.append(snapshot_map[sender])
                    successful_sender_ids.append(sender)
                    sender_weights.append(weights[sender])
                    successful_edges += 1
            refs.append(runtime.clients[receiver].receive_weighted_push.remote(
                sender_params, successful_sender_ids, sender_weights,
            ))
        ray.get(refs)
        diagnostic_interval = max(1, int(cfg.decentralized_diagnostic_interval))
        include_local = (round_id % diagnostic_interval == 0 or
                         round_id == cfg.rounds - 1)
        params, metrics = runtime.evaluate_decentralized(active, include_local=include_local)
        summary = runtime.summarize_round_comm(round_id)
        avg_train_loss = sum(item["train_loss"] for item in train_results) / len(train_results)
        elapsed = time.perf_counter() - round_start
        _record_context(runtime, cfg, round_id, selected, active, dropped,
                        active_graph, elapsed)
        runtime.monitor.record_round(
            round_id=round_id, avg_train_loss=avg_train_loss,
            val_loss=metrics["consensus_loss"], val_acc=metrics["consensus_accuracy"],
            selected_count=len(selected), dropped_count=len(dropped),
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
              f"messages={summary['comm_success_count']}")

    runtime.save_final_model(params)
    print(f"D-PSGD completed in {time.perf_counter() - start_time:.2f}s")
