"""Actor-free communication traces for the Protocol V1 scalability study."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from baselines.common import build_resource_tree, shortest_path, undirected_neighbors
from deasfl.election import ClusteringManager, ElectionManager
from deasfl.runtime import build_client_tree_mapping


def _resource_params(resources: list[dict[str, Any]], aware: bool = True) -> list[dict[str, Any]]:
    result = []
    for item in resources:
        source = item.get("resource_params", item)
        value = {key: source[key] for key in (
            "cpu_cores", "memory_gb", "bandwidth_mbps", "gpu_memory_gb"
        )}
        if not aware:
            value["resource_score_override"] = 1.0
        result.append(value)
    return result


def _profile(link_profiles: dict[str, Any], src: int, dst: int) -> dict[str, float]:
    result = {
        "bandwidth_mbps": float(link_profiles.get("default_bandwidth_mbps", 100.0)),
        "latency_ms": float(link_profiles.get("default_latency_ms", 0.0)),
        "drop_rate": float(link_profiles.get("drop_rate", 0.0)),
    }
    overrides = link_profiles.get("edge_overrides", {})
    override = (overrides.get(f"{src}->{dst}") or overrides.get(f"{dst}->{src}") or {})
    result.update({key: float(value) for key, value in override.items() if key in result})
    return result


class TraceBuilder:
    def __init__(self, experiment_id: str, method: str, seed: int, round_id: int,
                 model_bytes: int, links: dict[str, Any]):
        self.experiment_id, self.method, self.seed = experiment_id, method, seed
        self.round_id, self.model_bytes, self.links = round_id, model_bytes, links
        self.events: list[dict[str, Any]] = []

    def transfer(self, src: int, dst: int, message_type: str, phase: str,
                 flow_id: str, hop_index: int = 1, hop_count: int = 1) -> None:
        profile = _profile(self.links, src, dst)
        index = len(self.events)
        self.events.append({
            "experiment_id": self.experiment_id, "method": self.method,
            "seed": self.seed, "round": self.round_id,
            "timestamp_start": None, "timestamp_end": None,
            "src": src, "dst": dst, "message_type": message_type,
            "message_id": f"r{self.round_id}-m{index}",
            "payload_bytes": self.model_bytes, **profile, "phase": phase,
            "flow_id": flow_id, "logical_parent_message_id": None,
            "link_type": "PHYSICAL_TRACE", "success": True,
            "physical_hop_index": hop_index, "physical_hop_count": hop_count,
            "attempt_count": 1, "backoff_ms": 0.0,
        })

    def path(self, path: list[int], message_type: str, phase: str, flow_id: str) -> None:
        count = len(path) - 1
        for index, (src, dst) in enumerate(zip(path[:-1], path[1:]), 1):
            self.transfer(src, dst, message_type, phase, flow_id, index, count)


def _context(experiment_id: str, method: str, seed: int, round_id: int,
             nodes: list[int], roles: dict[int, str], parents: dict[int, int | None],
             children: dict[int, list[int]], by_phase: dict[str, dict[str, int]]) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id, "method": method, "seed": seed,
        "round": round_id, "node_ids": nodes, "roles": roles,
        "parents": parents, "children_counts": {n: len(children.get(n, [])) for n in nodes},
        "active_nodes": nodes, "failed_nodes": [], "eligible_node_ratio": 1.0,
        "reachable_node_ratio": 1.0, "eligible_sample_ratio": 1.0,
        "aggregated_sample_ratio": 1.0, "round_completion_ratio": 1.0,
        "round_wall_clock_seconds": None, "round_latency_sim_ms": None,
        "aggregation_inputs_by_phase": by_phase,
    }


def _tree_inputs(children: dict[int, list[int]]) -> dict[str, int]:
    return {str(node): len(items) + 1 for node, items in children.items() if items}


def resource_tree_trace(graph: dict[int, list[int]], resources: list[dict[str, Any]],
                        links: dict[str, Any], model_bytes: int, seed: int,
                        rounds: int, experiment_id: str) -> tuple[list[dict], list[dict]]:
    graph = undirected_neighbors(graph)
    edge_cost = {}
    for src, neighbors in graph.items():
        for dst in neighbors:
            p = _profile(links, src, dst)
            edge_cost[(src, dst)] = p["latency_ms"] + model_bytes * 8 / p["bandwidth_mbps"] / 1000
    root, parents, children = build_resource_tree(graph, resources, edge_cost)
    events, contexts = [], []
    roles = {node: ("RESOURCE_ROOT" if node == root else "CLIENT") for node in graph}
    for round_id in range(rounds):
        builder = TraceBuilder(experiment_id, "resource_tree", seed, round_id, model_bytes, links)
        for child, parent in sorted(parents.items()):
            if parent is not None:
                builder.transfer(child, parent, "AGGREGATED_MODEL", "MODEL_UPLOAD", "resource-tree:up")
        for child, parent in sorted(parents.items()):
            if parent is not None:
                builder.transfer(parent, child, "GLOBAL_MODEL", "MODEL_DISTRIBUTION", "resource-tree:down")
        events.extend(builder.events)
        contexts.append(_context(experiment_id, "resource_tree", seed, round_id,
                                 list(graph), roles, parents, children,
                                 {"MODEL_UPLOAD": _tree_inputs(children)}))
    return events, contexts


def fedavg_edge_trace(graph: dict[int, list[int]], resources: list[dict[str, Any]],
                      links: dict[str, Any], model_bytes: int, seed: int,
                      rounds: int, experiment_id: str) -> tuple[list[dict], list[dict], int]:
    graph = undirected_neighbors(graph)
    # A strong, deterministic edge-server baseline: place it on the same
    # highest-resource root selected by the Resource-Tree oracle.
    server, _, _ = build_resource_tree(graph, resources)
    events, contexts = [], []
    children = {node: [] for node in graph}
    roles = {node: ("EDGE_SERVER" if node == server else "CLIENT") for node in graph}
    for round_id in range(rounds):
        builder = TraceBuilder(experiment_id, "fedavg_edge", seed, round_id, model_bytes, links)
        for node in graph:
            if node == server:
                continue
            path = shortest_path(graph, node, server)
            builder.path(path, "MODEL_UPLOAD", "MODEL_UPLOAD", f"fedavg:{node}:up")
            builder.path(list(reversed(path)), "GLOBAL_MODEL", "MODEL_DISTRIBUTION", f"fedavg:{node}:down")
        events.extend(builder.events)
        contexts.append(_context(experiment_id, "fedavg_edge", seed, round_id,
                                 list(graph), roles, {n: None for n in graph}, children,
                                 {"MODEL_UPLOAD": {str(server): len(graph)}}))
    return events, contexts, server


def drift_trace(graph: dict[int, list[int]], resources: list[dict[str, Any]],
                links: dict[str, Any], model_bytes: int, seed: int, rounds: int,
                experiment_id: str, config: dict[str, Any]) -> tuple[list[dict], list[dict], dict]:
    graph = undirected_neighbors(graph)
    topology = [graph[node] for node in range(len(graph))]
    clients = [{"id": node} for node in range(len(graph))]
    election = ElectionManager(
        clients, topology,
        base_trigger_time=float(config.get("election_base_trigger_time", 10.0)),
        trigger_factor=float(config.get("election_trigger_factor", 0.1)), seed=seed,
        resource_params=_resource_params(resources, bool(config.get("resource_awareness", True))),
    ).start_election()
    clustering = ClusteringManager(
        clients, topology, election,
        max_child=int(config.get("cluster_max_child", 3)),
        max_retry=int(config.get("cluster_max_retry", 3)),
        request_timeout=float(config.get("cluster_request_timeout", 2.0)),
        cluster_offset=float(config.get("cluster_offset", 30.0)),
        distribution_similarity_threshold=float(config.get("distribution_similarity_threshold", 0.0)),
    ).run_clustering()
    mapping = build_client_tree_mapping(clustering, len(graph))
    heads = list(clustering["cluster_heads"])
    root = int(election["best_client_id"])
    routing = {i: info.get("next_hop", -1) for i, info in enumerate(election["all_nodes_info"])}
    parents = {node: info.get("parent") for node, info in mapping.items()}
    children = {node: list(info.get("children", [])) for node, info in mapping.items()}
    roles = {node: ("ROOT" if node == root else "CLUSTER_HEAD" if node in heads
                    else "INTERMEDIATE" if children[node] else "CLIENT") for node in graph}
    events, contexts = [], []
    for round_id in range(rounds):
        builder = TraceBuilder(experiment_id, "drift", seed, round_id, model_bytes, links)
        for child, parent in sorted(parents.items()):
            if parent is not None:
                builder.transfer(child, parent, "AGGREGATED_MODEL", "INTRA_CLUSTER_UPLOAD", "drift:intra-up")
        for head in heads:
            path, seen, current = [head], {head}, head
            while current != root:
                current = int(routing[current])
                if current < 0 or current in seen:
                    raise ValueError(f"DRIFT route from cluster head {head} cannot reach root {root}")
                path.append(current); seen.add(current)
            builder.path(path, "AGGREGATED_MODEL", "INTER_CLUSTER_UPLOAD", f"drift:{head}:inter-up")
            builder.path(list(reversed(path)), "GLOBAL_MODEL", "INTER_CLUSTER_DOWNLOAD", f"drift:{head}:inter-down")
        for child, parent in sorted(parents.items()):
            if parent is not None:
                builder.transfer(parent, child, "GLOBAL_MODEL", "INTRA_CLUSTER_DOWNLOAD", "drift:intra-down")
        events.extend(builder.events)
        intra = _tree_inputs(children)
        contexts.append(_context(experiment_id, "drift", seed, round_id, list(graph),
                                 roles, parents, children,
                                 {"INTRA_CLUSTER_UPLOAD": intra,
                                  "INTER_CLUSTER_UPLOAD": {str(root): len(heads)}}))
    metadata = {"best_client": root, "cluster_heads": heads,
                "num_clusters": len(heads), "parents": parents}
    return events, contexts, metadata
