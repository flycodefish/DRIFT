from __future__ import annotations

from collections import deque
import heapq
from typing import Any

import ray
import torch

from deasfl.aggregation import (
    WeightedState,
    leaf_weighted_state,
    merge_weighted_states,
)


def client_endpoint(node_id: int) -> str:
    return f"client_{node_id}"


def set_all_models(runtime: Any, params: dict[str, torch.Tensor]) -> None:
    """Set the same initial/global state without charging bootstrap traffic."""
    ray.get([client.receive_model.remote(params) for client in runtime.clients])


def undirected_neighbors(neighbors: dict[int, list[int]],
                         nodes: set[int] | None = None) -> dict[int, list[int]]:
    """Interpret the frozen neighbor list as an undirected physical graph.

    The project topology artifact is an adjacency list and its mesh generator
    writes each undirected edge once.  Baselines must therefore explicitly
    close it symmetrically before constructing trees or Metropolis weights.
    """
    allowed = set(neighbors) if nodes is None else set(nodes)
    result = {node: set() for node in allowed}
    for src, values in neighbors.items():
        if src not in allowed:
            continue
        for dst in values:
            if dst in allowed and dst != src:
                result[src].add(dst)
                result[dst].add(src)
    return {node: sorted(values) for node, values in result.items()}


def connected_components(neighbors: dict[int, list[int]]) -> list[set[int]]:
    """Return deterministic connected components, largest component first."""
    remaining = set(neighbors)
    components = []
    while remaining:
        start = min(remaining)
        component = {start}
        queue = deque([start])
        remaining.remove(start)
        while queue:
            node = queue.popleft()
            for neighbor in neighbors.get(node, []):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return sorted(components, key=lambda values: (-len(values), min(values)))


def prune_fixed_tree(children: dict[int, list[int]], active: set[int],
                     root: int) -> tuple[dict[int, list[int]], set[int]]:
    """Keep only nodes reachable through active nodes in a fixed tree."""
    pruned = {node: [] for node in children}
    if root not in active:
        return pruned, set()
    reachable = {root}
    queue = deque([root])
    while queue:
        node = queue.popleft()
        for child in children.get(node, []):
            if child in active:
                pruned[node].append(child)
                reachable.add(child)
                queue.append(child)
    return pruned, reachable


def shortest_path(neighbors: dict[int, list[int]], source: int,
                  destination: int) -> list[int]:
    if source == destination:
        return [source]
    parents: dict[int, int | None] = {source: None}
    queue = deque([source])
    while queue:
        current = queue.popleft()
        for neighbor in sorted(neighbors.get(current, [])):
            if neighbor in parents:
                continue
            parents[neighbor] = current
            if neighbor == destination:
                path = [destination]
                while parents[path[-1]] is not None:
                    path.append(parents[path[-1]])
                return list(reversed(path))
            queue.append(neighbor)
    raise ValueError(f"no physical path from {source} to {destination}")


def resource_score(resources: list[dict[str, Any]], node_id: int) -> float:
    by_id = {int(item.get("node_id", index)): item
             for index, item in enumerate(resources)}
    item = by_id[node_id]
    params = item.get("resource_params", item)
    fields = ("compute_capacity", "bandwidth_mbps", "memory_gb")
    values = []
    for field in fields:
        values.append(float(params.get(field, 0.0)))
    maxima = [
        max(float((entry.get("resource_params", entry)).get(field, 0.0))
            for entry in resources) for field in fields
    ]
    # Frozen V1 score: compute throughput is the primary resource, followed
    # by network bandwidth and memory.  The constants are part of the
    # baseline definition, not tuned per run.
    weights = (0.50, 0.30, 0.20)
    return sum(weight * value / maximum
               for value, maximum, weight in zip(values, maxima, weights)
               if maximum > 0)


def build_resource_tree(neighbors: dict[int, list[int]], resources: list[dict[str, Any]],
                        edge_cost: dict[tuple[int, int], float] | None = None
                        ) -> tuple[int, dict[int, int | None], dict[int, list[int]]]:
    """Build a deterministic latency-weighted tree rooted at the best node."""
    nodes = sorted(neighbors)
    if not nodes:
        raise ValueError("physical topology is empty")
    root = max(nodes, key=lambda node: (resource_score(resources, node), -node))
    distances = {node: float("inf") for node in nodes}
    distances[root] = 0.0
    parents: dict[int, int | None] = {root: None}
    children: dict[int, list[int]] = {node: [] for node in nodes}
    queue: list[tuple[float, int]] = [(0.0, root)]
    while queue:
        distance, current = heapq.heappop(queue)
        if distance != distances[current]:
            continue
        for neighbor in sorted(neighbors.get(current, [])):
            weight = (edge_cost or {}).get((current, neighbor), 1.0)
            candidate = distance + weight
            old_parent = parents.get(neighbor)
            if (candidate < distances[neighbor] or
                    (candidate == distances[neighbor] and
                     (old_parent is None or current < old_parent))):
                distances[neighbor] = candidate
                parents[neighbor] = current
                heapq.heappush(queue, (candidate, neighbor))
    if len(parents) != len(nodes):
        raise ValueError("physical topology is disconnected; Resource-Tree cannot span all clients")
    for child, parent in parents.items():
        if parent is not None:
            children[parent].append(child)
    for children_list in children.values():
        children_list.sort()
    return root, parents, children


def build_fixed_tree(
    neighbors: dict[int, list[int]], root: int = 0
) -> tuple[int, dict[int, int | None], dict[int, list[int]]]:
    """Build a deterministic topology-only spanning tree.

    This is the fixed hierarchy baseline: the root and parent choices are
    determined once from node ids and the physical graph, never from resource
    scores, data distributions, elections, or runtime events.
    """
    nodes = sorted(neighbors)
    if not nodes:
        raise ValueError("physical topology is empty")
    if root not in neighbors:
        raise ValueError(f"fixed root {root} is not present in the topology")
    parents: dict[int, int | None] = {root: None}
    children: dict[int, list[int]] = {node: [] for node in nodes}
    queue = deque([root])
    while queue:
        current = queue.popleft()
        for neighbor in sorted(neighbors.get(current, [])):
            if neighbor in parents:
                continue
            parents[neighbor] = current
            children[current].append(neighbor)
            queue.append(neighbor)
    if len(parents) != len(nodes):
        raise ValueError("physical topology is disconnected; fixed hierarchy cannot span all clients")
    return root, parents, children


def _distances(neighbors: dict[int, list[int]], source: int) -> dict[int, int]:
    distances = {source: 0}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in sorted(neighbors[node]):
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return distances


def build_fixed_hierarchy(
    neighbors: dict[int, list[int]], root: int, num_clusters: int,
) -> tuple[list[int], dict[int, int], dict[int, int | None], dict[int, list[int]]]:
    """Create fixed topology-nearest clusters and their intra-cluster trees."""
    nodes = sorted(neighbors)
    if root not in neighbors:
        raise ValueError(f"fixed root {root} is not present in the topology")
    if not 1 <= num_clusters <= len(nodes):
        raise ValueError("fixed_hierarchical_num_clusters must be in [1, num_clients]")
    heads = [root]
    while len(heads) < num_clusters:
        nearest = {
            node: min(_distances(neighbors, head).get(node, float("inf")) for head in heads)
            for node in nodes if node not in heads
        }
        candidate = max(nearest, key=lambda node: (nearest[node], -node))
        heads.append(candidate)
    heads.sort()

    # Multi-source BFS yields connected, topology-nearest clusters.  A tie is
    # resolved by the lower head id because initial heads enter the queue sorted.
    owner = {head: head for head in heads}
    parents: dict[int, int | None] = {head: None for head in heads}
    queue = deque(heads)
    while queue:
        current = queue.popleft()
        for neighbor in sorted(neighbors[current]):
            if neighbor in owner:
                continue
            owner[neighbor] = owner[current]
            parents[neighbor] = current
            queue.append(neighbor)
    if len(owner) != len(nodes):
        raise ValueError("fixed hierarchy cannot span a disconnected topology")
    children = {node: [] for node in nodes}
    for node, parent in parents.items():
        if parent is not None:
            children[parent].append(node)
    for values in children.values():
        values.sort()
    return heads, owner, parents, children


def record_path(runtime: Any, round_id: int, path: list[int], model_bytes: int,
                kind: str, message_type: str, phase: str, flow_id: str,
                link_type: str = "DIRECT") -> bool:
    """Record a routed model transfer and stop at its first failed hop."""
    hop_count = max(0, len(path) - 1)
    for hop_index, (src, dst) in enumerate(zip(path, path[1:]), start=1):
        if not runtime._record_comm(
            round_id, client_endpoint(src), client_endpoint(dst), model_bytes, kind,
            message_type=message_type, phase=phase, flow_id=flow_id,
            link_type=link_type, physical_hop_index=hop_index,
            physical_hop_count=hop_count,
        ):
            return False
    return True


def aggregate_tree(node_id: int, children: dict[int, list[int]],
                   model_states: dict[int, dict[str, torch.Tensor]],
                   sample_counts: dict[int, int],
                   runtime: Any, round_id: int,
                   model_bytes: int, *, phase: str = "MODEL_UPLOAD",
                   message_type: str = "AGGREGATED_MODEL",
                   flow_prefix: str = "tree"
                   ) -> tuple[WeightedState, set[int]]:
    """Return an unnormalised float64 subtree sum.

    Delaying normalisation until the final root fusion keeps hierarchical
    sample-weighted aggregation numerically aligned with flat FedAvg.
    """
    parts = [leaf_weighted_state(model_states[node_id], sample_counts[node_id])]
    included_nodes = {node_id}
    for child_id in children[node_id]:
        child_state, child_nodes = aggregate_tree(
            child_id, children, model_states, sample_counts, runtime,
            round_id, model_bytes, phase=phase, message_type=message_type,
            flow_prefix=flow_prefix,
        )
        if record_path(
            runtime, round_id, [child_id, node_id], model_bytes, "upload",
            message_type, phase, f"{flow_prefix}:up:{child_id}:{node_id}",
        ):
            parts.append(child_state)
            included_nodes.update(child_nodes)
    return merge_weighted_states(parts), included_nodes


def distribute_tree(node_id: int, children: dict[int, list[int]],
                    params: dict[str, torch.Tensor], runtime: Any,
                    round_id: int, model_bytes: int, *,
                    phase: str = "MODEL_DISTRIBUTION",
                    message_type: str = "GLOBAL_MODEL",
                    flow_prefix: str = "tree") -> set[int]:
    delivered_nodes = {node_id}
    for child_id in children[node_id]:
        if record_path(
            runtime, round_id, [node_id, child_id], model_bytes, "broadcast",
            message_type, phase, f"{flow_prefix}:down:{node_id}:{child_id}",
        ):
            ray.get(runtime.clients[child_id].receive_model.remote(params))
            delivered_nodes.update(distribute_tree(
                child_id, children, params, runtime, round_id, model_bytes,
                phase=phase, message_type=message_type, flow_prefix=flow_prefix,
            ))
    return delivered_nodes
