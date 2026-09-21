"""Pure, deterministic helpers for the independent FedUC reproduction.

The paper's complete CenSyn convergence-bound estimator is intentionally not
approximated here.  ``label_distribution_proxy`` is an explicit FedUC-inspired
fallback and must never be reported as the original FedUC objective.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import heapq
import math


@dataclass(frozen=True)
class AssignmentDecision:
    worker_id: int
    aggregator_id: int
    objective: float
    completion_ms: float
    upload_ms: float


def deterministic_resource_score(resource: Mapping[str, float]) -> float:
    return (
        3.0 * float(resource.get("cpu_cores", 0.0))
        + 0.8 * float(resource.get("memory_gb", 0.0))
        + 0.02 * float(resource.get("bandwidth_mbps", 0.0))
        + 1.5 * float(resource.get("gpu_memory_gb", 0.0))
    )


def select_aggregators(
    resources: Sequence[Mapping[str, float]], count: int,
) -> list[int]:
    """Select top deterministic resource-score nodes, then lower node id."""
    if not 1 <= int(count) <= len(resources):
        raise ValueError("aggregator count must be in [1, num_clients]")
    by_id = {
        int(item.get("node_id", index)): item
        for index, item in enumerate(resources)
    }
    if len(by_id) != len(resources):
        raise ValueError("node resource ids must be unique")
    return sorted(
        by_id,
        key=lambda node: (-deterministic_resource_score(by_id[node]), node),
    )[: int(count)]


def local_training_times_ms(
    sample_counts: Mapping[int, int],
    resources: Sequence[Mapping[str, float]],
    *,
    local_epochs: int,
    reference_ms_per_sample_epoch: float,
) -> dict[int, float]:
    by_id = {
        int(item.get("node_id", index)): item
        for index, item in enumerate(resources)
    }
    result = {}
    for node, samples in sample_counts.items():
        capacity = float(by_id[node].get("compute_capacity", 0.0))
        if capacity <= 0:
            raise ValueError(f"compute capacity must be positive for node {node}")
        result[node] = (
            float(reference_ms_per_sample_epoch)
            * int(samples) * int(local_epochs) / capacity
        )
    return result


def shortest_latency_path(
    graph: Mapping[int, Sequence[int]],
    source: int,
    destination: int,
    edge_latency_ms: Mapping[tuple[int, int], float],
) -> tuple[list[int], float]:
    """Return a deterministic minimum-latency physical path."""
    if source == destination:
        return [source], 0.0
    distances = {int(source): 0.0}
    paths = {int(source): (int(source),)}
    queue = [(0.0, (int(source),), int(source))]
    while queue:
        distance, signature, node = heapq.heappop(queue)
        if distance != distances.get(node) or signature != paths.get(node):
            continue
        if node == destination:
            return list(signature), distance
        for neighbor in sorted(graph.get(node, [])):
            cost = float(edge_latency_ms[(node, neighbor)])
            candidate = distance + cost
            candidate_path = (*signature, int(neighbor))
            if (candidate < distances.get(neighbor, math.inf) - 1e-12 or
                    (abs(candidate - distances.get(neighbor, math.inf)) <= 1e-12
                     and candidate_path < paths.get(neighbor, (math.inf,)))):
                distances[neighbor] = candidate
                paths[neighbor] = candidate_path
                heapq.heappush(queue, (candidate, candidate_path, neighbor))
    raise ValueError(f"no physical path from {source} to {destination}")


def time_sharing_completion_ms(
    members: Sequence[int],
    aggregator: int,
    train_ms: Mapping[int, float],
    upload_ms: Mapping[tuple[int, int], float],
) -> float:
    """Paper Eq. (50)-(51), with deterministic node-id tie breaking."""
    ordered = sorted(members, key=lambda node: (train_ms[node], node))
    completion = 0.0
    for worker in ordered:
        transfer = 0.0 if worker == aggregator else upload_ms[(worker, aggregator)]
        completion = max(completion, float(train_ms[worker])) + float(transfer)
    return completion


def _balanced_capacities(num_workers: int, aggregators: Sequence[int]) -> dict[int, int]:
    base, remainder = divmod(int(num_workers), len(aggregators))
    return {
        aggregator: base + (index < remainder)
        for index, aggregator in enumerate(sorted(aggregators))
    }


def build_reference_assignment(
    workers: Sequence[int],
    aggregators: Sequence[int],
    upload_ms: Mapping[tuple[int, int], float],
) -> dict[int, list[int]]:
    """Build a deterministic balanced minimum-upload reference assignment."""
    worker_set = set(workers)
    if not set(aggregators) <= worker_set:
        raise ValueError("every aggregator must also be a worker")
    capacities = _balanced_capacities(len(workers), aggregators)
    groups = {aggregator: [aggregator] for aggregator in aggregators}
    remaining = {aggregator: capacities[aggregator] - 1 for aggregator in aggregators}
    for worker in sorted(worker_set - set(aggregators)):
        feasible = [aggregator for aggregator in aggregators if remaining[aggregator] > 0]
        if not feasible:
            raise RuntimeError("balanced reference assignment exhausted all capacities")
        chosen = min(feasible, key=lambda aggregator: (upload_ms[(worker, aggregator)], aggregator))
        groups[chosen].append(worker)
        remaining[chosen] -= 1
    return {aggregator: sorted(members) for aggregator, members in groups.items()}


def frozen_deadline_ms(
    reference_groups: Mapping[int, Sequence[int]],
    train_ms: Mapping[int, float],
    upload_ms: Mapping[tuple[int, int], float],
    *,
    slack: float = 1.25,
) -> float:
    if float(slack) < 1.0:
        raise ValueError("deadline slack must be at least 1.0")
    worst = max(
        time_sharing_completion_ms(members, aggregator, train_ms, upload_ms)
        for aggregator, members in reference_groups.items()
    )
    return float(slack) * worst


def label_distribution_proxy(
    groups: Mapping[int, Sequence[int]],
    label_counts: Mapping[int, Sequence[int]],
) -> float:
    """Sample-mass-weighted cluster-to-global squared distribution distance."""
    if not label_counts:
        raise ValueError("label counts cannot be empty")
    width = len(next(iter(label_counts.values())))
    global_counts = [0.0] * width
    for counts in label_counts.values():
        if len(counts) != width:
            raise ValueError("all label-count vectors must have equal length")
        for index, value in enumerate(counts):
            global_counts[index] += float(value)
    total = sum(global_counts)
    if total <= 0:
        raise ValueError("global label mass must be positive")
    global_dist = [value / total for value in global_counts]
    objective = 0.0
    for members in groups.values():
        cluster = [0.0] * width
        for worker in members:
            for index, value in enumerate(label_counts[worker]):
                cluster[index] += float(value)
        mass = sum(cluster)
        if mass <= 0:
            continue
        distance = sum((cluster[index] / mass - global_dist[index]) ** 2
                       for index in range(width))
        objective += (mass / total) * distance
    return objective


def greedy_assignment(
    workers: Sequence[int],
    aggregators: Sequence[int],
    sample_counts: Mapping[int, int],
    train_ms: Mapping[int, float],
    upload_ms: Mapping[tuple[int, int], float],
    deadline_ms: float,
    objective: Callable[[Mapping[int, Sequence[int]]], float],
) -> tuple[dict[int, list[int]], list[AssignmentDecision]]:
    """Deterministic Algorithm-2 assignment with fixed self-owned aggregators."""
    workers = sorted(set(int(node) for node in workers))
    aggregators = sorted(set(int(node) for node in aggregators))
    if not aggregators or not set(aggregators) <= set(workers):
        raise ValueError("aggregators must be a non-empty subset of workers")
    if deadline_ms <= 0 or not math.isfinite(deadline_ms):
        raise ValueError("deadline must be finite and positive")
    groups = {aggregator: [aggregator] for aggregator in aggregators}
    decisions = []
    queue = sorted(
        set(workers) - set(aggregators),
        key=lambda node: (-int(sample_counts[node]), node),
    )
    for worker in queue:
        candidates = []
        for aggregator in aggregators:
            trial = {head: list(members) for head, members in groups.items()}
            trial[aggregator].append(worker)
            completion = time_sharing_completion_ms(
                trial[aggregator], aggregator, train_ms, upload_ms
            )
            if completion > deadline_ms + 1e-9:
                continue
            value = float(objective(trial))
            candidates.append((
                value, completion, upload_ms[(worker, aggregator)], aggregator,
            ))
        if not candidates:
            raise RuntimeError(
                f"FedUC assignment is infeasible for worker {worker} under deadline {deadline_ms}"
            )
        value, completion, transfer, chosen = min(candidates)
        groups[chosen].append(worker)
        decisions.append(AssignmentDecision(
            worker_id=worker, aggregator_id=chosen, objective=value,
            completion_ms=completion, upload_ms=float(transfer),
        ))
    return ({head: sorted(members) for head, members in groups.items()}, decisions)
