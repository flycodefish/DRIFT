"""Centralized, experiment-only cluster assignments for controlled ablations.

These helpers never inspect validation or test data.  They preserve a supplied
cluster-size profile and only change which client label distributions share a
cluster.  They are not part of the deployable distributed DRIFT protocol.
"""

from __future__ import annotations

import math
import random
from typing import Any, Iterable, Sequence

import numpy as np


JSD_QUANTILE_MODES = {
    "jsd_q00": 0.0,
    "jsd_q25": 0.25,
    "jsd_q50": 0.50,
    "jsd_q75": 0.75,
    "jsd_q100": 1.0,
}
EXPERIMENTAL_MODES = {
    "random_matched", "oracle_data_similar", *JSD_QUANTILE_MODES,
}


def _normalized_distribution(value: Sequence[float]) -> np.ndarray:
    distribution = np.asarray(value, dtype=np.float64)
    if distribution.ndim != 1 or distribution.size == 0:
        raise ValueError("Each label distribution must be a non-empty vector")
    if np.any(distribution < 0):
        raise ValueError("Label distributions cannot contain negative values")
    total = float(distribution.sum())
    if total <= 0:
        raise ValueError("Label distributions must have positive mass")
    return distribution / total


def pairwise_jsd_matrix(label_distributions: Sequence[Sequence[float]]) -> np.ndarray:
    """Return the natural-log Jensen-Shannon divergence matrix."""
    values = [_normalized_distribution(value) for value in label_distributions]
    if len({value.shape for value in values}) > 1:
        raise ValueError("All label distributions must have the same dimension")
    count = len(values)
    matrix = np.zeros((count, count), dtype=np.float64)
    epsilon = 1e-12
    for left in range(count):
        p = values[left] + epsilon
        p /= p.sum()
        for right in range(left + 1, count):
            q = values[right] + epsilon
            q /= q.sum()
            midpoint = (p + q) / 2.0
            divergence = float(
                (np.sum(p * np.log(p / midpoint))
                 + np.sum(q * np.log(q / midpoint))) / 2.0
            )
            matrix[left, right] = matrix[right, left] = divergence
    return matrix


def cluster_sizes_from_result(clustering_result: dict[str, Any]) -> list[int]:
    """Extract a complete cluster-size multiset from a DRIFT result.

    Distributed clustering can leave a client outside every rooted cluster.
    For the opt-in matched-assignment controls, preserve such clients as
    singleton capacities instead of silently dropping them.  Overlapping or
    unknown client IDs remain invalid because they cannot define a fair size
    profile.
    """
    node_info = clustering_result.get("all_nodes_info", [])
    expected = [info.get("client_id") for info in node_info]
    if any(client_id is None for client_id in expected) or len(set(expected)) != len(expected):
        raise ValueError("Reference clustering has invalid client IDs")
    expected_ids = set(expected)

    sizes = []
    covered: set[int] = set()
    for head_id in clustering_result.get("cluster_heads", []):
        cluster = clustering_result.get("clusters", {}).get(head_id)
        if cluster is None:
            cluster = clustering_result.get("clusters", {}).get(str(head_id), {})
        group = [head_id, *cluster.get("members", [])]
        group_ids = set(group)
        if len(group_ids) != len(group):
            raise ValueError("Reference cluster contains duplicate client IDs")
        if not group_ids <= expected_ids:
            raise ValueError("Reference cluster contains unknown client IDs")
        if covered & group_ids:
            raise ValueError("Reference clusters overlap")
        covered.update(group_ids)
        sizes.append(len(group_ids))

    sizes.extend([1] * len(expected_ids - covered))
    if not sizes:
        raise ValueError("Reference clustering does not define any clients")
    return sorted(sizes, reverse=True)


def _assignment_objective(groups: Sequence[Sequence[int]], distances: np.ndarray) -> float:
    total = 0.0
    for group in groups:
        for index, left in enumerate(group):
            for right in group[index + 1:]:
                total += float(distances[left, right])
    return total


def _random_groups(client_ids: Sequence[int], sizes: Sequence[int], rng: random.Random) -> list[list[int]]:
    shuffled = list(client_ids)
    rng.shuffle(shuffled)
    groups, offset = [], 0
    for size in sizes:
        groups.append(shuffled[offset:offset + size])
        offset += size
    return groups


def _improve_by_swaps(
    groups: list[list[int]],
    distances: np.ndarray,
    *,
    target: float | None = None,
    maximize: bool = False,
) -> list[list[int]]:
    """Deterministic pair-swap descent that preserves every cluster capacity."""
    def loss(score: float) -> float:
        if target is not None:
            return abs(score - target)
        return -score if maximize else score

    while True:
        current = _assignment_objective(groups, distances)
        best_loss = loss(current)
        best_swap = None
        for left_group in range(len(groups)):
            for right_group in range(left_group + 1, len(groups)):
                for left_pos in range(len(groups[left_group])):
                    for right_pos in range(len(groups[right_group])):
                        candidate = [list(group) for group in groups]
                        candidate[left_group][left_pos], candidate[right_group][right_pos] = (
                            candidate[right_group][right_pos], candidate[left_group][left_pos]
                        )
                        score = _assignment_objective(candidate, distances)
                        candidate_loss = loss(score)
                        if candidate_loss < best_loss - 1e-12:
                            best_loss = candidate_loss
                            best_swap = (left_group, left_pos, right_group, right_pos)
        if best_swap is None:
            return groups
        left_group, left_pos, right_group, right_pos = best_swap
        groups[left_group][left_pos], groups[right_group][right_pos] = (
            groups[right_group][right_pos], groups[left_group][left_pos]
        )


def _search_extreme_groups(
    client_ids: Sequence[int],
    sizes: Sequence[int],
    distances: np.ndarray,
    seed: int,
    restarts: int,
    *,
    maximize: bool,
) -> tuple[list[list[int]], float]:
    best_groups = None
    best_score = -math.inf if maximize else math.inf
    for restart in range(max(1, int(restarts))):
        groups = _random_groups(
            client_ids, sizes, random.Random(seed + restart * 104729)
        )
        groups = _improve_by_swaps(groups, distances, maximize=maximize)
        score = _assignment_objective(groups, distances)
        signature = tuple(tuple(sorted(group)) for group in groups)
        best_signature = (
            tuple(tuple(sorted(group)) for group in best_groups)
            if best_groups is not None else None
        )
        improves = score > best_score + 1e-12 if maximize else score < best_score - 1e-12
        if improves or (
            abs(score - best_score) <= 1e-12
            and (best_signature is None or signature < best_signature)
        ):
            best_score = score
            best_groups = [list(group) for group in groups]
    return best_groups, best_score


def matched_groups(
    label_distributions: Sequence[Sequence[float]],
    cluster_sizes: Sequence[int],
    mode: str,
    seed: int,
    oracle_restarts: int = 32,
) -> list[list[int]]:
    """Build a size-matched random or train-label-only Oracle assignment."""
    if mode not in EXPERIMENTAL_MODES:
        raise ValueError(f"Unsupported experimental clustering mode: {mode}")
    sizes = [int(size) for size in cluster_sizes]
    if any(size <= 0 for size in sizes):
        raise ValueError("Cluster sizes must be positive")
    client_ids = list(range(len(label_distributions)))
    if sum(sizes) != len(client_ids):
        raise ValueError("Cluster sizes must cover every client exactly once")
    if mode == "random_matched":
        groups = _random_groups(client_ids, sizes, random.Random(seed))
        return [sorted(group) for group in groups]

    distances = pairwise_jsd_matrix(label_distributions)
    min_groups, min_score = _search_extreme_groups(
        client_ids, sizes, distances, seed, oracle_restarts, maximize=False
    )
    if mode in {"oracle_data_similar", "jsd_q00"}:
        return [sorted(group) for group in min_groups]

    max_groups, max_score = _search_extreme_groups(
        client_ids, sizes, distances, seed + 49999, oracle_restarts, maximize=True
    )
    if mode == "jsd_q100":
        return [sorted(group) for group in max_groups]

    quantile = JSD_QUANTILE_MODES[mode]
    target = min_score + quantile * (max_score - min_score)
    candidates = [min_groups, max_groups]
    for restart in range(max(1, int(oracle_restarts))):
        groups = _random_groups(
            client_ids, sizes, random.Random(seed + 99991 + restart * 104729)
        )
        candidates.append(_improve_by_swaps(groups, distances, target=target))
    best_groups = min(
        candidates,
        key=lambda groups: (
            abs(_assignment_objective(groups, distances) - target),
            tuple(tuple(sorted(group)) for group in groups),
        ),
    )
    return [sorted(group) for group in best_groups]


def build_clustering_result(
    groups: Iterable[Sequence[int]],
    election_results: dict[str, Any],
    max_child: int,
    mode: str,
) -> dict[str, Any]:
    """Convert fixed membership groups to the existing DRIFT result schema."""
    if max_child <= 0:
        raise ValueError("max_child must be positive")
    node_info = election_results.get("all_nodes_info", [])
    scores = {
        node_id: float(info.get("my_score", 0.0))
        for node_id, info in enumerate(node_info)
    }
    parents: dict[int, int | None] = {}
    cluster_ids: dict[int, int] = {}
    cluster_map: dict[int, dict[str, Any]] = {}
    heads = []
    for raw_group in groups:
        group = sorted(int(node_id) for node_id in raw_group)
        if not group:
            raise ValueError("Clusters cannot be empty")
        head = min(group, key=lambda node_id: (-scores.get(node_id, 0.0), node_id))
        ordered = [head] + [node_id for node_id in group if node_id != head]
        parents[head] = None
        for position, node_id in enumerate(ordered[1:], start=1):
            parent_position = (position - 1) // max_child
            parents[node_id] = ordered[parent_position]
        for node_id in group:
            cluster_ids[node_id] = head
        heads.append(head)
        cluster_map[head] = {
            "head": head,
            "members": [node_id for node_id in group if node_id != head],
        }

    expected = set(range(len(node_info)))
    if set(parents) != expected:
        raise ValueError("Experimental clusters must contain every client exactly once")
    all_nodes_info = []
    for node_id in sorted(parents):
        parent = parents[node_id]
        all_nodes_info.append({
            "client_id": node_id,
            "parent": parent,
            "cluster_id": cluster_ids[node_id],
            "state": "cluster_head" if parent is None else "member",
            "resource_score": scores.get(node_id, 0.0),
            "assignment_mode": mode,
        })
    return {
        "cluster_heads": sorted(heads),
        "num_clusters": len(heads),
        "clusters": cluster_map,
        "all_nodes_info": all_nodes_info,
        "assignment_mode": mode,
        "experimental_only": True,
    }


def build_matched_clustering_result(
    reference_result: dict[str, Any],
    label_distributions: Sequence[Sequence[float]],
    election_results: dict[str, Any],
    max_child: int,
    mode: str,
    seed: int,
) -> dict[str, Any]:
    sizes = cluster_sizes_from_result(reference_result)
    groups = matched_groups(label_distributions, sizes, mode=mode, seed=seed)
    return build_clustering_result(groups, election_results, max_child, mode)
