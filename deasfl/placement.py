"""
Locality-Aware Client Placement for DRIFT.

This module implements the "distribution -> physical node" mapping (a
permutation ``pi``) WITHOUT touching the original Dirichlet data generation.

Two concepts are kept strictly separate:

  A. Data-distribution generation:
     The project's existing Dirichlet partition produces ``N`` client
     datasets with label distributions ``P_1(y), ..., P_N(y)``.  This step
     is left untouched and keeps its original randomness.

  B. Node placement:
     Given those ``N`` distributions plus a topology ``E``, we decide which
     generated distribution is hosted by which physical/topology node.  Only
     this permutation ``pi`` is ever changed here.  No client dataset is
     edited, so the multiset of client sample indices is invariant.

The placement DECISION reads only:
  * ``label_distributions`` (derived from the train partition) and
  * the topology ``E``.
It never reads test data, DAAC clustering output, or training metrics.

``mode == "random"``   -> a seed-controlled random permutation (control).
``mode == "locality"`` -> a seed-controlled stochastic pair-swap that lowers
                          the mean JSD over topology edges, starting from the
                          same seed's random permutation.

This intends a SOFT spatial-locality prior, i.e.:
    P( JSD(P_i,P_j) is small | (i,j) is an edge )  >
    P( JSD(P_i,P_j) is small | (i,j) is not an edge )
It is *not* a fixed semantic partitioning, and it is not optimized to
convergence.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# JSD
# ---------------------------------------------------------------------------
def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (symmetric, in ``[0, ln2]``)."""
    eps = 1e-10
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))) / 2.0


# ---------------------------------------------------------------------------
# Topology / edge helpers
# ---------------------------------------------------------------------------
def build_edges(topology: Sequence[Sequence[int]]) -> List[Tuple[int, int]]:
    """Return the undirected unique edge list for a neighbour-list topology."""
    edges = set()
    for i, neighbours in enumerate(topology):
        for j in neighbours:
            if i != j:
                edges.add((min(i, j), max(i, j)))
    return sorted(edges)


def build_non_edges(
    num_nodes: int, edges: Sequence[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """Return all unordered node pairs that are NOT edges."""
    edge_set = set(edges)
    return [
        (i, j)
        for i in range(num_nodes)
        for j in range(i + 1, num_nodes)
        if (i, j) not in edge_set
    ]


def edge_jsd_values(
    perm: Sequence[int],
    dists: Sequence[np.ndarray],
    edges: Sequence[Tuple[int, int]],
) -> List[float]:
    return [jsd(dists[perm[i]], dists[perm[j]]) for (i, j) in edges]


def nonedge_jsd_values(
    perm: Sequence[int],
    dists: Sequence[np.ndarray],
    non_edges: Sequence[Tuple[int, int]],
) -> List[float]:
    return [jsd(dists[perm[i]], dists[perm[j]]) for (i, j) in non_edges]


def mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


# ---------------------------------------------------------------------------
# Permutations
# ---------------------------------------------------------------------------
def random_permutation(num_nodes: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    perm = list(range(num_nodes))
    rng.shuffle(perm)
    return perm


def identity_permutation(num_nodes: int) -> List[int]:
    return list(range(num_nodes))


def locality_permutation(
    dists: Sequence[np.ndarray],
    edges: Sequence[Tuple[int, int]],
    steps: int,
    temperature: float,
    seed: int,
    init_perm: Optional[Sequence[int]] = None,
) -> Tuple[List[int], float]:
    """Seed-controlled stochastic pair-swap optimisation of EdgeJSD.

    Starts from ``init_perm`` (default: the same seed's random permutation),
    repeatedly swaps two topology nodes' distributions and keeps the swap when
    it lowers the mean edge JSD, occasionally (per ``temperature``) accepting a
    worse swap to escape local minima.  Runs for exactly ``steps`` iterations
    (no unbounded convergence).  Returns the best permutation and its score.

    Inputs are only ``dists`` (client label distributions) and ``edges``; it
    never touches test data, DAAC results, or training metrics.
    """
    num_nodes = len(dists)
    rng = random.Random(seed)
    rng_spawned = random.Random(seed ^ 0x9E3779B9)  # independent swap RNG

    cur = list(init_perm) if init_perm is not None else random_permutation(num_nodes, seed)
    cur_score = mean(edge_jsd_values(cur, dists, edges))
    best = list(cur)
    best_score = cur_score

    for _ in range(steps):
        a, b = rng_spawned.sample(range(num_nodes), 2)
        cand = list(cur)
        cand[a], cand[b] = cand[b], cand[a]
        cand_score = mean(edge_jsd_values(cand, dists, edges))
        delta = cand_score - cur_score
        accept = delta < 0 or (rng_spawned.random() < math.exp(-delta / max(temperature, 1e-12)))
        if accept:
            cur, cur_score = cand, cand_score
            if cur_score < best_score:
                best, best_score = list(cur), cur_score

    return best, best_score


# ---------------------------------------------------------------------------
# Applying a permutation to the per-client data
# ---------------------------------------------------------------------------
def apply_permutation(items: Sequence, perm: Sequence[int]) -> List:
    """Return ``items`` reordered so that output index ``i`` = ``items[perm[i]]``."""
    return [items[i] for i in perm]


def permutation_is_bijection(perm: Sequence[int], num_nodes: int) -> bool:
    return (
        len(perm) == num_nodes
        and sorted(perm) == list(range(num_nodes))
    )


# ---------------------------------------------------------------------------
# High level: compute a placement result from config + distributions + topology
# ---------------------------------------------------------------------------
def compute_placement(
    config,
    label_distributions: Sequence[np.ndarray],
    topology: Sequence[Sequence[int]],
    seed: Optional[int] = None,
) -> Dict:
    """Compute random and (optionally) locality placements for diagnostics.

    ``random`` is always produced.  ``locality`` is produced when the config's
    ``data_placement_mode == "locality"``.  Returns a dict with the permutation,
    the edge / non-edge JSD arrays, and summary statistics.  This function is
    deterministic given (seed, label_distributions, topology).
    """
    num = len(label_distributions)
    mode = getattr(config, "data_placement_mode", "random")
    steps = int(getattr(config, "locality_placement_steps", 2000))
    temperature = float(getattr(config, "locality_temperature", 0.05))
    if seed is None:
        seed = int(getattr(config, "seed", 42))
    loc_seed = getattr(config, "locality_seed", None)
    loc_seed = int(loc_seed) if loc_seed is not None else int(seed)

    edges = build_edges(topology)
    non_edges = build_non_edges(num, edges)

    perm_r = random_permutation(num, seed)
    edge_r = edge_jsd_values(perm_r, label_distributions, edges)
    nonedge_r = nonedge_jsd_values(perm_r, label_distributions, non_edges)

    result: Dict = {
        "num_nodes": num,
        "mode": mode,
        "seed": seed,
        "locality_seed": loc_seed,
        "steps": steps,
        "temperature": temperature,
        "edges": edges,
        "non_edges": non_edges,
        "random": {
            "permutation": perm_r,
            "edge_jsd": edge_r,
            "nonedge_jsd": nonedge_r,
        },
        "locality": None,
    }

    if mode == "locality":
        perm_l, best_score = locality_permutation(
            label_distributions, edges, steps, temperature, loc_seed, init_perm=perm_r
        )
        edge_l = edge_jsd_values(perm_l, label_distributions, edges)
        nonedge_l = nonedge_jsd_values(perm_l, label_distributions, non_edges)
        result["locality"] = {
            "permutation": perm_l,
            "edge_jsd": edge_l,
            "nonedge_jsd": nonedge_l,
            "best_edge_jsd": best_score,
        }

    return result


def summarize(values: Sequence[float]) -> Dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
