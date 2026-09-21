from __future__ import annotations

import pytest

from baselines.feduc_objective import (
    build_reference_assignment,
    deterministic_resource_score,
    frozen_deadline_ms,
    greedy_assignment,
    label_distribution_proxy,
    select_aggregators,
    shortest_latency_path,
    time_sharing_completion_ms,
)


def test_select_aggregators_is_deterministic_and_uses_node_id_tie_break():
    resources = [
        {"node_id": 2, "cpu_cores": 4, "memory_gb": 8, "bandwidth_mbps": 100, "gpu_memory_gb": 0},
        {"node_id": 0, "cpu_cores": 8, "memory_gb": 8, "bandwidth_mbps": 100, "gpu_memory_gb": 0},
        {"node_id": 1, "cpu_cores": 8, "memory_gb": 8, "bandwidth_mbps": 100, "gpu_memory_gb": 0},
    ]
    assert deterministic_resource_score(resources[1]) == deterministic_resource_score(resources[2])
    assert select_aggregators(resources, 2) == [0, 1]


def test_time_sharing_completion_matches_paper_recurrence():
    train = {0: 3.0, 1: 1.0, 2: 2.0}
    upload = {(0, 0): 0.0, (1, 0): 4.0, (2, 0): 1.0}
    # worker 1: max(0,1)+4=5; worker 2: max(5,2)+1=6;
    # aggregator 0: max(6,3)+0=6.
    assert time_sharing_completion_ms([0, 1, 2], 0, train, upload) == 6.0


def test_shortest_latency_path_prefers_lower_cost_then_lexicographic_path():
    graph = {0: [1, 2], 1: [0, 3], 2: [0, 3], 3: [1, 2]}
    costs = {(a, b): 1.0 for a, values in graph.items() for b in values}
    assert shortest_latency_path(graph, 0, 3, costs) == ([0, 1, 3], 2.0)


def test_reference_assignment_covers_workers_once_and_deadline_is_frozen():
    workers = [0, 1, 2, 3]
    heads = [0, 2]
    upload = {(worker, head): abs(worker - head) for worker in workers for head in heads}
    groups = build_reference_assignment(workers, heads, upload)
    assert sorted(node for members in groups.values() for node in members) == workers
    assert all(len(members) == 2 for members in groups.values())
    train = {node: 1.0 for node in workers}
    assert frozen_deadline_ms(groups, train, upload, slack=1.25) == 2.5


def test_greedy_assignment_uses_proxy_and_is_deterministic():
    workers = [0, 1, 2, 3]
    heads = [0, 2]
    samples = {node: 10 for node in workers}
    train = {node: 1.0 for node in workers}
    upload = {(worker, head): (0.0 if worker == head else 1.0)
              for worker in workers for head in heads}
    labels = {
        0: [10, 0], 1: [0, 10], 2: [10, 0], 3: [0, 10],
    }
    objective = lambda groups: label_distribution_proxy(groups, labels)
    groups, decisions = greedy_assignment(
        workers, heads, samples, train, upload, 10.0, objective
    )
    assert groups == {0: [0, 1], 2: [2, 3]}
    assert [decision.worker_id for decision in decisions] == [1, 3]


def test_greedy_assignment_fails_instead_of_relaxing_deadline():
    with pytest.raises(RuntimeError, match="infeasible"):
        greedy_assignment(
            [0, 1], [0], {0: 1, 1: 1}, {0: 1.0, 1: 1.0},
            {(0, 0): 0.0, (1, 0): 100.0}, 2.0,
            lambda groups: 0.0,
        )
