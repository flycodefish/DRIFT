import torch
from types import SimpleNamespace
from unittest.mock import Mock, patch

from baselines.common import (
    aggregate_tree,
    build_resource_tree,
    build_fixed_hierarchy,
    build_fixed_tree,
    connected_components,
    prune_fixed_tree,
    undirected_neighbors,
)
from baselines.resource_tree import _synchronize_rejoining_clients
from baselines.dpsgd import metropolis_weights
from baselines.dynamic_dpsgd import _apply_events, _edges, _targets, _trace, _validate_trace
from baselines.epidemic_learning import (
    default_epidemic_sample_size,
    resolve_epidemic_sample_size,
    sample_epidemic_targets,
)
from deasfl.aggregation import (
    finalize_weighted_state,
    leaf_weighted_state,
    merge_weighted_states,
)


def test_fixed_tree_is_topology_only_and_spanning():
    neighbors = {0: [1, 2], 1: [0, 3], 2: [0, 4], 3: [1], 4: [2]}
    root, parents, children = build_fixed_tree(neighbors, root=0)
    assert root == 0
    assert set(parents) == set(neighbors)
    assert sum(len(items) for items in children.values()) == len(neighbors) - 1
    assert parents[3] == 1
    assert parents[4] == 2


def test_baseline_graph_is_closed_to_undirected_physical_edges():
    graph = undirected_neighbors({0: [1, 2], 1: [2], 2: []})
    assert graph == {0: [1, 2], 1: [0, 2], 2: [0, 1]}


def test_connected_components_are_largest_first_and_deterministic():
    graph = {0: [1], 1: [0], 2: [], 3: [4], 4: [3]}
    assert connected_components(graph) == [{0, 1}, {3, 4}, {2}]


def test_fixed_tree_failure_cuts_descendants_without_repair():
    children = {0: [1], 1: [2], 2: []}
    pruned, reachable = prune_fixed_tree(children, {0, 2}, 0)
    assert reachable == {0}
    assert pruned == {0: [], 1: [], 2: []}


def test_dynamic_trace_target_and_edge_parsing():
    assert _targets({"target": 3}) == [3]
    assert _targets({"targets": [1, 2]}) == [1, 2]
    assert _edges({"edges": [[2, 0], [1, 3]]}) == [(0, 2), (1, 3)]


def test_dynamic_trace_rejects_missing_targets():
    import pytest
    with pytest.raises(ValueError):
        _targets({"event_type": "node_failure"})
    with pytest.raises(ValueError):
        _edges({"event_type": "link_failure"})


class _EventMonitor:
    def __init__(self):
        self.events = []

    def record_topology_event(self, **event):
        self.events.append(event)


def test_dynamic_events_are_persistent_and_recorded():
    monitor = _EventMonitor()
    available = {0, 1, 2}
    disabled = set()
    events = [
        {"round": 0, "event_type": "node_failure", "node_ids": [1]},
        {"round": 1, "event_type": "node_recovery", "node_ids": [1]},
        {"round": 1, "event_type": "link_failure", "edges": [[0, 2]]},
    ]
    _apply_events(events, 0, 3, available, disabled, monitor)
    assert available == {0, 2}
    _apply_events(events, 1, 3, available, disabled, monitor)
    assert available == {0, 1, 2} and disabled == {(0, 2)}
    assert [event["target_type"] for event in monitor.events] == ["node", "node", "link"]


def test_dynamic_trace_rejects_nonphysical_edges_and_bad_state_transitions():
    graph = {0: [1], 1: [0, 2], 2: [1]}
    with __import__("pytest").raises(ValueError, match="non-physical"):
        _validate_trace([{"round": 0, "event_type": "link_failure",
                          "edges": [[0, 2]]}], 3, graph, 1, set())
    with __import__("pytest").raises(ValueError, match="already failed"):
        _validate_trace([{"round": 0, "event_type": "link_failure",
                          "edges": [[0, 1]]},
                         {"round": 0, "event_type": "link_failure",
                          "edges": [[0, 1]]}], 3, graph, 1, set())


def test_dynamic_trace_rejects_non_object_events_cleanly():
    import pytest
    with pytest.raises(ValueError, match="must be an object"):
        _validate_trace(["not-an-event"], 2, {0: [1], 1: [0]}, 1, set())


def test_dynamic_trace_rejects_scalar_top_level():
    import pytest
    from types import SimpleNamespace
    runtime = SimpleNamespace(frozen_inputs=None)
    cfg = SimpleNamespace(dynamic_failure_trace_json="42")
    with pytest.raises(ValueError, match="list or an object"):
        _trace(runtime, cfg)


def test_fixed_hierarchy_has_fixed_connected_clusters():
    graph = undirected_neighbors({0: [1], 1: [2], 2: [3], 3: [4], 4: []})
    heads, owner, parents, children = build_fixed_hierarchy(graph, root=0, num_clusters=2)
    assert heads[0] == 0 and len(heads) == 2
    assert set(owner) == set(graph)
    assert all(parent in graph[node] for node, parent in parents.items() if parent is not None)
    assert sum(len(nodes) for nodes in children.values()) == len(graph) - len(heads)


def test_resource_tree_uses_latency_weighted_shortest_paths():
    graph = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    resources = [
        {"node_id": 0, "compute_capacity": 1, "bandwidth_mbps": 1, "memory_gb": 1},
        {"node_id": 1, "compute_capacity": 0, "bandwidth_mbps": 0, "memory_gb": 0},
        {"node_id": 2, "compute_capacity": 0, "bandwidth_mbps": 0, "memory_gb": 0},
    ]
    _, parents, _ = build_resource_tree(
        graph, resources,
        {(0, 1): 10, (1, 0): 10, (0, 2): 1, (2, 0): 1, (1, 2): 1, (2, 1): 1},
    )
    assert parents[1] == 2


def test_resource_tree_synchronizes_rejoining_client_before_training():
    receiver = Mock()
    receiver.remote.return_value = "object-ref"
    runtime = SimpleNamespace(
        clients=[SimpleNamespace(receive_model=Mock()),
                 SimpleNamespace(receive_model=receiver)],
    )
    params = {"weight": torch.ones(2)}
    with patch("baselines.resource_tree.record_path", return_value=True) as transfer, \
         patch("baselines.resource_tree.ray.get") as ray_get:
        _synchronize_rejoining_clients(
            runtime, params, active=[0, 1], previously_active={0},
            graph={0: [1], 1: [0]}, round_id=2,
        )
    transfer.assert_called_once()
    assert transfer.call_args.args[2] == [0, 1]
    receiver.remote.assert_called_once_with(params)
    ray_get.assert_called_once_with("object-ref")


def test_resource_tree_does_not_resynchronize_continuously_active_clients():
    runtime = SimpleNamespace(clients=[])
    with patch("baselines.resource_tree.record_path") as transfer:
        _synchronize_rejoining_clients(
            runtime, {"weight": torch.ones(1)}, active=[0],
            previously_active={0}, graph={0: []}, round_id=1,
        )
    transfer.assert_not_called()


def test_metropolis_weights_are_row_stochastic():
    neighbors = {0: [1, 2], 1: [0], 2: [0]}
    for node in neighbors:
        weights = metropolis_weights(neighbors, node)
        assert set(weights) == {node, *neighbors[node]}
        assert abs(sum(weights.values()) - 1.0) < 1e-12
        assert all(value >= 0.0 for value in weights.values())


def test_metropolis_rejects_directed_neighbor_lists():
    try:
        metropolis_weights({0: [1], 1: []}, 0)
    except ValueError as exc:
        assert "undirected" in str(exc)
    else:  # pragma: no cover - guards a silent invalid D-PSGD matrix.
        raise AssertionError("directed graph was accepted")


def test_epidemic_sample_size_and_targets_are_deterministic_and_exclude_self():
    import random

    assert default_epidemic_sample_size(1) == 0
    assert default_epidemic_sample_size(5) == 3
    first = sample_epidemic_targets(2, {0, 1, 2, 3, 4}, random.Random(44), 3)
    second = sample_epidemic_targets(2, {0, 1, 2, 3, 4}, random.Random(44), 3)
    assert first == second
    assert len(first) == 3 and 2 not in first and len(set(first)) == 3


def test_epidemic_sampling_caps_at_reachable_peer_count():
    import random

    targets = sample_epidemic_targets(0, {0, 1}, random.Random(7), 8)
    assert targets == [1]


def test_epidemic_sample_size_rejects_invalid_values():
    import pytest

    assert resolve_epidemic_sample_size(5, 0) == 3
    assert resolve_epidemic_sample_size(5, 2) == 2
    with pytest.raises(ValueError, match="non-negative"):
        resolve_epidemic_sample_size(5, -1)
    with pytest.raises(ValueError, match="num_clients - 1"):
        resolve_epidemic_sample_size(5, 5)


def test_weighted_tree_fusion_matches_sample_weighted_average():
    states = [{"w": torch.tensor([1.0])}, {"w": torch.tensor([4.0])}]
    weighted = [
        leaf_weighted_state(states[0], 1),
        leaf_weighted_state(states[1], 3),
    ]
    fused = finalize_weighted_state(merge_weighted_states(weighted))
    assert torch.allclose(fused["w"], torch.tensor([3.25]))


class _TreeRuntime:
    def __init__(self, deliveries):
        self.deliveries = iter(deliveries)
        self.events = []

    def _record_comm(self, *args, **kwargs):
        self.events.append((args, kwargs))
        return next(self.deliveries)


def test_failed_tree_upload_excludes_entire_child_subtree():
    runtime = _TreeRuntime([True, False])  # 2->1 succeeds, 1->0 fails
    states = {
        0: {"w": torch.tensor([0.0])},
        1: {"w": torch.tensor([2.0])},
        2: {"w": torch.tensor([4.0])},
    }
    result, included = aggregate_tree(
        0, {0: [1], 1: [2], 2: []}, states, {0: 1, 1: 1, 2: 1},
        runtime, 0, 4,
    )
    assert result.sample_count == 1 and included == {0}
    assert torch.allclose(finalize_weighted_state(result)["w"], torch.tensor([0.0]))


def test_nested_tree_preserves_the_same_float64_weighted_sum_as_flat_fedavg():
    runtime = _TreeRuntime([True, True])
    states = {
        0: {"w": torch.tensor([1.0])},
        1: {"w": torch.tensor([3.0])},
        2: {"w": torch.tensor([9.0])},
    }
    weighted, included = aggregate_tree(
        0, {0: [1], 1: [2], 2: []}, states, {0: 2, 1: 3, 2: 5},
        runtime, 0, 4,
    )
    assert included == {0, 1, 2}
    expected = finalize_weighted_state(merge_weighted_states([
        leaf_weighted_state(states[0], 2),
        leaf_weighted_state(states[1], 3),
        leaf_weighted_state(states[2], 5),
    ]))
    assert torch.equal(finalize_weighted_state(weighted)["w"], expected["w"])
