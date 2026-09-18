from __future__ import annotations

from deasfl.cluster_assignments import (
    _assignment_objective,
    build_clustering_result,
    cluster_sizes_from_result,
    matched_groups,
    pairwise_jsd_matrix,
)
from deasfl.runtime import build_client_tree_mapping


LABELS = [
    [0.95, 0.05], [0.90, 0.10], [0.85, 0.15],
    [0.10, 0.90], [0.05, 0.95], [0.15, 0.85],
]


def test_matched_assignments_preserve_capacities_and_are_deterministic() -> None:
    sizes = [3, 2, 1]
    for mode in (
        "random_matched", "oracle_data_similar",
        "jsd_q00", "jsd_q25", "jsd_q50", "jsd_q75", "jsd_q100",
    ):
        first = matched_groups(LABELS, sizes, mode=mode, seed=42, oracle_restarts=4)
        second = matched_groups(LABELS, sizes, mode=mode, seed=42, oracle_restarts=4)
        assert first == second
        assert [len(group) for group in first] == sizes
        assert sorted(node for group in first for node in group) == list(range(6))


def test_oracle_reduces_train_label_jsd_without_changing_sizes() -> None:
    sizes = [3, 3]
    distances = pairwise_jsd_matrix(LABELS)
    random_groups = matched_groups(LABELS, sizes, mode="random_matched", seed=1)
    oracle_groups = matched_groups(
        LABELS, sizes, mode="oracle_data_similar", seed=1, oracle_restarts=8
    )
    assert [len(group) for group in oracle_groups] == [3, 3]
    assert _assignment_objective(oracle_groups, distances) < _assignment_objective(
        random_groups, distances
    )


def test_jsd_quantile_modes_span_ordered_achievable_range() -> None:
    sizes = [3, 3]
    distances = pairwise_jsd_matrix(LABELS)
    scores = []
    for mode in ("jsd_q00", "jsd_q25", "jsd_q50", "jsd_q75", "jsd_q100"):
        groups = matched_groups(LABELS, sizes, mode=mode, seed=7, oracle_restarts=8)
        scores.append(_assignment_objective(groups, distances))
        assert [len(group) for group in groups] == sizes
    assert scores == sorted(scores)
    assert scores[0] < scores[-1]


def test_result_schema_covers_clients_and_respects_max_child() -> None:
    election = {
        "all_nodes_info": [
            {"my_score": float(node_id)} for node_id in range(6)
        ]
    }
    result = build_clustering_result(
        [[0, 1, 2, 3], [4, 5]], election, max_child=2,
        mode="oracle_data_similar",
    )
    tree = build_client_tree_mapping(result, 6)
    assert result["experimental_only"] is True
    assert sorted(result["cluster_heads"]) == [3, 5]
    assert set(tree) == set(range(6))
    assert all(len(info["children"]) <= 2 for info in tree.values())
    assert sorted(tree[head]["total_clients"] for head in result["cluster_heads"]) == [2, 4]


def test_reference_size_profile_preserves_uncovered_clients_as_singletons() -> None:
    reference = {
        "cluster_heads": [0, 3],
        "clusters": {
            0: {"members": [1, 2]},
            3: {"members": [4]},
        },
        "all_nodes_info": [{"client_id": node_id} for node_id in range(6)],
    }

    assert cluster_sizes_from_result(reference) == [3, 2, 1]


def test_reference_size_profile_rejects_overlapping_clusters() -> None:
    reference = {
        "cluster_heads": [0, 2],
        "clusters": {
            0: {"members": [1]},
            2: {"members": [1, 3]},
        },
        "all_nodes_info": [{"client_id": node_id} for node_id in range(4)],
    }

    try:
        cluster_sizes_from_result(reference)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("overlapping reference clusters must be rejected")
