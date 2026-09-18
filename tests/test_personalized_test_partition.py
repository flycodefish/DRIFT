from __future__ import annotations

import torch

from fl_core.data_loader import build_distribution_matched_test_indices


def test_distribution_matched_test_partition_is_complete_and_deterministic() -> None:
    train_labels = torch.tensor([0, 0, 0, 1, 1, 0, 1, 1, 1, 1])
    clients = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    test_labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    first = build_distribution_matched_test_indices(
        train_labels, clients, test_labels, num_classes=2, seed=42
    )
    second = build_distribution_matched_test_indices(
        train_labels, clients, test_labels, num_classes=2, seed=42
    )
    assert first == second
    flattened = [index for shard in first for index in shard]
    assert sorted(flattened) == list(range(len(test_labels)))
    assert len(flattened) == len(set(flattened))


def test_absent_train_class_never_appears_in_client_test_partition() -> None:
    train_labels = torch.tensor([0, 0, 0, 1, 1, 1])
    clients = [[0, 1, 2], [3, 4, 5]]
    test_labels = torch.tensor([0, 0, 1, 1])
    result = build_distribution_matched_test_indices(
        train_labels, clients, test_labels, num_classes=2, seed=7
    )
    client_zero_test_labels = test_labels[torch.tensor(result[0])]
    client_one_test_labels = test_labels[torch.tensor(result[1])]
    assert set(client_zero_test_labels.tolist()) == {0}
    assert set(client_one_test_labels.tolist()) == {1}
