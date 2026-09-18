"""Contract tests for the official-split HAR input path."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from fl_core.data_loader import HARMLP, MergeSingletonBatchSampler, load_har_data


HAR_DATA_DIR = os.environ.get("HAR_DATA_DIR", "data")
HAR_ROOT = Path(HAR_DATA_DIR) / "har/UCI HAR Dataset"


HAR_DATA_MISSING = pytest.mark.skipif(
    not HAR_ROOT.is_dir(), reason="UCI HAR data is not installed"
)


def _kwargs():
    return {
        "data_dir": HAR_DATA_DIR,
        "num_clients": 5,
        "partition_mode": "dirichlet",
        "alpha": 1.0,
        "seed": 42,
        "split_policy": "official_subject_split",
    }


@HAR_DATA_MISSING
def test_har_partition_is_deterministic_and_reusable():
    first = load_har_data(**_kwargs())
    second = load_har_data(**_kwargs())
    assert first[2] == second[2]

    frozen = load_har_data(**_kwargs(), client_indices=first[2])
    assert frozen[2] == first[2]
    for left, right in zip(first[0], frozen[0]):
        assert torch.equal(left["images"], right["images"])
        assert torch.equal(left["labels"], right["labels"])


@HAR_DATA_MISSING
def test_har_uses_official_test_as_global_evaluation_only():
    result = load_har_data(**_kwargs())
    assert sum(len(client["labels"]) for client in result[0]) == 7352
    assert sum(len(client["labels"]) for client in result[1]) == 0
    assert sorted(index for shard in result[2] for index in shard) == list(range(7352))
    assert len(result[3]["labels"]) == 2947
    raw_train = np.loadtxt(HAR_ROOT / "train/X_train.txt")
    raw_test = np.loadtxt(HAR_ROOT / "test/X_test.txt")
    train_mean = torch.tensor(raw_train, dtype=torch.float32).mean(dim=0)
    train_std = torch.tensor(raw_train, dtype=torch.float32).std(dim=0)
    train_std[train_std < 1e-8] = 1.0
    expected_test = (torch.tensor(raw_test, dtype=torch.float32) - train_mean) / train_std
    assert torch.equal(result[3]["images"][0], expected_test[0])
    assert torch.equal(result[3]["images"][-1], expected_test[-1])


@HAR_DATA_MISSING
def test_har_local_validation_seed_does_not_change_new_partition():
    first = load_har_data(**_kwargs(), local_val_seed=1)
    second = load_har_data(**_kwargs(), local_val_seed=999)
    assert first[2] == second[2]
    assert sum(len(client["labels"]) for client in first[1]) == 0


@HAR_DATA_MISSING
def test_har_can_replay_legacy_post_holdout_frozen_indices():
    legacy_count = 6616
    shards = [list(range(start, legacy_count if client == 4 else start + legacy_count // 5))
              for client, start in enumerate(range(0, 5 * (legacy_count // 5), legacy_count // 5))]
    result = load_har_data(**_kwargs(), client_indices=shards)
    assert sum(len(client["labels"]) for client in result[0]) == legacy_count
    assert sum(len(client["labels"]) for client in result[1]) == 736


@HAR_DATA_MISSING
def test_har_rejects_unknown_partition_mode():
    kwargs = _kwargs()
    kwargs["partition_mode"] = "unknown"
    with pytest.raises(ValueError, match="Unsupported HAR partition mode"):
        load_har_data(**kwargs)


def test_har_model_contract():
    model = HARMLP(input_dim=561, num_classes=6)
    output = model(torch.randn(4, 561))
    assert output.shape == (4, 6)


def test_singleton_safe_sampler_keeps_all_samples():
    sampler = MergeSingletonBatchSampler(list(range(65)), batch_size=32)
    batches = list(iter(sampler))
    assert sorted(index for batch in batches for index in batch) == list(range(65))
    assert all(len(batch) != 1 for batch in batches)
    assert len(sampler) == len(batches) == 2


def test_singleton_safe_sampler_rejects_unsafe_batch_settings():
    with pytest.raises(ValueError, match="batch_size >= 2"):
        MergeSingletonBatchSampler(list(range(4)), batch_size=1)
    with pytest.raises(ValueError, match="at least two samples"):
        MergeSingletonBatchSampler([0], batch_size=2)
