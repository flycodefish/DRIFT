"""Shared sample-weighted aggregation primitives."""

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


@dataclass
class WeightedState:
    """A float64 weighted sum whose normalization is deliberately delayed."""

    weighted_sum: dict[str, torch.Tensor]
    sample_count: int


def leaf_weighted_state(state: Mapping[str, torch.Tensor], sample_count: int,
                        device: torch.device | None = None) -> WeightedState:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    target = device or torch.device('cpu')
    return WeightedState(
        weighted_sum={
            key: value.to(target).double() * float(sample_count)
            for key, value in state.items()
        },
        sample_count=int(sample_count),
    )


def merge_weighted_states(states: Sequence[WeightedState]) -> WeightedState:
    if not states:
        raise ValueError("states must not be empty")
    merged = {
        key: torch.zeros_like(value)
        for key, value in states[0].weighted_sum.items()
    }
    for state in states:
        for key, value in state.weighted_sum.items():
            merged[key].add_(value)
    return WeightedState(merged, sum(state.sample_count for state in states))


def finalize_weighted_state(state: WeightedState) -> dict[str, torch.Tensor]:
    if state.sample_count <= 0:
        raise ValueError("cannot normalize a zero-sample state")
    return {
        key: (value / state.sample_count).float()
        for key, value in state.weighted_sum.items()
    }


def sample_weighted_average(states: Sequence[Mapping[str, torch.Tensor]],
                            sample_counts: Sequence[int]) -> dict[str, torch.Tensor]:
    weighted = [
        leaf_weighted_state(state, count)
        for state, count in zip(states, sample_counts)
    ]
    return finalize_weighted_state(merge_weighted_states(weighted))
