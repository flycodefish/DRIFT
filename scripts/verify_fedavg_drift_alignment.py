"""Deterministic regression checks for FedAvg/DRIFT alignment."""

from copy import deepcopy

import torch

from deasfl.runtime import (
    ModelForwardMessage,
    ModelWrapper,
    _calc_subtree_clients,
    aggregate_models_at_best_client,
    hierarchical_weighted_reduce,
    set_tree_sample_counts,
)
from deasfl.aggregation import sample_weighted_average
from fl_core.data_loader import CNNModel, load_cifar10_data


def build_fixed_forest(sample_counts):
    parents = {0: None, 1: 0, 2: 1, 3: None, 4: 3}
    heads = [0, 3]
    tree = {}
    for node_id, parent_id in parents.items():
        children = [child for child, parent in parents.items() if parent == node_id]
        tree[node_id] = {
            'parent': parent_id,
            'children': children,
            'cluster_id': 0 if node_id < 3 else 3,
            'total_clients': 1,
            'sample_count': 0,
            'total_samples': 0,
            'is_leaf': not children,
        }
    for head in heads:
        _calc_subtree_clients(tree, head)
    set_tree_sample_counts(tree, sample_counts, heads)
    return tree, heads


def max_difference(left, right):
    return max((left[key].float() - right[key].float()).abs().max().item()
               for key in left)


def verify_partition():
    kwargs = dict(
        data_dir='./data', num_clients=5, partition_mode='dirichlet',
        alpha=0.5, seed=42,
    )
    first = load_cifar10_data(**kwargs)
    second = load_cifar10_data(**kwargs)
    indices_a, indices_b = first[2], second[2]
    assert indices_a == indices_b
    assert sum(map(len, indices_a)) == 50_000
    assert len(first[3]['labels']) == 10_000
    print('partition_alignment=PASSED (50000 train, 10000 test)')


def verify_initialization():
    torch.manual_seed(42)
    fedavg = CNNModel(num_classes=10).state_dict()
    torch.manual_seed(42)
    drift = CNNModel(num_classes=10).state_dict()
    difference = max_difference(fedavg, drift)
    assert difference == 0.0
    print(f'initialization_max_difference={difference:.10g}')


def verify_aggregation():
    sample_counts = [71, 113, 197, 89, 241]
    states = []
    for client_id in range(5):
        generator = torch.Generator().manual_seed(100 + client_id)
        states.append({
            'weight': torch.randn(17, 11, generator=generator),
            'bias': torch.randn(17, generator=generator),
        })

    tree, heads = build_fixed_forest(sample_counts)
    models = [ModelWrapper(deepcopy(state)) for state in states]
    cluster_states = hierarchical_weighted_reduce(models, tree, heads)
    messages = [
        ModelForwardMessage(
            source_id=head,
            weighted_state=cluster_states[head],
            path_stack=[head, 0],
        )
        for head in heads
    ]
    drift, _ = aggregate_models_at_best_client(
        best_client_id=0,
        models=models,
        message_queue={0: messages},
        device=torch.device('cpu'),
    )
    fedavg = sample_weighted_average(states, sample_counts)
    difference = max_difference(fedavg, drift)
    print(f'aggregation_max_difference={difference:.10g}')
    if difference > 1e-6:
        raise SystemExit('FAILED: DRIFT hierarchy diverged from FedAvg')
    print('aggregation_alignment=PASSED')


if __name__ == '__main__':
    verify_partition()
    verify_initialization()
    verify_aggregation()
