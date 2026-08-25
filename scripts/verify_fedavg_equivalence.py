"""Verify that original DRIFT hierarchical averaging equals FedAvg.

This is a deterministic numerical regression check. It intentionally contains
no experimental aggregation mechanism.
"""

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deasfl.runtime import (  # noqa: E402
    _calc_subtree_clients,
    flat_aggregate,
    hierarchical_aggregate,
    set_tree_sample_counts,
)
from fl_core.data_loader import CNNModel, load_cifar10_data  # noqa: E402


def build_fixed_forest(sample_counts):
    parents = {0: None, 1: 0, 2: 1, 3: None, 4: 3}
    heads = [0, 3]
    tree = {}
    for node_id, parent_id in parents.items():
        children = [child for child, parent in parents.items()
                    if parent == node_id]
        head = 0 if node_id in (0, 1, 2) else 3
        tree[node_id] = {
            'parent': parent_id,
            'children': children,
            'cluster_id': head,
            'total_clients': 1,
            'sample_count': 0,
            'total_samples': 0,
            'is_leaf': not children,
        }
    for head in heads:
        _calc_subtree_clients(tree, head)
    set_tree_sample_counts(tree, sample_counts, heads)
    return tree, heads


def train_local(model, images, labels, seed, batch_size):
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for images_batch, labels_batch in loader:
        optimizer.zero_grad()
        loss = criterion(model(images_batch), labels_batch)
        loss.backward()
        optimizer.step()


def accuracy(params, images, labels, batch_size):
    model = CNNModel(num_classes=10)
    model.load_state_dict(params)
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(labels), batch_size):
            logits = model(images[start:start + batch_size])
            correct += (logits.argmax(dim=1) == labels[start:start + batch_size]).sum().item()
    return correct / len(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='./data')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--samples-per-client', type=int, default=128)
    parser.add_argument('--test-samples', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    train_data, _, _, global_test, _ = load_cifar10_data(
        data_dir=args.data_dir,
        num_clients=5,
        partition_mode='dirichlet',
        alpha=0.5,
    )
    base_model = CNNModel(num_classes=10)
    base_state = copy.deepcopy(base_model.state_dict())
    models = []
    sample_counts = []
    for client_id, shard in enumerate(train_data):
        count = min(args.samples_per_client, len(shard['labels']))
        model = CNNModel(num_classes=10)
        model.load_state_dict(base_state)
        train_local(
            model,
            shard['images'][:count],
            shard['labels'][:count],
            seed=args.seed + client_id,
            batch_size=args.batch_size,
        )
        models.append(model)
        sample_counts.append(count)

    tree, heads = build_fixed_forest(sample_counts)
    subtrees = hierarchical_aggregate(
        models, tree, heads, torch.device('cpu')
    )
    drift_params = flat_aggregate(
        [subtrees[head] for head in heads],
        weights=[tree[head]['total_samples'] for head in heads],
        device=torch.device('cpu'),
    )
    fedavg_params = flat_aggregate(
        [model.state_dict() for model in models],
        weights=sample_counts,
        device=torch.device('cpu'),
    )

    max_difference = max(
        (drift_params[key].float() - fedavg_params[key].float()).abs().max().item()
        for key in drift_params
    )
    test_images = global_test['images'][:args.test_samples]
    test_labels = global_test['labels'][:args.test_samples]
    drift_accuracy = accuracy(drift_params, test_images, test_labels, args.batch_size)
    fedavg_accuracy = accuracy(fedavg_params, test_images, test_labels, args.batch_size)

    print(f'max_parameter_difference={max_difference:.10g}')
    print(f'drift_accuracy={drift_accuracy:.6f}')
    print(f'fedavg_accuracy={fedavg_accuracy:.6f}')
    if max_difference > 1e-6 or drift_accuracy != fedavg_accuracy:
        raise SystemExit('FAILED: hierarchical aggregation diverged from FedAvg')
    print('PASSED: original DRIFT hierarchy is numerically equivalent to FedAvg')


if __name__ == '__main__':
    main()
