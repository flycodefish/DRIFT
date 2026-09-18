import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

from deasfl.runtime import (
    CommStats,
    _calc_subtree_clients,
    distribute_models_to_cluster,
    flat_aggregate,
    hierarchical_aggregate,
    set_tree_sample_counts,
    weighted_validation_accuracy,
)
from fl_core.actors import CNN
from fl_core.data_loader import CNNModel, load_cifar10_data


class ScalarModel(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([value], dtype=torch.float32))


def make_forest(parents, heads):
    tree = {}
    for node_id, parent_id in parents.items():
        children = [child for child, parent in parents.items()
                    if parent == node_id]
        cluster_id = next(
            head for head in heads
            if node_id == head or _belongs_to(node_id, head, parents)
        )
        tree[node_id] = {
            'parent': parent_id,
            'children': children,
            'cluster_id': cluster_id,
            'total_clients': 1,
            'sample_count': 0,
            'total_samples': 0,
            'is_leaf': not children,
        }
    for head in heads:
        _calc_subtree_clients(tree, head)
    return tree


def _belongs_to(node_id, head, parents):
    current = node_id
    while parents[current] is not None:
        current = parents[current]
    return current == head


class FedAvgEquivalenceTest(unittest.TestCase):
    def setUp(self):
        self.models = [ScalarModel(value) for value in (1, 2, 4, 8, 16, 32)]
        self.samples = [3, 5, 7, 11, 13, 17]
        self.device = torch.device('cpu')

    def test_two_level_hierarchy_equals_sample_weighted_fedavg(self):
        parents = {0: None, 1: 0, 2: 1, 3: None, 4: 3, 5: 3}
        heads = [0, 3]
        tree = make_forest(parents, heads)
        set_tree_sample_counts(tree, self.samples, heads)

        subtree_models = hierarchical_aggregate(
            self.models, tree, heads, self.device
        )
        hierarchical = flat_aggregate(
            [subtree_models[head] for head in heads],
            weights=[tree[head]['total_samples'] for head in heads],
            device=self.device,
        )
        fedavg = flat_aggregate(
            [model.state_dict() for model in self.models],
            weights=self.samples,
            device=self.device,
        )

        self.assertTrue(torch.allclose(
            hierarchical['weight'], fedavg['weight'], atol=1e-7, rtol=1e-7
        ))

    def test_tree_shape_does_not_change_global_result(self):
        forests = [
            ({0: None, 1: 0, 2: 1, 3: None, 4: 3, 5: 3}, [0, 3]),
            ({0: 2, 1: 2, 2: None, 3: 5, 4: 5, 5: None}, [2, 5]),
        ]
        outputs = []
        for parents, heads in forests:
            tree = make_forest(parents, heads)
            set_tree_sample_counts(tree, self.samples, heads)
            subtree_models = hierarchical_aggregate(
                self.models, tree, heads, self.device
            )
            outputs.append(flat_aggregate(
                [subtree_models[head] for head in heads],
                weights=[tree[head]['total_samples'] for head in heads],
                device=self.device,
            )['weight'])

        self.assertTrue(torch.allclose(outputs[0], outputs[1], atol=1e-7, rtol=1e-7))

    def test_global_model_is_broadcast_to_every_tree_node(self):
        parents = {0: None, 1: 0, 2: 1, 3: 0}
        tree = make_forest(parents, [0])
        models = [ScalarModel(value) for value in (1, 2, 3, 4)]
        global_params = ScalarModel(99).state_dict()
        comm = CommStats()

        distribute_models_to_cluster(
            aggregated_models={0: global_params},
            tree_mapping=tree,
            models=models,
            cluster_heads=[0],
            device=self.device,
            comm=comm,
        )

        for model in models:
            self.assertTrue(torch.equal(
                model.state_dict()['weight'], global_params['weight']
            ))
        self.assertEqual(comm.intra_cluster_dist_msgs, 3)

    def test_validation_accuracy_uses_sample_weights(self):
        results = [
            {'val_acc': 0.5, 'num_samples': 10},
            {'val_acc': 1.0, 'num_samples': 30},
        ]
        self.assertAlmostEqual(weighted_validation_accuracy(results), 0.875)

    def test_fedavg_and_drift_use_the_same_cnn(self):
        fedavg = CNN(num_classes=10)
        drift = CNNModel(num_classes=10)
        self.assertEqual(list(fedavg.state_dict()), list(drift.state_dict()))
        for key in fedavg.state_dict():
            self.assertEqual(fedavg.state_dict()[key].shape,
                             drift.state_dict()[key].shape)

    @mock.patch('torchvision.datasets.CIFAR10')
    def test_cifar_official_test_split_never_enters_training(self, dataset_cls):
        train = mock.Mock()
        train.data = np.zeros((200, 32, 32, 3), dtype=np.uint8)
        train.targets = [0] * 200
        test = mock.Mock()
        test.data = np.ones((100, 32, 32, 3), dtype=np.uint8)
        test.targets = [1] * 100
        dataset_cls.side_effect = [train, test]

        train_data, val_data, _, global_test, _ = load_cifar10_data(
            data_dir='unused',
            num_clients=2,
            partition_mode='iid',
        )

        self.assertEqual(
            set(torch.cat([item['labels'] for item in train_data]).tolist()),
            {0},
        )
        self.assertEqual(
            set(torch.cat([item['labels'] for item in val_data]).tolist()),
            {1},
        )
        self.assertEqual(set(global_test['labels'].tolist()), {1})


if __name__ == '__main__':
    unittest.main()
