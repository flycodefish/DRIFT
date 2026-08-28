"""Unit tests for the Locality-Aware Client Placement module.

These are synthetic, fast, and do not touch the training loop, DAAC clustering,
test data, or any experiment artifacts.
"""

import math
import unittest

import numpy as np

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deasfl.placement import (  # noqa: E402
    apply_permutation,
    build_edges,
    build_non_edges,
    compute_placement,
    edge_jsd_values,
    identity_permutation,
    jsd,
    locality_permutation,
    mean,
    nonedge_jsd_values,
    permutation_is_bijection,
    random_permutation,
)

from deasfl.config import NewFLConfig  # noqa: E402


def _dists(num=8, num_classes=4, seed=7):
    rng = np.random.RandomState(seed)
    return [rng.dirichlet(np.ones(num_classes)) for _ in range(num)]


class _Cfg:
    def __init__(self, mode="locality", steps=200, temperature=0.05, seed=42):
        self.data_placement_mode = mode
        self.locality_placement_steps = steps
        self.locality_temperature = temperature
        self.locality_seed = None
        self.seed = seed


class TestJsd(unittest.TestCase):
    def test_symmetric_and_bounded(self):
        p = np.array([0.8, 0.1, 0.1])
        q = np.array([0.1, 0.8, 0.1])
        self.assertAlmostEqual(jsd(p, q), jsd(q, p), places=8)
        self.assertGreater(jsd(p, q), 0.0)
        self.assertLessEqual(jsd(p, q), math.log(2) + 1e-9)
        self.assertAlmostEqual(jsd(p, p), 0.0, places=8)


class TestEdges(unittest.TestCase):
    def test_edges_undirected_unique(self):
        topo = [[1], [0, 2], [1]]
        edges = build_edges(topo)
        self.assertEqual(edges, [(0, 1), (1, 2)])
        non = build_non_edges(3, edges)
        self.assertEqual(non, [(0, 2)])


class TestPermutations(unittest.TestCase):
    def test_random_seed_controlled(self):
        self.assertEqual(random_permutation(10, 3), random_permutation(10, 3))
        self.assertNotEqual(random_permutation(10, 3), random_permutation(10, 4))

    def test_identity(self):
        self.assertEqual(identity_permutation(5), [0, 1, 2, 3, 4])

    def test_bijection(self):
        for seed in range(5):
            p = random_permutation(12, seed)
            self.assertTrue(permutation_is_bijection(p, 12))

    def test_apply_permutation(self):
        self.assertEqual(apply_permutation(["a", "b", "c"], [2, 0, 1]), ["c", "a", "b"])


class TestLocality(unittest.TestCase):
    def test_locality_reduces_edge_jsd(self):
        dists = _dists(num=8, seed=11)
        topo = [[1, 2, 3], [0, 4], [0, 5], [0, 6], [1, 7], [2], [3], [4]]
        edges = build_edges(topo)
        init = random_permutation(len(dists), 5)
        base = mean(edge_jsd_values(init, dists, edges))
        perm, best = locality_permutation(dists, edges, steps=500, temperature=0.05,
                                          seed=5, init_perm=init)
        self.assertLessEqual(best, base + 1e-9)

    def test_locality_bounded_steps(self):
        dists = _dists(num=6, seed=3)
        topo = [[1, 2], [0, 3], [0, 4], [1, 5], [2], [3]]
        edges = build_edges(topo)
        perm, _ = locality_permutation(dists, edges, steps=37, temperature=0.0,
                                       seed=9, init_perm=list(range(6)))
        self.assertEqual(len(perm), 6)
        self.assertTrue(permutation_is_bijection(perm, 6))

    def test_nonedge_jsd(self):
        dists = _dists(num=5, seed=2)
        edges = [(0, 1), (1, 2)]
        non = build_non_edges(5, edges)
        p = random_permutation(5, 1)
        self.assertEqual(len(nonedge_jsd_values(p, dists, non)), len(non))
        self.assertEqual(len(edge_jsd_values(p, dists, edges)), 2)


class TestComputePlacement(unittest.TestCase):
    def test_random_mode_has_no_locality(self):
        cfg = _Cfg(mode="random", steps=0)
        dists = _dists(num=8, seed=1)
        topo = [[1, 2], [0, 3], [0, 4], [1, 5], [2, 6], [3, 7], [4], [5]]
        res = compute_placement(cfg, dists, topo, seed=42)
        self.assertIsNone(res["locality"])
        self.assertTrue(res["random"]["permutation"] is not None)

    def test_locality_mode_lowers_edge_jsd(self):
        cfg = _Cfg(mode="locality", steps=300)
        dists = _dists(num=8, seed=1)
        topo = [[1, 2], [0, 3], [0, 4], [1, 5], [2, 6], [3, 7], [4], [5]]
        res = compute_placement(cfg, dists, topo, seed=42)
        rand_mean = mean(res["random"]["edge_jsd"])
        loc_mean = mean(res["locality"]["edge_jsd"])
        self.assertLessEqual(loc_mean, rand_mean + 1e-9)


class TestConfigBackwardCompat(unittest.TestCase):
    def test_defaults(self):
        cfg = NewFLConfig()
        self.assertEqual(cfg.data_placement_mode, "random")
        self.assertEqual(cfg.locality_placement_steps, 2000)
        self.assertIsNone(cfg.locality_seed)

    def test_from_yaml_legacy_config_ok(self):
        cfg = NewFLConfig.from_yaml("configs/cifar10/config_de_asfl.yaml")
        self.assertEqual(cfg.num_clients, 30)
        self.assertEqual(cfg.data_placement_mode, "random")


if __name__ == "__main__":
    unittest.main()
