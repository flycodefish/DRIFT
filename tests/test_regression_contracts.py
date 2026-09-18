from __future__ import annotations

import json
import hashlib
import math
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
import deasfl.runtime as deasfl_runtime_module

from deasfl.config import NewFLConfig
from deasfl.runtime import (
    CommStats, ModelWrapper, NewFLRuntime, aggregate_models_at_best_client,
    distribute_returned_models_to_clusters, forward_model_to_best_client,
    hierarchical_weighted_reduce, leaf_weighted_state, return_model_to_cluster_heads,
)
from deasfl.election import (
    ClusterMessage, ClusterMessageType, ClusterState, DistributedClustering,
    ElectionManager,
)
from fl_core.actors import MLP, get_model_params
from fl_core.artifacts import (
    generate_node_resources, load_frozen_inputs, normalize_physical_graph,
    validate_frozen_inputs, write_completion_status, write_experiment_inputs,
)
from fl_core.config import FLConfig
from fl_core.runtime import RayFLRuntime
from fl_core.system_metrics import build_metrics_summary, build_node_round_stats
from scripts.calculate_offline_metrics import (
    _canonical_config_value, _load_simulation, _validate_calibration_artifacts,
    node_failure_rounds, recovery_metrics,
    round_latency_metrics, target_metrics, topology_recovery_metrics,
    validate_comparable_runs,
)
from scripts.verify_10round_prefix import compare_prefixes
from fl_core.simulation import simulate_round, transfer_latency_ms


class RegressionContractTests(unittest.TestCase):
    @staticmethod
    def _write_comparable_run(root: Path, config: dict) -> None:
        config_dir = root / "config"
        config_dir.mkdir(parents=True)
        (config_dir / "experiment.json").write_text(json.dumps(config), encoding="utf-8")
        (config_dir / "data_partition.json").write_text(
            json.dumps({"dataset": config.get("dataset", "cifar10"),
                        "client_indices": [[0, 1]], "sample_counts": [2]}),
            encoding="utf-8",
        )
        for name in (
            "initial_model.pt", "physical_graph.json", "node_resources.json",
            "link_profiles.json", "failure_trace.json",
        ):
            (config_dir / name).write_bytes(b"same")
        write_completion_status(str(root), 1, 1, "COMPLETE")
        (root / "test_round_metrics.csv").write_text(
            "round_id,val_acc\n0,0.5\n", encoding="utf-8"
        )

    def test_node_failure_round_matching_is_case_insensitive(self):
        topology = [
            {"round": "0", "event_type": "node_failure"},
            {"round": "2", "event_type": " NODE_FAILURE "},
            {"round": "3", "event_type": "link_failure"},
            {"round": "", "event_type": "node_failure"},
        ]
        self.assertEqual(node_failure_rounds(topology), [0, 2])

    def test_cluster_diagnostics_are_disabled_by_default(self):
        self.assertFalse(NewFLConfig().enable_cluster_diagnostics)
        self.assertEqual(NewFLConfig().intra_cluster_rounds, 1)
        self.assertEqual(NewFLConfig().clustering_assignment_mode, "distributed")
        self.assertEqual(NewFLConfig().client_test_partition_mode, "global_balanced")
        self.assertFalse(NewFLConfig().enable_cluster_personalized_evaluation)

    def test_cluster_jsd_uses_natural_log_and_pairwise_mean(self):
        runtime = NewFLRuntime(NewFLConfig(use_gpu=False, num_clients=3))
        runtime.label_distributions = [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
        self.assertAlmostEqual(runtime._mean_pairwise_label_jsd([0, 1]), 0.0)
        self.assertAlmostEqual(
            runtime._mean_pairwise_label_jsd([0, 2]), math.log(2.0), places=6
        )

    def test_cluster_diagnostics_evaluate_intermediate_model_without_actor_mutation(self):
        config = NewFLConfig(
            use_gpu=False, num_clients=2, model="harmlp", input_dim=2,
            num_classes=2, batch_size=2, enable_cluster_diagnostics=True,
            cluster_diagnostics_global_test=True, rounds=30,
            cluster_diagnostics_interval=5,
        )
        runtime = NewFLRuntime(config)
        images = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        labels = torch.tensor([0, 1])
        runtime.clients = [
            {"val_data": {"images": images[:1], "labels": labels[:1]}},
            {"val_data": {"images": images[1:], "labels": labels[1:]}},
        ]
        runtime.global_test_data = {"images": images, "labels": labels}
        runtime.label_distributions = [[1.0, 0.0], [0.0, 1.0]]
        runtime.tree_mapping = {
            0: {"children": [1]},
            1: {"children": []},
        }
        model = runtime._create_model()
        state = leaf_weighted_state(model.state_dict(), sample_count=2)
        self.assertEqual(runtime._evaluate_cluster_diagnostics(1, [0], {0: state}), [])
        rows = runtime._evaluate_cluster_diagnostics(29, [0], {0: state})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["num_members"], 2)
        self.assertEqual(rows[0]["validation_samples"], 2)
        self.assertEqual(rows[0]["label_pair_count"], 1)
        self.assertAlmostEqual(rows[0]["mean_pairwise_label_jsd"], math.log(2.0), places=6)
        self.assertIsNotNone(rows[0]["cluster_global_accuracy"])

    def test_jsd_constraint_cannot_be_bypassed_by_simultaneous_trigger(self):
        receiver = DistributedClustering(
            client_id=1, neighbors=[0], resource_score=1.0,
            election_trigger_time=1.0, distribution_similarity_threshold=0.2,
            label_distribution=[0.0, 1.0],
        )
        receiver.state = ClusterState.TRIGGERED
        message = ClusterMessage(
            msg_type=ClusterMessageType.AGG_REQ,
            sender_id=0, receiver_id=1, child_count=0, cluster_id=None,
            request_seq=1, sender_trigger_time=receiver.cluster_trigger_time,
            sender_label_distribution=[1.0, 0.0],
        )
        response = receiver.receive_agg_req(message)
        self.assertEqual(response.msg_type, ClusterMessageType.REJECT)
        self.assertIsNone(receiver.parent)

    def test_offline_target_counts_zero_based_round_as_one_round(self):
        result = target_metrics([0.2, 0.6, 0.7], 0.6)
        self.assertEqual(result["rounds_to_target"], 2)
        self.assertEqual(result["target_status"], "reached")

    def test_round_latency_summary_uses_linear_p95(self):
        result = round_latency_metrics([
            {"round_latency_sim_ms": value} for value in (10, 20, 30, 40, 50)
        ])
        self.assertEqual(result["mean_round_latency_sim_ms"], 30.0)
        self.assertEqual(result["p95_round_latency_sim_ms"], 48.0)

    def test_offline_recovery_uses_gap_and_patience(self):
        result = recovery_metrics(
            [0.80, 0.70, 0.795, 0.798, 0.80],
            [0.80, 0.80, 0.80, 0.80, 0.80],
            [{"round": 1, "event_type": "NODE_FAILURE"}],
            gap_pp=1.0,
            patience=2,
        )
        self.assertEqual(result["failure_rounds"], [1])
        self.assertAlmostEqual(result["accuracy_drop_pp"], 10.0)
        self.assertEqual(result["max_drop_round"], 1)
        self.assertEqual(result["accuracy_recovery_round"], 3)
        self.assertEqual(result["accuracy_recovery_rounds"], 2)
        self.assertEqual(result["accuracy_recovery_status"], "recovered")

    def test_fedavg_and_drift_partition_aliases_are_comparable(self):
        fedavg = {"data_partition_mode": "dirichlet", "dirichlet_alpha": 0.1,
                  "classes_per_client": 2}
        drift = {"alpha": 0.1, "classes_per_client": None}
        self.assertEqual(_canonical_config_value(fedavg, "data_partition_mode"),
                         _canonical_config_value(drift, "data_partition_mode"))
        self.assertEqual(_canonical_config_value(fedavg, "dirichlet_alpha"),
                         _canonical_config_value(drift, "dirichlet_alpha"))
        self.assertIsNone(_canonical_config_value(fedavg, "classes_per_client"))
        self.assertEqual(
            _canonical_config_value({"partition": "iid", "alpha": 1.0}, "data_partition_mode"),
            "iid",
        )

    def test_calibration_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact = Path(temp_dir) / "benchmark.json"
            artifact.write_text("{}", encoding="utf-8")
            assumptions = {
                "calibration_artifact": str(artifact),
                "calibration_sha256": "0" * 64,
            }
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                _validate_calibration_artifacts(assumptions, Path(temp_dir) / "config.json")

    def test_missing_calibration_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing.json"
            with self.assertRaisesRegex(ValueError, "Missing simulation calibration artifact"):
                _validate_calibration_artifacts(
                    {"calibration_artifact": str(missing),
                     "calibration_sha256": "0" * 64},
                    Path(temp_dir) / "config.json",
                )

    def test_simulation_model_and_batch_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact = root / "benchmark.json"
            artifact.write_text("{}", encoding="utf-8")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            simulation = root / "simulation.json"
            base = {
                "status": "frozen", "reference_model": "harmlp",
                "reference_batch_size": 64,
                "calibration_artifact": str(artifact),
                "calibration_sha256": digest,
            }
            simulation.write_text(json.dumps(base), encoding="utf-8")
            run = root / "run"
            (run / "config").mkdir(parents=True)
            args = SimpleNamespace(simulation_config=simulation, run=run)

            (run / "config" / "experiment.json").write_text(
                json.dumps({"model": "resnet18", "batch_size": 64}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "reference_model"):
                _load_simulation(args, [0.5])

            (run / "config" / "experiment.json").write_text(
                json.dumps({"model": "harmlp", "batch_size": 32}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "reference_batch_size"):
                _load_simulation(args, [0.5])

    def test_comparison_canonicalizes_optimizer_and_participation_fields(self):
        left = {"weight_decay": 1e-4, "client_sample_ratio": 1.0,
                "client_drop_rate": 0.0}
        right = {"weight_decay": 0.0001, "client_sample_ratio": 1,
                 "client_drop_rate": 0}
        for key in left:
            self.assertEqual(_canonical_config_value(left, key),
                             _canonical_config_value(right, key))

    def test_comparison_rejects_different_participation_ratio(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            left, right = Path(temp_dir) / "left", Path(temp_dir) / "right"
            base = {"client_sample_ratio": 1.0, "client_drop_rate": 0.0}
            self._write_comparable_run(left, base)
            self._write_comparable_run(right, {**base, "client_sample_ratio": 0.5})
            with self.assertRaisesRegex(ValueError, "client_sample_ratio"):
                validate_comparable_runs(left, right, "test")

    def test_comparison_rejects_different_client_drop_rate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            left, right = Path(temp_dir) / "left", Path(temp_dir) / "right"
            base = {"client_sample_ratio": 1.0, "client_drop_rate": 0.0}
            self._write_comparable_run(left, base)
            self._write_comparable_run(right, {**base, "client_drop_rate": 0.2})
            with self.assertRaisesRegex(ValueError, "client_drop_rate"):
                validate_comparable_runs(left, right, "test")

    def test_cifar_comparison_ignores_irrelevant_har_seed_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            left, right = Path(temp_dir) / "left", Path(temp_dir) / "right"
            self._write_comparable_run(
                left, {"dataset": "cifar10", "har_local_val_seed": 42}
            )
            self._write_comparable_run(right, {"dataset": "cifar10"})
            left_partition = {"dataset": "cifar10", "client_indices": [[0, 1]],
                              "sample_counts": [2], "local_val_seed": 42}
            right_partition = dict(left_partition)
            right_partition.pop("local_val_seed")
            (left / "config" / "data_partition.json").write_text(
                json.dumps(left_partition), encoding="utf-8"
            )
            (right / "config" / "data_partition.json").write_text(
                json.dumps(right_partition), encoding="utf-8"
            )
            validate_comparable_runs(left, right, "test")

    def test_har_comparison_keeps_local_validation_seed_strict(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            left, right = Path(temp_dir) / "left", Path(temp_dir) / "right"
            self._write_comparable_run(left, {"dataset": "har", "har_local_val_seed": 42})
            self._write_comparable_run(right, {"dataset": "har", "har_local_val_seed": 43})
            with self.assertRaisesRegex(ValueError, "har_local_val_seed"):
                validate_comparable_runs(left, right, "test")

    def test_drift_passes_strict_frozen_input_flag(self):
        cfg = NewFLConfig(
            dataset="synthetic", num_clients=2, use_gpu=False,
            frozen_inputs_dir="unused", strict_frozen_inputs=True,
        )
        bundle = {
            "data_partition.json": {
                "client_indices": [[0, 1], [2, 3]],
                "dataset": "synthetic", "split_policy": "default",
            },
            "node_resources.json": [],
        }
        captured = {}

        def capture(*args, **kwargs):
            captured["strict"] = kwargs["strict"]

        with patch.object(deasfl_runtime_module, "load_frozen_inputs", return_value=bundle), \
             patch.object(deasfl_runtime_module, "validate_frozen_inputs", side_effect=capture), \
             patch.object(deasfl_runtime_module.ray, "init"):
            NewFLRuntime(cfg).start()
        self.assertTrue(captured["strict"])

    def test_recovery_does_not_claim_drop_when_fault_run_is_better(self):
        result = recovery_metrics(
            [0.8, 0.81], [0.8, 0.80],
            [{"round": 1, "event_type": "node_failure"}], 1.0, 1,
        )
        self.assertEqual(result["accuracy_drop_pp"], 0.0)
        self.assertIsNone(result["max_drop_round"])

    def test_topology_recovery_matches_case_insensitive_events(self):
        result = topology_recovery_metrics([
            {"round": 1, "event_type": "LINK_FAILURE", "target_type": "link", "target_id": "0-1"},
            {"round": 3, "event_type": "link_recovery", "target_type": "link", "target_id": "0-1"},
        ])
        self.assertEqual(result["availability_recovery_rounds"], 2)
        self.assertEqual(result["availability_recovery_status"], "recovered")

    def test_topology_recovery_requires_every_failure_to_have_a_distinct_recovery(self):
        result = topology_recovery_metrics([
            {"round": 1, "event_type": "node_failure", "target_type": "node", "target_id": "1"},
            {"round": 2, "event_type": "node_failure", "target_type": "node", "target_id": "1"},
            {"round": 3, "event_type": "node_recovery", "target_type": "node", "target_id": "1"},
        ])
        self.assertEqual(result["availability_recovery_status"], "not_recovered")
        self.assertEqual(result["availability_recovery_unmatched_failures"], 1)

    def test_dynamic_initial_inactive_nodes_are_frozen_without_events(self):
        cfg = FLConfig(
            algorithm="dynamic_dpsgd", num_clients=2, use_gpu=False,
            dynamic_initial_inactive_nodes="[1]",
        )
        runtime = RayFLRuntime(cfg)
        self.assertEqual(runtime._experiment_failure_trace(), {
            "events": [], "initial_inactive_nodes": [1],
        })

    def test_epidemic_dynamic_failure_trace_is_frozen(self):
        cfg = FLConfig(
            algorithm="epidemic_learning", num_clients=2, use_gpu=False,
            dynamic_failure_trace_json=(
                '{"events":[{"round":0,"event_type":"link_failure",'
                '"edges":[[0,1]]}]}'
            ),
        )
        runtime = RayFLRuntime(cfg)
        self.assertEqual(runtime._experiment_failure_trace(), {
            "events": [{
                "round": 0, "event_type": "link_failure", "edges": [[0, 1]],
            }],
        })

    def test_epidemic_stochastic_dropout_is_frozen_without_dynamic_trace(self):
        runtime = RayFLRuntime(FLConfig(
            algorithm="epidemic_learning", num_clients=2, use_gpu=False,
            seed=17, client_drop_rate=0.25,
        ))
        self.assertEqual(runtime._experiment_failure_trace(), {
            "mode": "stochastic_client_dropout",
            "client_drop_rate": 0.25,
            "seed": 17,
        })

    def test_physical_graph_is_normalized_once_as_undirected(self):
        self.assertEqual(
            normalize_physical_graph({"0": [1], "1": [2], "2": []}, 3),
            {0: [1], 1: [0, 2], 2: [1]},
        )

    def test_runtime_loads_frozen_edge_overrides(self):
        runtime = RayFLRuntime(FLConfig(num_clients=2, use_gpu=False))
        runtime.frozen_inputs = {"link_profiles.json": {"edge_overrides": {
            "client_0->client_1": {
                "bandwidth_mbps": 12.5, "latency_ms": 7, "drop_rate": 0.25,
            }
        }}}
        runtime._build_edge_overrides()
        self.assertEqual(runtime.edge_overrides["client_0->client_1"], {
            "bandwidth_mbps": 12.5, "latency_ms": 7.0, "drop_rate": 0.25,
        })

    def test_simulation_retry_attempts_and_backoff(self):
        event = {
            "payload_bytes": 1_000_000, "bandwidth_mbps": 100,
            "latency_ms": 5, "attempt_count": 3, "backoff_ms": 10,
        }
        self.assertAlmostEqual(transfer_latency_ms(
            event,
            {"default_bandwidth_mbps": 100, "default_link_latency_ms": 5},
        ), 265.0)

    def test_simulation_separates_overlapping_protocol_phases(self):
        def event(src, dst, phase):
            return {
                "src": src, "dst": dst, "message_type": "GLOBAL_MODEL",
                "phase": phase, "payload_bytes": 1_000_000,
                "bandwidth_mbps": 100, "latency_ms": 5, "success": True,
            }
        events = [
            event(0, 1, "INTRA_CLUSTER_UPLOAD"),
            event(1, 2, "INTRA_CLUSTER_UPLOAD"),
            event(2, 1, "INTER_CLUSTER_UPLOAD"),
            event(1, 2, "INTER_CLUSTER_DOWNLOAD"),
            event(2, 1, "INTRA_CLUSTER_DOWNLOAD"),
            event(1, 0, "INTRA_CLUSTER_DOWNLOAD"),
        ]
        result = simulate_round(
            events, [], [1, 1, 1], 1,
            {"default_bandwidth_mbps": 100, "default_link_latency_ms": 5,
             "default_compute_capacity": 1, "reference_train_ms_per_sample_epoch": 0.1,
             "aggregation_ms_per_model_mb": 0.05},
        )
        self.assertAlmostEqual(result["intra_cluster_upload_ms"], 170.0)
        self.assertAlmostEqual(result["inter_cluster_upload_ms"], 85.0)
        self.assertAlmostEqual(result["round_latency_sim_ms"], 510.1)

    def test_simulation_applies_shared_interface_capacity(self):
        events = [{
            "src": i, "dst": 2, "message_type": "MODEL_UPLOAD",
            "phase": "MODEL_UPLOAD", "payload_bytes": 1_000_000,
            "bandwidth_mbps": 100, "latency_ms": 5, "success": True,
        } for i in (0, 1)]
        result = simulate_round(
            events, [], [1, 1, 1], 1,
            {"default_bandwidth_mbps": 100, "default_link_latency_ms": 5,
             "default_compute_capacity": 1, "reference_train_ms_per_sample_epoch": 0,
             "aggregation_ms_per_model_mb": 0,
             "network_capacity_by_node": {"2": 50}},
        )
        self.assertAlmostEqual(result["upload_critical_path_ms"], 330.0)

    def test_simulation_serializes_routed_peer_hops_and_receiver_average(self):
        events = [
            {
                "src": "client_0", "dst": "client_1",
                "message_type": "PEER_MODEL", "phase": "PEER_EXCHANGE",
                "flow_id": "peer:0:2", "physical_hop_index": 1,
                "physical_hop_count": 2, "payload_bytes": 1_000_000,
                "bandwidth_mbps": 100, "latency_ms": 5, "success": True,
            },
            {
                "src": "client_1", "dst": "client_2",
                "message_type": "PEER_MODEL", "phase": "PEER_EXCHANGE",
                "flow_id": "peer:0:2", "physical_hop_index": 2,
                "physical_hop_count": 2, "payload_bytes": 1_000_000,
                "bandwidth_mbps": 100, "latency_ms": 5, "success": True,
            },
        ]
        result = simulate_round(
            events, [], [1, 1, 1], 1,
            {"default_bandwidth_mbps": 100, "default_link_latency_ms": 5,
             "default_compute_capacity": 1,
             "reference_train_ms_per_sample_epoch": 0,
             "aggregation_ms_per_model_mb": 2},
            {"aggregation_inputs_by_phase": {
                "PEER_EXCHANGE": {"2": 1}
            }},
        )
        self.assertAlmostEqual(result["peer_exchange_critical_path_ms"], 170.0)
        self.assertAlmostEqual(result["aggregation_critical_path_ms"], 2.0)
        self.assertAlmostEqual(result["round_latency_sim_ms"], 172.0)

    def test_simulation_serializes_peer_fan_in_at_receiver_interface(self):
        events = [{
            "src": f"client_{sender}", "dst": "client_2",
            "message_type": "PEER_MODEL", "phase": "PEER_EXCHANGE",
            "flow_id": f"peer:{sender}:2", "physical_hop_index": 1,
            "physical_hop_count": 1, "payload_bytes": 1_000_000,
            "bandwidth_mbps": 100, "latency_ms": 5, "success": True,
        } for sender in (0, 1)]
        result = simulate_round(
            events, [], [1, 1, 1], 1,
            {"default_bandwidth_mbps": 100, "default_link_latency_ms": 5,
             "default_compute_capacity": 1,
             "reference_train_ms_per_sample_epoch": 0,
             "aggregation_ms_per_model_mb": 2},
            {"aggregation_inputs_by_phase": {
                "PEER_EXCHANGE": {"2": 2}
            }},
        )
        self.assertAlmostEqual(result["peer_exchange_critical_path_ms"], 170.0)
        self.assertAlmostEqual(result["aggregation_critical_path_ms"], 4.0)
        self.assertAlmostEqual(result["round_latency_sim_ms"], 174.0)

    def test_incomplete_routed_flow_is_not_averaged_by_intermediate_node(self):
        events = [
            {
                "src": "client_2", "dst": "client_1",
                "message_type": "PEER_MODEL", "phase": "PEER_EXCHANGE",
                "flow_id": "complete", "physical_hop_index": 1,
                "physical_hop_count": 1, "payload_bytes": 1_000_000,
                "bandwidth_mbps": 100, "latency_ms": 5, "success": True,
            },
            {
                "src": "client_0", "dst": "client_1",
                "message_type": "PEER_MODEL", "phase": "PEER_EXCHANGE",
                "flow_id": "failed-after-hop-one", "physical_hop_index": 1,
                "physical_hop_count": 2, "payload_bytes": 1_000_000,
                "bandwidth_mbps": 100, "latency_ms": 5, "success": True,
            },
        ]
        result = simulate_round(
            events, [], [1, 1, 1], 1,
            {"default_bandwidth_mbps": 100, "default_link_latency_ms": 5,
             "default_compute_capacity": 1,
             "reference_train_ms_per_sample_epoch": 0,
             "aggregation_ms_per_model_mb": 2},
            {"aggregation_inputs_by_phase": {
                "PEER_EXCHANGE": {"1": 1}
            }},
        )
        self.assertAlmostEqual(result["peer_exchange_critical_path_ms"], 170.0)
        self.assertAlmostEqual(result["round_latency_sim_ms"], 170.0)

    def test_simulation_places_each_tree_aggregation_on_critical_path(self):
        events = [{
            "src": src, "dst": dst, "message_type": "AGGREGATED_MODEL",
            "phase": "INTRA_CLUSTER_UPLOAD", "payload_bytes": 1_000_000,
            "bandwidth_mbps": 100, "latency_ms": 5, "success": True,
        } for src, dst in ((0, 1), (1, 2))]
        result = simulate_round(
            events, [], [1, 1, 1], 1,
            {"default_bandwidth_mbps": 100, "default_link_latency_ms": 5,
             "default_compute_capacity": 1,
             "reference_train_ms_per_sample_epoch": 0,
             "aggregation_fixed_ms": 0,
             "aggregation_ms_per_model_mb": 2},
        )
        self.assertAlmostEqual(result["upload_critical_path_ms"], 170.0)
        self.assertAlmostEqual(result["aggregation_critical_path_ms"], 4.0)
        self.assertAlmostEqual(result["round_latency_sim_ms"], 174.0)

    def test_simulation_joins_interface_queue_and_tree_dependency(self):
        events = [{
            "src": src, "dst": dst, "message_type": "AGGREGATED_MODEL",
            "phase": "INTRA_CLUSTER_UPLOAD", "payload_bytes": 1_000_000,
            "bandwidth_mbps": 100, "latency_ms": 5, "success": True,
        } for src, dst in ((0, 1), (2, 1), (1, 3))]
        result = simulate_round(
            events, [], [1, 1, 1, 1], 1,
            {"default_bandwidth_mbps": 100, "default_link_latency_ms": 5,
             "default_compute_capacity": 1,
             "reference_train_ms_per_sample_epoch": 0,
             "aggregation_fixed_ms": 0,
             "aggregation_ms_per_model_mb": 2},
        )
        # Two uploads share node 1's receive interface, then node 1
        # aggregates before its upload to node 3.
        self.assertAlmostEqual(result["upload_critical_path_ms"], 170.0)
        self.assertAlmostEqual(result["round_latency_sim_ms"], 261.0)

    def test_simulation_uses_semantic_context_aggregation_inputs(self):
        event = {
            "src": 0, "dst": 1, "message_type": "AGGREGATED_MODEL",
            "phase": "INTRA_CLUSTER_UPLOAD", "payload_bytes": 1_000_000,
            "bandwidth_mbps": 100, "latency_ms": 5, "success": True,
        }
        result = simulate_round(
            [event], [], [1, 1], 1,
            {"default_bandwidth_mbps": 100, "default_link_latency_ms": 5,
             "default_compute_capacity": 1,
             "reference_train_ms_per_sample_epoch": 0,
             "aggregation_ms_per_model_mb": 2},
            {"aggregation_inputs_by_phase": {
                "INTRA_CLUSTER_UPLOAD": {"1": 2}
            }},
        )
        self.assertAlmostEqual(result["round_latency_sim_ms"], 89.0)

    def test_simulation_matches_client_prefixed_event_ids_to_context_ids(self):
        event = {
            "src": "client_0", "dst": "client_1",
            "message_type": "AGGREGATED_MODEL", "phase": "INTRA_CLUSTER_UPLOAD",
            "payload_bytes": 1_000_000, "bandwidth_mbps": 100,
            "latency_ms": 5, "success": True,
        }
        result = simulate_round(
            [event], [], [1, 1], 1,
            {"default_bandwidth_mbps": 100, "default_link_latency_ms": 5,
             "default_compute_capacity": 1,
             "reference_train_ms_per_sample_epoch": 0,
             "aggregation_ms_per_model_mb": 2,
             "aggregation_capacity_by_node": {"1": 1}},
            {"aggregation_inputs_by_phase": {
                "INTRA_CLUSTER_UPLOAD": {"1": 2}
            }},
        )
        self.assertAlmostEqual(result["aggregation_critical_path_ms"], 4.0)
        self.assertAlmostEqual(result["round_latency_sim_ms"], 89.0)

    def test_simulation_fedavg_star_server_path(self):
        uploads = [{
            "src": node, "dst": "server", "message_type": "MODEL_UPLOAD",
            "phase": "MODEL_UPLOAD", "payload_bytes": 1_000_000,
            "bandwidth_mbps": 100, "latency_ms": 5, "success": True,
        } for node in (0, 1, 2)]
        downloads = [{
            "src": "server", "dst": node, "message_type": "GLOBAL_MODEL",
            "phase": "MODEL_DISTRIBUTION", "payload_bytes": 1_000_000,
            "bandwidth_mbps": 100, "latency_ms": 5, "success": True,
        } for node in (0, 1)]
        result = simulate_round(
            uploads + downloads, [], [1, 1, 1], 1,
            {"default_bandwidth_mbps": 100, "default_link_latency_ms": 5,
             "default_compute_capacity": 1,
             "reference_train_ms_per_sample_epoch": 0,
             "aggregation_ms_per_model_mb": 2,
             "network_capacity_by_node": {"server": 1000}},
        )
        self.assertAlmostEqual(result["upload_critical_path_ms"], 255.0)
        self.assertAlmostEqual(result["download_critical_path_ms"], 170.0)
        self.assertAlmostEqual(result["aggregation_critical_path_ms"], 6.0)
        self.assertAlmostEqual(result["round_latency_sim_ms"], 431.0)

    def test_simulation_drift_two_cluster_multistage_path(self):
        events = []
        for src, dst in ((0, 1), (2, 3)):
            events.append({
                "src": src, "dst": dst, "message_type": "AGGREGATED_MODEL",
                "phase": "INTRA_CLUSTER_UPLOAD", "payload_bytes": 1_000_000,
                "bandwidth_mbps": 100, "latency_ms": 5, "success": True,
            })
        for src, dst, message_type, phase in (
            (1, 4, "AGGREGATED_MODEL", "INTER_CLUSTER_UPLOAD"),
            (3, 4, "AGGREGATED_MODEL", "INTER_CLUSTER_UPLOAD"),
            (4, 1, "GLOBAL_MODEL", "INTER_CLUSTER_DOWNLOAD"),
            (4, 3, "GLOBAL_MODEL", "INTER_CLUSTER_DOWNLOAD"),
            (1, 0, "GLOBAL_MODEL", "INTRA_CLUSTER_DOWNLOAD"),
            (3, 2, "GLOBAL_MODEL", "INTRA_CLUSTER_DOWNLOAD"),
        ):
            events.append({
                "src": src, "dst": dst, "message_type": message_type,
                "phase": phase, "payload_bytes": 1_000_000,
                "bandwidth_mbps": 100, "latency_ms": 5, "success": True,
            })
        result = simulate_round(
            events, [], [1, 1, 1, 1, 1], 1,
            {"default_bandwidth_mbps": 100, "default_link_latency_ms": 5,
             "default_compute_capacity": 1,
             "reference_train_ms_per_sample_epoch": 0,
             "aggregation_ms_per_model_mb": 2},
            {"aggregation_inputs_by_phase": {
                "INTRA_CLUSTER_UPLOAD": {"1": 2, "3": 2},
                "INTER_CLUSTER_UPLOAD": {"4": 2},
            }},
        )
        self.assertAlmostEqual(result["upload_critical_path_ms"], 255.0)
        self.assertAlmostEqual(result["download_critical_path_ms"], 255.0)
        self.assertAlmostEqual(result["aggregation_critical_path_ms"], 8.0)
        self.assertAlmostEqual(result["round_latency_sim_ms"], 518.0)

    def test_cluster_routes_return_without_flat_redistribution(self):
        tree = {
            0: {"parent": None, "children": [1, 2], "cluster_id": 0,
                "sample_count": 1},
            1: {"parent": 0, "children": [], "cluster_id": 0,
                "sample_count": 2},
            2: {"parent": 0, "children": [], "cluster_id": 0,
                "sample_count": 3},
            3: {"parent": None, "children": [4], "cluster_id": 3,
                "sample_count": 4},
            4: {"parent": 3, "children": [], "cluster_id": 3,
                "sample_count": 5},
        }
        models = [ModelWrapper({"w": torch.tensor([float(i)])}) for i in range(5)]
        comm = CommStats(round_id=0)
        weighted = hierarchical_weighted_reduce(models, tree, [0, 3], comm=comm)
        routes = {
            0: {"next_hop": 0}, 1: {"next_hop": 0},
            2: {"next_hop": 0}, 3: {"next_hop": 2},
            4: {"next_hop": 3},
        }
        queue = {}
        for head in [0, 3]:
            forward_model_to_best_client(
                head, weighted[head], routes, 0, queue, comm=comm
            )
        final, paths = aggregate_models_at_best_client(
            0, models, queue, torch.device("cpu")
        )
        self.assertAlmostEqual(final["w"].item(), 40 / 15, places=6)

        return_queue = {}
        return_model_to_cluster_heads(final, paths, return_queue, comm=comm)
        returned = {head: return_queue[head][-1]["model_dict"] for head in [0, 3]}
        distribute_returned_models_to_clusters(
            returned, tree, models, [0, 3], comm=comm
        )
        self.assertEqual(comm.intra_cluster_msgs, 3)
        self.assertEqual(comm.inter_cluster_forward_hops, 2)
        self.assertEqual(comm.inter_cluster_return_hops, 2)
        self.assertEqual(comm.intra_cluster_dist_msgs, 3)
        self.assertEqual(comm.total_msgs, 10)
        self.assertEqual(comm.flat_download_msgs, 0)
        self.assertEqual(len(comm.events), 10)
        for model in models:
            self.assertAlmostEqual(model.state_dict()["w"].item(), 40 / 15, places=6)

    def test_explicit_simulated_time_model(self):
        events = [{"src": 0, "dst": 1, "message_type": "MODEL_UPLOAD",
                   "payload_bytes": 1_000_000, "bandwidth_mbps": None,
                   "latency_ms": None, "success": True}]
        result = simulate_round(
            events,
            [{"node_id": 1, "aggregation_inputs": 1}],
            [10, 20], 1,
            {"default_bandwidth_mbps": 100, "default_link_latency_ms": 5,
             "default_compute_capacity": 1,
             "reference_train_ms_per_sample_epoch": 1,
             "aggregation_ms_per_model_mb": 2},
        )
        self.assertAlmostEqual(result["upload_critical_path_ms"], 85.0)
        self.assertAlmostEqual(result["round_latency_sim_ms"], 107.0)

    def test_ten_round_prefix_comparator(self):
        rows = "round_id,avg_train_loss,val_loss,val_acc\n" + "".join(
            f"{index},{1-index/100},{2-index/100},{index/100}\n"
            for index in range(10)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "fedavg.csv"
            second = Path(temp_dir) / "drift.csv"
            first.write_text(rows, encoding="utf-8")
            second.write_text(rows, encoding="utf-8")
            self.assertTrue(compare_prefixes(str(first), str(second))["passed"])

    def test_node_resources_do_not_depend_on_global_rng_state(self):
        clients = [{"id": index} for index in range(3)]
        topology = [[1], [0, 2], [1]]
        first = ElectionManager(clients, topology, seed=17).start_election()
        for _ in range(100):
            random.random()
        second = ElectionManager(clients, topology, seed=17).start_election()
        first_resources = [item["resource_params"] for item in first["all_nodes_info"]]
        second_resources = [item["resource_params"] for item in second["all_nodes_info"]]
        self.assertEqual(first_resources, second_resources)
        self.assertEqual(first["best_client_id"], second["best_client_id"])

    def test_system_metrics_are_recomputed_from_hop_events(self):
        events = [
            {"round": 0, "src": 0, "dst": 1, "message_type": "MODEL_UPLOAD",
             "payload_bytes": 100, "success": True},
            {"round": 0, "src": 2, "dst": 1, "message_type": "MODEL_UPLOAD",
             "payload_bytes": 100, "success": True},
            {"round": 0, "src": 1, "dst": 0, "message_type": "GLOBAL_MODEL",
             "payload_bytes": 100, "success": True},
        ]
        contexts = [{
            "round": 0, "node_ids": [0, 1, 2], "active_nodes": [0, 1, 2],
            "failed_nodes": [], "method": "test", "seed": 42,
        }]
        node_rows = build_node_round_stats(events, contexts)
        summary = build_metrics_summary(events, node_rows, contexts, [])
        self.assertEqual(summary["total_messages"], 3)
        self.assertEqual(summary["total_bytes"], 300)
        self.assertEqual(summary["max_peak_node_traffic_bytes"], 300)
        root = next(row for row in node_rows if str(row["node_id"]) == "1")
        self.assertEqual(root["aggregation_inputs"], 2)
        self.assertEqual(summary["energy_status"], "unsupported")

    def test_capacity_normalized_load_uses_aggregation_work(self):
        from fl_core.system_metrics import capacity_normalized_load_metrics

        node_rows = [
            {"node_id": 0, "aggregation_inputs": 2},
            {"node_id": 0, "aggregation_inputs": 2},
            {"node_id": 1, "aggregation_inputs": 2},
            {"node_id": 2, "aggregation_inputs": 0},
        ]
        result = capacity_normalized_load_metrics(
            node_rows, {"0": 1.0, "1": 0.5, "2": 0.25}
        )
        self.assertEqual(result["capacity_normalized_load_by_node"], {
            "0": 4.0, "1": 4.0, "2": 0.0,
        })
        self.assertEqual(result["max_capacity_normalized_load"], 4.0)
        self.assertEqual(result["capacity_normalized_load_imbalance"], 1.5)

    def test_capacity_normalized_load_rejects_nonpositive_capacity(self):
        from fl_core.system_metrics import capacity_normalized_load_metrics

        with self.assertRaises(ValueError):
            capacity_normalized_load_metrics([], {"0": 0.0})

    def test_repair_cost_is_explicit_when_control_plane_is_uninstrumented(self):
        events = [{"round": 0, "src": 0, "dst": 1, "method": "drift",
                   "message_type": "MODEL_UPLOAD", "payload_bytes": 10, "success": True}]
        contexts = [{"round": 0, "node_ids": [0, 1], "active_nodes": [0, 1],
                     "failed_nodes": [], "method": "drift", "seed": 1}]
        summary = build_metrics_summary(events, build_node_round_stats(events, contexts), contexts, [])
        self.assertIsNone(summary["repair_messages"])
        self.assertEqual(summary["repair_control_status"], "uninstrumented")

    def test_repair_cost_counts_control_events_once(self):
        events = [{"round": 0, "src": 0, "dst": 1, "method": "drift",
                   "phase": "CONTROL", "message_type": "REPAIR", "payload_bytes": 17,
                   "success": True}]
        contexts = [{"round": 0, "node_ids": [0, 1], "active_nodes": [0, 1],
                     "failed_nodes": [], "method": "drift", "seed": 1}]
        summary = build_metrics_summary(events, build_node_round_stats(events, contexts), contexts, [])
        self.assertEqual(summary["repair_messages"], 1)
        self.assertEqual(summary["repair_bytes"], 17)
        self.assertEqual(summary["repair_control_status"], "instrumented")

    def test_learning_rate_schedule_alignment(self):
        fed = RayFLRuntime(FLConfig(
            rounds=10,
            learning_rate_schedule_rounds=100,
            learning_rate=1e-3,
            learning_rate_min=1e-4,
            warmup_rounds=5,
        ))
        drift = NewFLRuntime(NewFLConfig(
            rounds=10,
            learning_rate_schedule_rounds=100,
            learning_rate=1e-3,
            learning_rate_min=1e-4,
            warmup_rounds=5,
        ))
        for round_id in range(10):
            expected = drift._calc_learning_rate(round_id)
            initial = fed.cfg.learning_rate
            minimum = fed.cfg.learning_rate_min
            warmup = fed.cfg.warmup_rounds
            if round_id < warmup:
                actual = initial * ((round_id + 1) / warmup)
            else:
                progress = round_id - warmup
                total = fed.cfg.learning_rate_schedule_rounds - warmup
                actual = minimum + 0.5 * (initial - minimum) * (
                    1 + math.cos(math.pi * progress / total)
                )
            self.assertAlmostEqual(actual, expected, places=10)

    def test_drift_global_evaluation_is_sample_level(self):
        cfg = NewFLConfig(
            dataset="synthetic", model="simple", num_classes=3,
            batch_size=4, use_gpu=False,
        )
        runtime = NewFLRuntime(cfg)
        torch.manual_seed(7)
        model = runtime._create_model()
        features = torch.randn(11, 10)
        labels = torch.randint(0, 3, (11,))
        runtime.set_global_test_data({"images": features, "labels": labels})

        loss, accuracy = runtime._evaluate_global_params(model.state_dict())
        with torch.no_grad():
            logits = model(features)
            expected_loss = nn.CrossEntropyLoss()(logits, labels).item()
            expected_accuracy = (logits.argmax(dim=1) == labels).float().mean().item()

        self.assertAlmostEqual(loss, expected_loss, places=6)
        self.assertAlmostEqual(accuracy, expected_accuracy, places=6)

    def test_shared_artifact_contract(self):
        cfg = FLConfig(num_clients=2, rounds=3, use_gpu=False)
        torch.manual_seed(3)
        initial_model = get_model_params(MLP(cfg.input_dim, cfg.num_classes))
        with tempfile.TemporaryDirectory() as temp_dir:
            write_experiment_inputs(
                output_dir=temp_dir,
                config=cfg,
                client_indices=[[0, 2], [1, 3]],
                initial_model=initial_model,
                physical_graph={0: [1], 1: [0]},
                node_resources=[{"node_id": 0}, {"node_id": 1}],
                link_profiles={"default": {"bandwidth_mbps": 100.0}},
                failure_trace=[],
                partition_metadata={
                    "dataset": "synthetic",
                    "split_policy": "default",
                    "partition_index_space_size": 4,
                },
            )
            write_completion_status(temp_dir, 3, 3, "COMPLETE")
            config_dir = Path(temp_dir) / "config"
            expected_files = {
                "experiment.json", "data_partition.json", "initial_model.pt",
                "physical_graph.json", "node_resources.json", "link_profiles.json",
                "failure_trace.json", "manifest.json", "completion_status.json",
            }
            self.assertEqual(expected_files, {path.name for path in config_dir.iterdir()})
            partition = json.loads((config_dir / "data_partition.json").read_text())
            self.assertEqual(partition["sample_counts"], [2, 2])
            restored = torch.load(config_dir / "initial_model.pt", weights_only=True)
            self.assertEqual(initial_model.keys(), restored.keys())
            for key in initial_model:
                self.assertTrue(torch.equal(initial_model[key], restored[key]))
            bundle = load_frozen_inputs(str(config_dir))
            validate_frozen_inputs(bundle, 2)
            self.assertEqual(bundle["data_partition.json"]["client_indices"],
                             [[0, 2], [1, 3]])
            validate_frozen_inputs(bundle, 2, expected_dataset="synthetic", strict=True)
            broken = dict(bundle)
            broken["data_partition.json"] = dict(bundle["data_partition.json"])
            broken["data_partition.json"]["client_indices"] = [[0, 2], [1]]
            broken["data_partition.json"]["sample_counts"] = [2, 1]
            broken["data_partition.json"]["partition_train_sample_count"] = 3
            with self.assertRaisesRegex(ValueError, "cover the complete index space"):
                validate_frozen_inputs(broken, 2, expected_dataset="synthetic", strict=True)

            duplicate_resources = dict(bundle)
            duplicate_resources["node_resources.json"] = [
                {"node_id": 0}, {"node_id": 0},
            ]
            with self.assertRaisesRegex(ValueError, "unique node_id"):
                validate_frozen_inputs(
                    duplicate_resources, 2, expected_dataset="synthetic", strict=True
                )

            invalid_failure = dict(bundle)
            invalid_failure["failure_trace.json"] = {
                "events": [{"event_type": "node_failure", "round": -1, "node_ids": [1]}]
            }
            with self.assertRaisesRegex(ValueError, "non-negative integer"):
                validate_frozen_inputs(
                    invalid_failure, 2, expected_dataset="synthetic", strict=True
                )

    def test_shared_node_resource_generation(self):
        self.assertEqual(generate_node_resources(4, 42),
                         generate_node_resources(4, 42))


if __name__ == "__main__":
    unittest.main()
