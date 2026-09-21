from __future__ import annotations

import json
import sys
import csv
import hashlib
from argparse import Namespace
from pathlib import Path

import pytest
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_experiment_matrix as matrix
from fl_core.artifacts import (
    failure_state_at_round, generate_failure_trace, generate_link_profiles,
    generate_node_resources,
)
from deasfl.election import DistributedElection, ElectionManager
from deasfl.runtime import NewFLRuntime
from scripts.calculate_offline_metrics import (
    _fedavg_edge_system_summary,
    _reinterpret_fedavg_edge,
)


def _args(tmp_path: Path, methods: str = "fedavg,drift") -> Namespace:
    return Namespace(
        scenario="unit_matrix",
        output_root=str(tmp_path / "matrix"),
        template_dir="configs/har",
        data_dir="data",
        dataset="har",
        model=None,
        input_dim=None,
        num_classes=None,
        methods=methods,
        seeds="42,43",
        num_clients=5,
        rounds=1,
        local_epochs=2,
        intra_cluster_rounds=1,
        global_sync_interval=1,
        batch_size=32,
        learning_rate=0.001,
        partition="dirichlet",
        alpha=1.0,
        use_gpu=True,
        drift_jsd_threshold=0.4,
        enable_cluster_diagnostics=False,
        cluster_diagnostics_interval=1,
        cluster_diagnostics_global_test=True,
    )


def test_prepare_plan_writes_shared_contract(tmp_path: Path) -> None:
    plans = matrix.prepare_plan(_args(tmp_path))

    assert len(plans) == 2
    for plan in plans:
        assert Path(plan["canonical_config"]).is_file()
        assert Path(plan["frozen_inputs_dir"]).name == "config"
        assert {item["method"] for item in plan["methods"]} == {"fedavg", "drift"}
        configs = [matrix._read_yaml(Path(item["config"])) for item in plan["methods"]]
        assert {config["frozen_inputs_dir"] for config in configs} == {
            plan["frozen_inputs_dir"]
        }
        assert all(config["strict_frozen_inputs"] is True for config in configs)
        assert plan["contract"]["learning_rate_schedule_rounds"] == 1
        assert plan["contract"]["warmup_rounds"] == 1
        assert len(plan["contract_hash"]) == 64


def test_manifest_is_json_and_commands_are_repo_relative(tmp_path: Path) -> None:
    plans = matrix.prepare_plan(_args(tmp_path, methods="dpsgd"))
    manifest = Path(tmp_path / "matrix" / "unit_matrix" / "seed_42" / "manifest.json")
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload["methods"][0]["method"] == "dpsgd"
    assert payload["methods"][0]["command"][1] == "scripts/run_dpsgd.py"
    assert plans[0]["seed"] == 42


def test_prepare_cifar10_uses_dataset_templates(tmp_path: Path) -> None:
    args = _args(tmp_path, methods="fedavg,drift")
    args.dataset = "cifar10"
    args.template_dir = "configs/cifar10"
    plans = matrix.prepare_plan(args)
    configs = [matrix._read_yaml(Path(item["config"])) for item in plans[0]["methods"]]

    assert {config["model"] for config in configs} == {"resnet18"}
    assert {config["dataset"] for config in configs} == {"cifar10"}


def test_scenario_spec_is_snapshotted_and_dynamic_dpsgd_is_registered(tmp_path: Path) -> None:
    spec = tmp_path / "e2.json"
    spec.write_text(json.dumps({
        "resource_profile": {"mode": "homogeneous", "capacity": 1.0},
        "network_profile": {"mode": "homogeneous"},
        "failure_profile": {"mode": "none"},
    }), encoding="utf-8")
    args = _args(tmp_path, methods="dynamic_dpsgd")
    args.scenario_spec = spec
    plans = matrix.prepare_plan(args)
    plan = plans[0]
    assert json.loads(Path(plan["scenario_spec"]).read_text(encoding="utf-8"))[
        "failure_profile"
    ]["mode"] == "none"
    assert plan["methods"][0]["command"][1] == "scripts/run_dynamic_dpsgd.py"
    assert matrix._digest(json.loads(Path(plan["scenario_spec"]).read_text())) == plan["scenario_spec_hash"]


def test_default_methods_keep_dynamic_dpsgd_opt_in() -> None:
    assert "dynamic_dpsgd" not in matrix.DEFAULT_METHODS
    assert "drift_no_resource_awareness" not in matrix.DEFAULT_METHODS
    assert set(matrix.DEFAULT_METHODS) == set(matrix.METHODS) - {
        "dynamic_dpsgd", "drift_no_resource_awareness", "drift_data_aware",
        "drift_random_matched", "drift_oracle_data_similar",
        "drift_jsd_q00", "drift_jsd_q25", "drift_jsd_q50",
        "drift_jsd_q75", "drift_jsd_q100",
        "feduc_inspired_censyn_i5",
    }


def test_feduc_inspired_is_opt_in_and_requires_seed_counts(tmp_path: Path) -> None:
    args = _args(tmp_path, methods="feduc_inspired_censyn_i5")
    args.dataset = "cifar10"
    args.template_dir = str(matrix.REPO_ROOT / "configs/cifar10")
    args.feduc_aggregator_counts = "42:9,43:3"
    args.feduc_deadline_slack = 1.25
    plans = matrix.prepare_plan(args)
    method = plans[0]["methods"][0]
    resolved = matrix._read_yaml(Path(method["config"]))
    assert method["command"][1] == "scripts/run_feduc.py"
    assert resolved["algorithm"] == "feduc_inspired_censyn"
    assert resolved["feduc_num_aggregators"] == 9
    assert resolved["feduc_global_sync_interval"] == 5
    assert resolved["feduc_objective_mode"] == "label_distribution_proxy"
    assert "client_test_partition_mode" not in resolved
    assert "enable_cluster_personalized_evaluation" not in resolved


def test_intra_cluster_rounds_only_apply_to_drift_variants(tmp_path: Path) -> None:
    args = _args(tmp_path, methods="fedavg,drift,drift_data_aware")
    args.intra_cluster_rounds = 3
    plans = matrix.prepare_plan(args)
    configs = {
        item["method"]: matrix._read_yaml(Path(item["config"]))
        for item in plans[0]["methods"]
    }
    assert "intra_cluster_rounds" not in configs["fedavg"]
    assert configs["drift"]["intra_cluster_rounds"] == 3
    assert configs["drift_data_aware"]["intra_cluster_rounds"] == 3
    assert plans[0]["contract"]["intra_cluster_rounds"] == 3


def test_global_sync_interval_only_applies_to_drift_variants(tmp_path: Path) -> None:
    args = _args(tmp_path, methods="fedavg,drift,drift_data_aware")
    args.global_sync_interval = 5
    plans = matrix.prepare_plan(args)
    configs = {
        item["method"]: matrix._read_yaml(Path(item["config"]))
        for item in plans[0]["methods"]
    }
    assert "global_sync_interval" not in configs["fedavg"]
    assert configs["drift"]["global_sync_interval"] == 5
    assert configs["drift_data_aware"]["global_sync_interval"] == 5
    assert plans[0]["contract"]["global_sync_interval"] == 5


def test_oracle_methods_are_explicit_experiment_only_configs(tmp_path: Path) -> None:
    args = _args(
        tmp_path,
        methods="drift_random_matched,drift_oracle_data_similar,drift_jsd_q50,drift_jsd_q100",
    )
    plans = matrix.prepare_plan(args)
    configs = {
        item["method"]: matrix._read_yaml(Path(item["config"]))
        for item in plans[0]["methods"]
    }
    assert configs["drift_random_matched"]["clustering_assignment_mode"] == "random_matched"
    assert configs["drift_oracle_data_similar"]["clustering_assignment_mode"] == "oracle_data_similar"
    assert configs["drift_jsd_q50"]["clustering_assignment_mode"] == "jsd_q50"
    assert configs["drift_jsd_q100"]["clustering_assignment_mode"] == "jsd_q100"
    assert all(config["enable_cluster_diagnostics"] for config in configs.values())
    assert all(config["distribution_similarity_threshold"] == 0.0 for config in configs.values())


def test_personalized_client_test_partition_is_opt_in(tmp_path: Path) -> None:
    args = _args(tmp_path, methods="drift_jsd_q00,drift_jsd_q100")
    args.dataset = "cifar10"
    args.template_dir = "configs/cifar10"
    args.client_test_partition_mode = "match_train_distribution"
    plans = matrix.prepare_plan(args)
    configs = [
        matrix._read_yaml(Path(item["config"]))
        for item in plans[0]["methods"]
    ]
    assert plans[0]["contract"]["client_test_partition_mode"] == "match_train_distribution"
    assert all(config["client_test_partition_mode"] == "match_train_distribution" for config in configs)
    assert all(config["enable_cluster_personalized_evaluation"] is True for config in configs)


def test_personalized_fields_are_not_injected_into_fedavg(tmp_path: Path) -> None:
    args = _args(tmp_path, methods="fedavg,drift")
    args.dataset = "cifar10"
    args.template_dir = "configs/cifar10"
    args.client_test_partition_mode = "match_train_distribution"
    plans = matrix.prepare_plan(args)
    configs = {
        item["method"]: matrix._read_yaml(Path(item["config"]))
        for item in plans[0]["methods"]
    }

    assert "client_test_partition_mode" not in configs["fedavg"]
    assert "enable_cluster_personalized_evaluation" not in configs["fedavg"]
    assert configs["drift"]["client_test_partition_mode"] == "match_train_distribution"
    assert configs["drift"]["enable_cluster_personalized_evaluation"] is True


def test_resource_awareness_ablation_uses_drift_launcher(tmp_path: Path) -> None:
    plans = matrix.prepare_plan(_args(tmp_path, methods="drift_no_resource_awareness"))
    method = plans[0]["methods"][0]
    config = matrix._read_yaml(Path(method["config"]))
    assert method["command"][1] == "scripts/test_30_clients.py"
    assert config["resource_awareness"] is False


def test_formal_scenario_profiles_are_deterministic_and_explicit() -> None:
    graph = {0: [1, 2], 1: [0], 2: [0]}
    resources = generate_node_resources(5, 42, {
        "mode": "tiers",
        "tiers": [
            {"fraction": 0.2, "capacity": 0.2},
            {"fraction": 0.6, "capacity": 0.5},
            {"fraction": 0.2, "capacity": 1.0},
        ],
    })
    assert sorted(item["compute_capacity"] for item in resources) == [0.2, 0.5, 0.5, 0.5, 1.0]
    links = generate_link_profiles(graph, {"mode": "medium_heterogeneous"}, seed=42)
    assert all("->" in key for key in links["edge_overrides"])
    trace = generate_failure_trace(graph, resources, 100, {
        "mode": "node_failure", "failure_round": 33, "recovery_round": 36,
    })
    assert trace["events"] == [
        {"round": 33, "event_type": "node_failure", "node_ids": [3]},
        {"round": 36, "event_type": "node_recovery", "node_ids": [3]},
    ]


def test_compute_tiers_do_not_change_network_capacity() -> None:
    resources = generate_node_resources(10, 42, {
        "mode": "tiers",
        "tiers": [
            {"fraction": 0.2, "capacity": 0.2},
            {"fraction": 0.6, "capacity": 0.5},
            {"fraction": 0.2, "capacity": 1.0},
        ],
    })
    assert {item["memory_gb"] for item in resources} == {32}
    assert {item["bandwidth_mbps"] for item in resources} == {1000}
    with pytest.raises(ValueError, match="sum to 1.0"):
        generate_node_resources(10, 42, {
            "mode": "tiers",
            "tiers": [{"fraction": 0.5, "capacity": 1.0}],
        })


def test_link_failure_uses_highest_betweenness_edge() -> None:
    line = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
    trace = generate_failure_trace(
        line, generate_node_resources(4, 42), 100,
        {"mode": "link_failure", "failure_round": 40, "recovery_round": 50},
    )
    assert trace["events"][0]["edges"] == [[1, 2]]
    with pytest.raises(ValueError, match="failure_round"):
        generate_failure_trace(
            line, generate_node_resources(4, 42), 3,
            {"mode": "link_failure", "failure_round": 40, "recovery_round": 50},
        )


def test_random_multi_node_failure_is_deterministic_shared_and_connected() -> None:
    graph = {
        node: sorted({(node - 1) % 10, (node + 1) % 10, (node + 5) % 10})
        for node in range(10)
    }
    resources = generate_node_resources(10, 42)
    profile = {
        "mode": "node_failure", "failure_round": 2, "recovery_round": 4,
        "selection_rule": "uniform_random_connected", "failure_count": 2,
    }
    first = generate_failure_trace(graph, resources, 6, profile, seed=42)
    second = generate_failure_trace(graph, resources, 6, profile, seed=42)
    assert first == second
    failed = set(first["events"][0]["node_ids"])
    assert len(failed) == 2
    assert first["events"][1]["node_ids"] == first["events"][0]["node_ids"]
    remaining = set(graph) - failed
    reached, frontier = set(), [min(remaining)]
    while frontier:
        node = frontier.pop()
        if node in reached:
            continue
        reached.add(node)
        frontier.extend(n for n in graph[node] if n in remaining and n not in reached)
    assert reached == remaining
    assert first["selection"]["remaining_connected"] is True


def test_random_multi_node_failure_rejects_invalid_failure_count() -> None:
    star = {0: [1, 2, 3], 1: [0], 2: [0], 3: [0]}
    resources = generate_node_resources(4, 42)
    with pytest.raises(ValueError, match="failure_count"):
        generate_failure_trace(star, resources, 6, {
            "mode": "node_failure", "failure_round": 2, "recovery_round": 4,
            "selection_rule": "uniform_random_connected", "failure_count": 4,
        }, seed=42)


def test_failure_state_is_persistent_until_recovery() -> None:
    bundle = {"failure_trace.json": {"events": [
        {"round": 1, "event_type": "node_failure", "node_ids": [2]},
        {"round": 2, "event_type": "link_failure", "edges": [[0, 1]]},
        {"round": 3, "event_type": "node_recovery", "node_ids": [2]},
        {"round": 4, "event_type": "link_recovery", "edges": [[0, 1]]},
    ]}}
    assert failure_state_at_round(bundle, 0) == (set(), set())
    assert failure_state_at_round(bundle, 2) == ({2}, {(0, 1)})
    assert failure_state_at_round(bundle, 3) == (set(), {(0, 1)})
    assert failure_state_at_round(bundle, 4) == (set(), set())


def test_drift_applies_frozen_node_recovery_persistently() -> None:
    runtime = object.__new__(NewFLRuntime)
    runtime.config = SimpleNamespace(enable_dynamic_nodes=False)
    runtime.frozen_inputs = {"failure_trace.json": {"events": [
        {"round": 0, "event_type": "node_failure", "node_ids": [1]},
        {"round": 2, "event_type": "node_recovery", "node_ids": [1]},
    ]}}
    runtime.active_nodes = {0, 1}
    runtime.failed_nodes = set()
    runtime.failed_edges = set()
    runtime.failure_log = []
    runtime.recovery_events = []
    runtime.tree_mapping = None
    runtime.client_sample_counts = [2, 3]
    runtime.clients = [{"indices": [0, 1]}, {"indices": [2, 3, 4]}]
    assert runtime._simulate_failures(0) == [1]
    assert runtime.failed_nodes == {1}
    assert runtime._simulate_failures(1) == []
    assert runtime.failed_nodes == {1}
    assert runtime._simulate_failures(2) == []
    assert runtime.failed_nodes == set()
    assert runtime._last_recovered_nodes == [1]
    assert runtime.client_sample_counts[1] == 3


def test_resource_awareness_override_removes_score_difference() -> None:
    low = DistributedElection(0, [], {
        "cpu_cores": 1, "memory_gb": 1, "bandwidth_mbps": 1,
        "gpu_memory_gb": 0, "resource_score_override": 1.0,
    }, seed=42)
    high = DistributedElection(1, [], {
        "cpu_cores": 64, "memory_gb": 128, "bandwidth_mbps": 10000,
        "gpu_memory_gb": 48, "resource_score_override": 1.0,
    }, seed=43)
    assert low.resource_score == high.resource_score == 1.0
    manager = ElectionManager(
        clients=[{"id": i} for i in range(3)],
        topology=[[1], [0, 2], [1]], seed=42,
        resource_params=[{
            "cpu_cores": 1, "memory_gb": 1, "bandwidth_mbps": 1,
            "gpu_memory_gb": 0, "resource_score_override": 1.0,
        } for _ in range(3)],
    )
    result = manager.start_election()
    assert result["best_client_id"] == 0
    assert all(item["best_client_id"] == 0 for item in result["all_nodes_info"])


def test_fedavg_edge_routes_server_messages_over_physical_graph(tmp_path: Path) -> None:
    config = tmp_path / "config"
    config.mkdir()
    (config / "physical_graph.json").write_text(
        json.dumps({"0": [1], "1": [0, 2], "2": [1]}), encoding="utf-8"
    )
    (config / "link_profiles.json").write_text(json.dumps({
        "default_bandwidth_mbps": 100.0, "default_latency_ms": 5.0,
        "drop_rate": 0.0, "edge_overrides": {
            "1->2": {"bandwidth_mbps": 20.0, "latency_ms": 10.0},
        },
    }), encoding="utf-8")
    events = [{
        "round": 0, "src": "server", "dst": "client_2",
        "message_id": "m0", "message_type": "BROADCAST", "success": True,
    }, {
        "round": 0, "src": "client_0", "dst": "server",
        "message_id": "m1", "message_type": "UPLOAD", "success": True,
    }]
    contexts = {0: {}}
    routed = _reinterpret_fedavg_edge(events, contexts, tmp_path, 0, 3)
    assert [(item["src"], item["dst"]) for item in routed] == [
        ("client_0", "client_1"), ("client_1", "client_2"),
    ]
    assert routed[-1]["bandwidth_mbps"] == 20.0
    assert contexts[0]["aggregation_inputs_by_phase"]["MODEL_UPLOAD"] == {
        "client_0": 3,
    }


def test_fedavg_edge_recomputes_capacity_normalized_server_load(tmp_path: Path) -> None:
    config, raw = tmp_path / "config", tmp_path / "raw"
    config.mkdir(); raw.mkdir()
    (config / "physical_graph.json").write_text(
        json.dumps({"0": [1], "1": [0]}), encoding="utf-8")
    (config / "link_profiles.json").write_text(json.dumps({
        "default_bandwidth_mbps": 100.0, "default_latency_ms": 5.0,
        "drop_rate": 0.0, "edge_overrides": {},
    }), encoding="utf-8")
    (config / "node_resources.json").write_text(json.dumps([
        {"node_id": 0, "compute_capacity": 0.5},
        {"node_id": 1, "compute_capacity": 1.0},
    ]), encoding="utf-8")
    fields = ["round", "src", "dst", "message_id", "message_type",
              "payload_bytes", "success", "phase"]
    with (raw / "communication_events.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([
            {"round": 0, "src": "client_1", "dst": "server", "message_id": "u",
             "message_type": "UPLOAD", "payload_bytes": 100, "success": True,
             "phase": "MODEL_UPLOAD"},
            {"round": 0, "src": "server", "dst": "client_1", "message_id": "d",
             "message_type": "BROADCAST", "payload_bytes": 100, "success": True,
             "phase": "MODEL_DISTRIBUTION"},
        ])
    (raw / "round_contexts.json").write_text(json.dumps([{
        "round": 0, "node_ids": ["0", "1"], "active_nodes": ["0", "1"],
        "failed_nodes": [], "roles": {}, "parents": {}, "children_counts": {},
        "aggregation_inputs_by_phase": {},
    }]), encoding="utf-8")
    summary = _fedavg_edge_system_summary(tmp_path, server_node=0, sample_count=2)
    assert summary["capacity_normalized_load_by_node"]["0"] == 4.0
    assert summary["max_capacity_normalized_load"] == 4.0


def test_matrix_summary_metric_keys_include_capacity_normalized_load() -> None:
    assert "max_capacity_normalized_load" in matrix.AGGREGATE_NUMERIC_KEYS


def test_run_guard_requires_explicit_long_run_permission(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_experiment_matrix.py",
            "--run-smoke",
            "--rounds",
            "4",
        ],
    )
    try:
        matrix.main()
    except ValueError as exc:
        assert "allow-long" in str(exc)
    else:  # pragma: no cover - assertion documents the safety contract.
        raise AssertionError("long smoke runs must require --allow-long")


def test_existing_contract_rejects_changed_rounds(tmp_path: Path) -> None:
    args = _args(tmp_path)
    plans = matrix.prepare_plan(args)
    changed = _args(tmp_path)
    changed.rounds = 3
    with pytest.raises(ValueError, match="contract differs"):
        matrix._validate_existing_contract(changed, plans)


def test_existing_contract_rejects_partial_seed_selection(tmp_path: Path) -> None:
    args = _args(tmp_path)
    plans = matrix.prepare_plan(args)
    changed = _args(tmp_path)
    changed.seeds = "42"
    with pytest.raises(ValueError, match="seed set"):
        matrix._validate_existing_contract(changed, plans)


def _fake_completed_run(root: Path, requested: int, actual: int) -> Path:
    run = root / "run_1"
    (run / "config").mkdir(parents=True)
    (run / "config" / "completion_status.json").write_text(
        json.dumps({
            "status": "COMPLETE",
            "completed_rounds": requested,
            "requested_rounds": requested,
        }), encoding="utf-8"
    )
    with (run / "fedavg_round_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["round", "val_acc"])
        writer.writeheader()
        for round_id in range(actual):
            writer.writerow({"round": round_id, "val_acc": 0.5})
    return run


def _fake_aggregate_plan(tmp_path: Path, requested: int, actual: int):
    output_root = tmp_path / "runs"
    run = _fake_completed_run(output_root, requested=requested, actual=actual)
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    for name in matrix.FROZEN_INPUT_FILES:
        payload = f"frozen:{name}".encode()
        (frozen / name).write_bytes(payload)
        (run / "config" / name).write_bytes(payload)
    (run / "config" / "experiment.json").write_text("{}", encoding="utf-8")
    planned_config = tmp_path / "fedavg.yaml"
    planned_config.write_text("{}", encoding="utf-8")
    canonical = tmp_path / "matrix" / "unit_matrix" / "seed_42" / "canonical_config.yaml"
    canonical.parent.mkdir(parents=True)
    canonical.write_text("{}", encoding="utf-8")
    method = {
        "method": "fedavg", "output_root": str(output_root),
        "config": str(planned_config),
    }
    plan = {
        "scenario": "unit_matrix", "seed": 42, "rounds": requested,
        "canonical_config": str(canonical), "methods": [method],
        "frozen_inputs_dir": str(frozen),
        "frozen_input_hashes": {
            name: hashlib.sha256((frozen / name).read_bytes()).hexdigest()
            for name in matrix.FROZEN_INPUT_FILES
        },
    }
    return plan, run


def test_aggregate_rejects_truncated_curve(tmp_path: Path) -> None:
    plan, _ = _fake_aggregate_plan(tmp_path, requested=2, actual=1)
    summary_path, _ = matrix.aggregate_matrix([plan])
    record = json.loads(summary_path.read_text(encoding="utf-8"))["records"][0]
    assert record["rounds"] == 1
    assert record["valid"] is False


def test_target_accuracy_triggers_offline_metrics_for_fedavg(tmp_path: Path, monkeypatch) -> None:
    plan, _ = _fake_aggregate_plan(tmp_path, requested=1, actual=1)
    calls = []
    monkeypatch.setattr(matrix.subprocess, "run", lambda command, **kwargs: calls.append(command))
    matrix.aggregate_matrix([plan], target_accuracy=0.4)
    assert any("--target-accuracy" in command for command in calls)


def test_failure_aggregate_pairs_same_method_control(tmp_path: Path, monkeypatch) -> None:
    plan, _ = _fake_aggregate_plan(tmp_path, requested=1, actual=1)
    control_root = tmp_path / "control"
    control_output = control_root / "seed_42" / "runs" / "fedavg"
    _fake_completed_run(control_output, requested=1, actual=1)
    calls = []
    monkeypatch.setattr(matrix.subprocess, "run", lambda command, **kwargs: calls.append(command))
    matrix.aggregate_matrix([plan], no_failure_control_root=control_root)
    assert any("--no-failure-control" in command for command in calls)


def test_metrics_summary_prefers_round_curve(tmp_path: Path) -> None:
    _, run = _fake_aggregate_plan(tmp_path, requested=1, actual=1)
    (run / "metrics").mkdir()
    (run / "metrics" / "metrics_summary.json").write_text(
        json.dumps({"final_accuracy": 0.1, "best_accuracy": 0.1}), encoding="utf-8"
    )
    summary = matrix._metrics_summary(run)
    assert summary["final_accuracy"] == 0.5
    assert summary["best_accuracy"] == 0.5


def test_metrics_summary_extracts_consensus_error(tmp_path: Path) -> None:
    _, run = _fake_aggregate_plan(tmp_path, requested=2, actual=2)
    metrics_path = run / "fedavg_round_metrics.csv"
    metrics_path.write_text(
        "round,val_acc,consensus_error\n0,0.4,0.3\n1,0.5,0.1\n",
        encoding="utf-8",
    )
    summary = matrix._metrics_summary(run)
    assert summary["final_consensus_error"] == pytest.approx(0.1)
    assert summary["mean_consensus_error"] == pytest.approx(0.2)


def test_aggregate_rejects_run_from_other_frozen_inputs(tmp_path: Path) -> None:
    plan, run = _fake_aggregate_plan(tmp_path, requested=1, actual=1)
    (run / "config" / "data_partition.json").write_text("changed", encoding="utf-8")
    summary_path, _ = matrix.aggregate_matrix([plan])
    record = json.loads(summary_path.read_text(encoding="utf-8"))["records"][0]
    assert record["valid"] is False
    assert "frozen inputs" in record["metrics_error"]


def test_aggregate_accepts_irrelevant_cifar_local_val_seed_metadata(tmp_path: Path) -> None:
    plan, run = _fake_aggregate_plan(tmp_path, requested=1, actual=1)
    frozen = Path(plan["frozen_inputs_dir"])
    expected = {"dataset": "cifar10", "client_indices": [[0, 1]], "sample_counts": [2],
                "local_val_seed": 42}
    actual = dict(expected)
    actual.pop("local_val_seed")
    (frozen / "data_partition.json").write_text(json.dumps(expected), encoding="utf-8")
    (run / "config" / "data_partition.json").write_text(json.dumps(actual), encoding="utf-8")
    plan["frozen_input_hashes"]["data_partition.json"] = hashlib.sha256(
        (frozen / "data_partition.json").read_bytes()
    ).hexdigest()
    (run / "config" / "experiment.json").write_text(
        json.dumps({"dataset": "cifar10", "har_local_val_seed": 42}), encoding="utf-8"
    )
    Path(plan["methods"][0]["config"]).write_text("dataset: cifar10\n", encoding="utf-8")

    summary_path, _ = matrix.aggregate_matrix([plan])
    record = json.loads(summary_path.read_text(encoding="utf-8"))["records"][0]
    assert record["valid"] is True


def test_aggregate_rejects_run_with_different_planned_rounds(tmp_path: Path) -> None:
    plan, _ = _fake_aggregate_plan(tmp_path, requested=2, actual=2)
    plan["rounds"] = 1
    summary_path, _ = matrix.aggregate_matrix([plan])
    record = json.loads(summary_path.read_text(encoding="utf-8"))["records"][0]
    assert record["valid"] is False
    assert "round mismatch" in record["metrics_error"]
