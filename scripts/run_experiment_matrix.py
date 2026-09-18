"""Unified experiment matrix generator, preflight checker, and scheduler.

The first version is intentionally conservative: it generates one frozen
input bundle per (scenario, seed), writes method-specific configs, supports
status/resume, and runs methods sequentially with independent logs. Formal
long runs require ``--allow-long`` explicitly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fl_core.artifacts import load_frozen_inputs, validate_frozen_inputs
from fl_core.config import FLConfig
from fl_core.data_loader import har_dataset_fingerprint, resolve_dataset_split_policy
from deasfl.config import NewFLConfig


FROZEN_INPUT_FILES = (
    "data_partition.json", "initial_model.pt", "physical_graph.json",
    "node_resources.json", "link_profiles.json", "failure_trace.json",
)
FROZEN_METADATA_FILES = ("experiment.json",)
COMMON_CONFIG_KEYS = (
    "dataset", "dataset_split_policy", "har_local_val_seed", "model",
    "input_dim", "num_classes", "num_clients", "rounds", "local_epochs",
    "batch_size", "learning_rate", "weight_decay", "learning_rate_min",
    "warmup_rounds", "cosine_annealing", "learning_rate_schedule_rounds",
    "data_partition_mode", "dirichlet_alpha", "seed", "deterministic_training",
    "client_sample_ratio", "client_drop_rate", "use_gpu",
)
SUMMARY_METRIC_KEYS = (
    "final_accuracy", "best_accuracy", "rounds_to_target", "target_status",
    "time_to_target_sim_ms", "accuracy_gap_to_fedavg_pp",
    "total_simulated_training_time_ms", "mean_round_latency_sim_ms",
    "p95_round_latency_sim_ms", "total_messages", "failed_messages",
    "total_bytes", "wan_bytes", "cross_cluster_bytes",
    "mean_peak_node_traffic_bytes", "max_peak_node_traffic_bytes",
    "communication_load_imbalance", "aggregation_load_imbalance",
    "max_capacity_normalized_load", "capacity_normalized_load_imbalance",
    "accuracy_recovery_rounds", "topology_recovery_time_sim_ms",
    "final_consensus_error", "mean_consensus_error", "rounds",
)
AGGREGATE_NUMERIC_KEYS = tuple(
    key for key in SUMMARY_METRIC_KEYS
    if key not in {"target_status", "rounds"}
)


METHODS: dict[str, dict[str, Any]] = {
    "fedavg": {
        "templates": {
            # The alignment template intentionally matches the DRIFT HAR
            # smoke contract (same frozen inputs and optimizer).
            "har": "config_fedavg_har_alignment_smoke.yaml",
            "cifar10": "config_fedavg.yaml",
        },
        "launcher": "scripts/run_fl.py",
        "kind": "run_fl",
    },
    "dpsgd": {
        "templates": {"har": "config_dpsgd_har_smoke.yaml", "cifar10": "config_dpsgd_smoke.yaml"},
        "launcher": "scripts/run_dpsgd.py",
        "kind": "config_only",
    },
    "dynamic_dpsgd": {
        "templates": {"har": "config_dpsgd_har_smoke.yaml", "cifar10": "config_dpsgd_smoke.yaml"},
        "launcher": "scripts/run_dynamic_dpsgd.py",
        "kind": "dynamic_dpsgd",
    },
    "resource_tree": {
        "templates": {"har": "config_resource_tree_har_smoke.yaml", "cifar10": "config_resource_tree_smoke.yaml"},
        "launcher": "scripts/run_resource_tree.py",
        "kind": "config_only",
    },
    "fixed_hierarchical": {
        "templates": {"har": "config_fixed_hierarchical_har_smoke.yaml", "cifar10": "config_fixed_hierarchical_smoke.yaml"},
        "launcher": "scripts/run_fixed_hierarchical.py",
        "kind": "config_only",
    },
    "epidemic_learning": {
        "templates": {"har": "config_epidemic_learning_har_smoke.yaml", "cifar10": "config_epidemic_learning_smoke.yaml"},
        "launcher": "scripts/run_epidemic_learning.py",
        "kind": "config_only",
    },
    "drift": {
        "templates": {"har": "config_de_asfl_har_alignment_smoke.yaml", "cifar10": "config_de_asfl.yaml"},
        "launcher": "scripts/test_30_clients.py",
        "kind": "drift",
    },
    "drift_no_resource_awareness": {
        "templates": {"har": "config_de_asfl_har_alignment_smoke.yaml", "cifar10": "config_de_asfl.yaml"},
        "launcher": "scripts/test_30_clients.py",
        "kind": "drift",
    },
    "drift_data_aware": {
        "templates": {"har": "config_de_asfl_har_alignment_smoke.yaml", "cifar10": "config_de_asfl.yaml"},
        "launcher": "scripts/test_30_clients.py",
        "kind": "drift",
    },
    "drift_random_matched": {
        "templates": {"har": "config_de_asfl_har_alignment_smoke.yaml", "cifar10": "config_de_asfl.yaml"},
        "launcher": "scripts/test_30_clients.py",
        "kind": "drift",
    },
    "drift_oracle_data_similar": {
        "templates": {"har": "config_de_asfl_har_alignment_smoke.yaml", "cifar10": "config_de_asfl.yaml"},
        "launcher": "scripts/test_30_clients.py",
        "kind": "drift",
    },
    **{
        f"drift_{mode}": {
            "templates": {"har": "config_de_asfl_har_alignment_smoke.yaml", "cifar10": "config_de_asfl.yaml"},
            "launcher": "scripts/test_30_clients.py",
            "kind": "drift",
        }
        for mode in ("jsd_q00", "jsd_q25", "jsd_q50", "jsd_q75", "jsd_q100")
    },
}

# Dynamic D-PSGD is an E4 failure baseline, not part of every static matrix.
# Keep it registered and explicitly selectable without silently expanding old
# commands that omit --methods.
DEFAULT_METHODS = (
    "fedavg", "dpsgd", "resource_tree", "fixed_hierarchical",
    "epidemic_learning", "drift",
)


def _method_spec(dataset: str, method: str) -> dict[str, str]:
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    templates = METHODS[method]["templates"]
    if dataset not in templates:
        raise ValueError(f"Dataset {dataset!r} has no {method} template")
    return {"template": templates[dataset], "launcher": METHODS[method]["launcher"],
            "kind": METHODS[method]["kind"]}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def _parse_csv(value: str, cast=str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _method_names(value: str) -> list[str]:
    names = _parse_csv(value)
    unknown = sorted(set(names) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}; available: {sorted(METHODS)}")
    if not names:
        raise ValueError("At least one method is required")
    if len(names) != len(set(names)):
        raise ValueError("Methods must be unique")
    return names


def _root(args: argparse.Namespace) -> Path:
    return Path(args.output_root).resolve() / args.scenario


def _seed_root(args: argparse.Namespace, seed: int) -> Path:
    return _root(args) / f"seed_{seed}"


def _effective_training(args: argparse.Namespace) -> dict[str, Any]:
    rounds = int(args.rounds)
    warmup = getattr(args, "warmup_rounds", None)
    if warmup is None:
        warmup = min(5, rounds)
    schedule_rounds = getattr(args, "learning_rate_schedule_rounds", None)
    if schedule_rounds is None:
        schedule_rounds = rounds
    return {
        "learning_rate": float(args.learning_rate),
        "learning_rate_min": float(getattr(args, "learning_rate_min", 1e-4)),
        "weight_decay": float(getattr(args, "weight_decay", 1e-4)),
        "warmup_rounds": int(warmup),
        "cosine_annealing": bool(getattr(args, "cosine_annealing", True)),
        "learning_rate_schedule_rounds": int(schedule_rounds),
        "client_sample_ratio": float(getattr(args, "client_sample_ratio", 1.0)),
        "client_drop_rate": float(getattr(args, "client_drop_rate", 0.0)),
    }


def _dataset_defaults(dataset: str) -> tuple[str, int, int]:
    defaults = {
        "har": ("harmlp", 561, 6),
        "cifar10": ("resnet18", 3072, 10),
    }
    try:
        return defaults[dataset]
    except KeyError as exc:
        raise ValueError("Supported datasets are har and cifar10") from exc


def _resolved_model_settings(args: argparse.Namespace) -> tuple[str, int, int]:
    model, input_dim, num_classes = _dataset_defaults(args.dataset)
    return (
        args.model or model,
        args.input_dim if args.input_dim is not None else input_dim,
        args.num_classes if args.num_classes is not None else num_classes,
    )


def _common_overrides(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    model, input_dim, num_classes = _resolved_model_settings(args)
    values = {
        "seed": seed,
        "num_clients": args.num_clients,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "data_dir": str(Path(args.data_dir).resolve()),
        "dataset": args.dataset,
        "model": model,
        "input_dim": input_dim,
        "num_classes": num_classes,
        "use_gpu": bool(args.use_gpu),
        "strict_frozen_inputs": True,
        "client_test_partition_mode": getattr(
            args, "client_test_partition_mode", "global_balanced"
        ),
        "enable_cluster_personalized_evaluation": (
            getattr(args, "client_test_partition_mode", "global_balanced")
            == "match_train_distribution"
        ),
    }
    values.update(_effective_training(args))
    return values


def _contract(args: argparse.Namespace, methods: list[str]) -> dict[str, Any]:
    model, input_dim, num_classes = _resolved_model_settings(args)
    return {
        "dataset": args.dataset,
        "data_dir": str(Path(args.data_dir).resolve()),
        "template_dir": str(Path(args.template_dir).resolve()),
        "model": model,
        "input_dim": input_dim,
        "num_classes": num_classes,
        "use_gpu": bool(args.use_gpu),
        "num_clients": int(args.num_clients),
        "rounds": int(args.rounds),
        "local_epochs": int(args.local_epochs),
        "intra_cluster_rounds": int(getattr(args, "intra_cluster_rounds", 1)),
        "batch_size": int(args.batch_size),
        "partition": args.partition,
        "alpha": float(args.alpha),
        "client_test_partition_mode": getattr(
            args, "client_test_partition_mode", "global_balanced"
        ),
        "methods": list(methods),
        "scenario_spec_hash": _digest(_scenario_spec(args)),
        **_effective_training(args),
    }


def _scenario_spec(args: argparse.Namespace) -> dict[str, Any]:
    if not getattr(args, "scenario_spec", None):
        return {}
    path = Path(args.scenario_spec).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Scenario spec must be a JSON object: {path}")
    return value


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_hashes(root: Path, names: tuple[str, ...] = FROZEN_INPUT_FILES) -> dict[str, str]:
    result = {}
    for name in names:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing immutable input: {path}")
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _canonical_config_value(config: dict[str, Any], key: str) -> Any:
    if key == "har_local_val_seed":
        dataset = str(config.get("dataset", "")).strip().lower()
        return config.get(key) if dataset == "har" else None
    if key == "data_partition_mode":
        value = config.get("data_partition_mode", config.get("partition"))
        return str(value).lower() if value is not None else None
    if key == "dirichlet_alpha":
        value = config.get("dirichlet_alpha", config.get("alpha"))
        return float(value) if value is not None else None
    if key == "dataset_split_policy" and config.get(key) is None:
        return "default"
    if key == "client_sample_ratio" and config.get(key) is None:
        return 1.0
    if key == "client_drop_rate" and config.get(key) is None:
        return 0.0
    value = config.get(key)
    if key in {"learning_rate", "learning_rate_min", "weight_decay",
               "dirichlet_alpha", "client_sample_ratio", "client_drop_rate"}:
        return float(value) if value is not None else None
    return value


def _apply_common(config: dict[str, Any], args: argparse.Namespace, seed: int,
                  frozen_dir: Path, method: str, output_dir: Path) -> dict[str, Any]:
    config.update(_common_overrides(args, seed))
    if method in METHODS and METHODS[method]["kind"] != "drift":
        config.pop("client_test_partition_mode", None)
        config.pop("enable_cluster_personalized_evaluation", None)
    config["output_dir"] = str(output_dir)
    config["program_name"] = f"{args.scenario}_{method}"
    config["frozen_inputs_dir"] = str(frozen_dir)
    if "dirichlet_alpha" in config:
        config["dirichlet_alpha"] = args.alpha
    if "alpha" in config:
        config["alpha"] = args.alpha
    if "data_partition_mode" in config:
        config["data_partition_mode"] = args.partition
    if "partition" in config:
        config["partition"] = args.partition
    return config


def _clean_drift_config(config: dict[str, Any]) -> dict[str, Any]:
    """Remove launcher-only FLConfig fields before NewFLConfig loading."""
    for key in ("program_name", "client_sample_ratio", "client_drop_rate"):
        config.pop(key, None)
    return config


def prepare_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    methods = _method_names(args.methods)
    seeds = _parse_csv(args.seeds, int)
    if not seeds:
        raise ValueError("At least one seed is required")
    template_dir = Path(args.template_dir).resolve()
    if args.dataset not in {"har", "cifar10"}:
        raise ValueError("Supported datasets are har and cifar10")
    if (getattr(args, "client_test_partition_mode", "global_balanced")
            == "match_train_distribution" and args.dataset != "cifar10"):
        raise ValueError("match_train_distribution is currently supported only for CIFAR-10")
    contract = _contract(args, methods)
    scenario_spec = _scenario_spec(args)
    plans = []
    for seed in seeds:
        seed_root = _seed_root(args, seed)
        frozen_root = seed_root / "frozen_inputs"
        frozen_dir = frozen_root / "config"
        seed_root.mkdir(parents=True, exist_ok=True)
        canonical_path_template = template_dir / _method_spec(args.dataset, "drift")["template"]
        if not canonical_path_template.is_file():
            raise FileNotFoundError(f"Missing canonical template: {canonical_path_template}")
        canonical = _read_yaml(canonical_path_template)
        _apply_common(canonical, args, seed, Path(""), "scenario", seed_root / "canonical")
        canonical["frozen_inputs_dir"] = ""
        # NewFLConfig has no program_name field; it is only a legacy launcher
        # concept used by the method-specific FLConfig files.
        canonical.pop("program_name", None)
        _clean_drift_config(canonical)
        canonical_path = seed_root / "canonical_config.yaml"
        _write_yaml(canonical_path, canonical)
        scenario_spec_path = seed_root / "scenario_spec.json"
        scenario_spec_path.write_text(
            json.dumps(scenario_spec, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        method_plans = []
        for method in methods:
            spec = _method_spec(args.dataset, method)
            template_path = template_dir / spec["template"]
            if not template_path.is_file():
                raise FileNotFoundError(f"Missing template for {method}: {template_path}")
            config = _read_yaml(template_dir / spec["template"])
            method_root = seed_root / "runs" / method
            config_path = seed_root / "configs" / f"{method}.yaml"
            _apply_common(config, args, seed, frozen_dir, method, method_root)
            if method == "dynamic_dpsgd":
                config["algorithm"] = "dynamic_dpsgd"
            if method == "drift_no_resource_awareness":
                config["resource_awareness"] = False
            if method in {
                "drift", "drift_no_resource_awareness", "drift_data_aware",
                "drift_random_matched", "drift_oracle_data_similar",
            } or method.startswith("drift_jsd_q"):
                config["intra_cluster_rounds"] = args.intra_cluster_rounds
            if method == "drift" and args.enable_cluster_diagnostics:
                config["distribution_similarity_threshold"] = 0.0
                config["enable_cluster_diagnostics"] = True
                config["cluster_diagnostics_interval"] = args.cluster_diagnostics_interval
                config["cluster_diagnostics_global_test"] = args.cluster_diagnostics_global_test
            if method == "drift_data_aware":
                if not 0 < args.drift_jsd_threshold <= math.log(2.0):
                    raise ValueError(
                        "drift_data_aware requires 0 < --drift-jsd-threshold <= ln(2)"
                    )
                config["distribution_similarity_threshold"] = args.drift_jsd_threshold
                config["enable_cluster_diagnostics"] = True
                config["cluster_diagnostics_interval"] = args.cluster_diagnostics_interval
                config["cluster_diagnostics_global_test"] = args.cluster_diagnostics_global_test
                config["method_name"] = "drift_data_aware"
            if method in {"drift_random_matched", "drift_oracle_data_similar"} or method.startswith("drift_jsd_q"):
                config["distribution_similarity_threshold"] = 0.0
                if method == "drift_random_matched":
                    config["clustering_assignment_mode"] = "random_matched"
                elif method == "drift_oracle_data_similar":
                    config["clustering_assignment_mode"] = "oracle_data_similar"
                else:
                    config["clustering_assignment_mode"] = method.removeprefix("drift_")
                config["enable_cluster_diagnostics"] = True
                config["cluster_diagnostics_interval"] = args.cluster_diagnostics_interval
                config["cluster_diagnostics_global_test"] = args.cluster_diagnostics_global_test
                config["method_name"] = method
            if method == "drift":
                _clean_drift_config(config)
            elif method in {
                "drift_no_resource_awareness", "drift_data_aware",
                "drift_random_matched", "drift_oracle_data_similar",
            } or method.startswith("drift_jsd_q"):
                _clean_drift_config(config)
            _write_yaml(config_path, config)
            command = [sys.executable, spec["launcher"], "--config", str(config_path)]
            if spec["kind"] == "run_fl":
                command.extend(["--data-dir", str(Path(args.data_dir).resolve())])
            elif spec["kind"] == "drift":
                command.extend([
                    "--rounds", str(args.rounds),
                    "--frozen-inputs-dir", str(frozen_dir),
                    "--data-dir", str(Path(args.data_dir).resolve()),
                ])
            method_plans.append({
                "method": method,
                "config": str(config_path),
                "output_root": str(method_root),
                "command": command,
                "log": str(seed_root / "logs" / f"{method}.log"),
            })
        manifest = {
            "schema_version": 3,
            "created_at_utc": _now(),
            "scenario": args.scenario,
            "seed": seed,
            "dataset": args.dataset,
            "data_dir": str(Path(args.data_dir).resolve()),
            "num_clients": args.num_clients,
            "rounds": args.rounds,
            "partition": args.partition,
            "alpha": args.alpha,
            "canonical_config": str(canonical_path),
            "scenario_spec": str(scenario_spec_path),
            "scenario_spec_hash": _digest(scenario_spec),
            "frozen_inputs_dir": str(frozen_dir),
            "frozen_inputs_root": str(frozen_root),
            "methods": method_plans,
            "contract": {**contract, "seed": seed},
            "contract_hash": _digest({**contract, "seed": seed}),
        }
        manifest["canonical_config_hash"] = hashlib.sha256(
            canonical_path.read_bytes()
        ).hexdigest()
        manifest["config_hashes"] = {
            method["method"]: hashlib.sha256(Path(method["config"]).read_bytes()).hexdigest()
            for method in method_plans
        }
        (seed_root / "logs").mkdir(exist_ok=True)
        (seed_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        plans.append(manifest)
    return plans


def generate_frozen_inputs(plans: list[dict[str, Any]]) -> None:
    for plan in plans:
        frozen = Path(plan["frozen_inputs_dir"])
        if (frozen / "data_partition.json").is_file():
            try:
                load_frozen_inputs(str(frozen))
                continue
            except (FileNotFoundError, ValueError, RuntimeError):
                # A previous interrupted generation may have left a partial
                # bundle. Rebuild it instead of silently accepting it.
                pass
        command = [
            sys.executable, "scripts/generate_frozen_inputs.py",
            "--config", plan["canonical_config"],
            # generate_frozen_inputs.py passes --output to
            # write_experiment_inputs, which creates its own config/child.
            "--output", str(Path(plan["frozen_inputs_root"])),
            "--data-dir", plan["data_dir"],
        ]
        if plan.get("scenario_spec"):
            command.extend(["--scenario-spec", plan["scenario_spec"]])
        log = frozen.parent.parent / "logs" / "generate_frozen_inputs.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as handle:
            subprocess.run(command, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
    for plan in plans:
        frozen = Path(plan["frozen_inputs_dir"])
        plan["frozen_input_hashes"] = _file_hashes(frozen)
        plan["frozen_metadata_hashes"] = _file_hashes(frozen, FROZEN_METADATA_FILES)
        manifest_path = Path(plan["canonical_config"]).parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["frozen_input_hashes"] = plan["frozen_input_hashes"]
        manifest["frozen_metadata_hashes"] = plan["frozen_metadata_hashes"]
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _validate_existing_contract(args: argparse.Namespace, plans: list[dict[str, Any]]) -> None:
    methods = _method_names(args.methods)
    requested_seeds = _parse_csv(args.seeds, int)
    if len(requested_seeds) != len(set(requested_seeds)):
        raise ValueError("Seeds must be unique")
    actual_seeds = sorted(int(plan["seed"]) for plan in plans)
    if sorted(requested_seeds) != actual_seeds:
        raise ValueError(
            f"Existing seed set {actual_seeds} differs from requested seed set "
            f"{sorted(requested_seeds)}; use the complete set or a new scenario name."
        )
    requested = _contract(args, methods)
    requested_without_seed = requested.copy()
    for plan in plans:
        recorded = plan.get("contract")
        if not isinstance(recorded, dict):
            raise ValueError(
                "Existing matrix manifest has no frozen contract; use a new scenario name "
                "before running it with the hardened scheduler."
            )
        expected = {**requested_without_seed, "seed": int(plan["seed"])}
        if _digest(expected) != plan.get("contract_hash"):
            raise ValueError(
                f"Scenario {plan['scenario']} seed={plan['seed']} contract differs from "
                "the existing manifest; use a new scenario name."
            )
        if list(plan.get("methods", [])) and [item["method"] for item in plan["methods"]] != methods:
            raise ValueError("Existing method order differs from the requested matrix")


def load_existing_plans(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Load manifests without rewriting configs or changing an experiment."""
    manifests = sorted(_root(args).glob("seed_*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(
            f"No matrix manifests found below {_root(args)}; run --generate-only first"
        )
    plans = []
    for manifest_path in manifests:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or "methods" not in value:
            raise ValueError(f"Invalid matrix manifest: {manifest_path}")
        plans.append(value)
    return plans


def preflight(plans: list[dict[str, Any]]) -> None:
    for plan in plans:
        if int(plan.get("schema_version", 0)) < 3:
            raise ValueError(
                f"Legacy matrix manifest for {plan['scenario']} seed={plan['seed']}; "
                "regenerate the scenario with the current scheduler."
            )
        scenario_path = Path(plan["scenario_spec"])
        scenario_spec = json.loads(scenario_path.read_text(encoding="utf-8"))
        if _digest(scenario_spec) != plan.get("scenario_spec_hash"):
            raise ValueError(f"Scenario spec hash mismatch: {scenario_path}")
        frozen = load_frozen_inputs(plan["frozen_inputs_dir"])
        if not isinstance(plan.get("contract"), dict) or _digest(plan["contract"]) != plan.get("contract_hash"):
            raise ValueError(f"Invalid or mutable experiment contract: seed={plan['seed']}")
        actual_hashes = _file_hashes(Path(plan["frozen_inputs_dir"]))
        recorded_hashes = plan.get("frozen_input_hashes")
        if recorded_hashes != actual_hashes:
            raise ValueError(f"Frozen input hash mismatch: {plan['scenario']} seed={plan['seed']}")
        actual_metadata_hashes = _file_hashes(
            Path(plan["frozen_inputs_dir"]), FROZEN_METADATA_FILES
        )
        if plan.get("frozen_metadata_hashes") != actual_metadata_hashes:
            raise ValueError(
                f"Frozen experiment metadata hash mismatch: {plan['scenario']} seed={plan['seed']}"
            )
        canonical_path = Path(plan["canonical_config"])
        actual_canonical_hash = hashlib.sha256(canonical_path.read_bytes()).hexdigest()
        if plan.get("canonical_config_hash") != actual_canonical_hash:
            raise ValueError(f"Canonical config hash mismatch: {canonical_path}")
        validate_frozen_inputs(
            frozen, plan["num_clients"], expected_dataset=plan["dataset"],
            expected_split_policy=resolve_dataset_split_policy(plan["dataset"], None),
            expected_dataset_fingerprint=(
                har_dataset_fingerprint(plan["data_dir"])
                if plan["dataset"] == "har" else None
            ), strict=True,
        )
        graph = frozen["physical_graph.json"]
        reachable = {0}
        frontier = [0]
        while frontier:
            node = frontier.pop()
            for neighbor in graph.get(node, []):
                if neighbor not in reachable:
                    reachable.add(neighbor)
                    frontier.append(neighbor)
        if len(reachable) != plan["num_clients"]:
            raise ValueError(
                f"Frozen physical graph is disconnected: {len(reachable)}/{plan['num_clients']} nodes reachable"
            )
        canonical = _read_yaml(Path(plan["canonical_config"]))
        _clean_drift_config(canonical)
        NewFLConfig(**canonical)
        common_reference = None
        for method in plan["methods"]:
            config_path = Path(method["config"])
            if not config_path.is_file():
                raise FileNotFoundError(f"Missing method config: {config_path}")
            expected_config_hash = plan.get("config_hashes", {}).get(method["method"])
            actual_config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
            if expected_config_hash != actual_config_hash:
                raise ValueError(f"Config hash mismatch for {method['method']}")
            config = _read_yaml(config_path)
            if config.get("frozen_inputs_dir") != plan["frozen_inputs_dir"]:
                raise ValueError(f"{method['method']} does not use the shared frozen input bundle")
            if METHODS[method["method"]]["kind"] == "drift":
                NewFLConfig(**_clean_drift_config(dict(config)))
            else:
                unknown = sorted(set(config) - set(FLConfig.__dataclass_fields__))
                if unknown:
                    raise ValueError(f"Unsupported {method['method']} config fields: {unknown}")
                FLConfig(**config)
            normalized = {
                key: _canonical_config_value(config, key) for key in COMMON_CONFIG_KEYS
            }
            if common_reference is None:
                common_reference = normalized
            else:
                mismatched = [key for key in COMMON_CONFIG_KEYS
                              if normalized[key] != common_reference[key]]
                if mismatched:
                    raise ValueError(
                        f"Training contract mismatch for {method['method']}: {mismatched}"
                    )
        print(f"PRECHECK PASS: {plan['scenario']} seed={plan['seed']}")


def _latest_run(output_root: Path) -> Path | None:
    statuses = list(output_root.glob("**/config/completion_status.json"))
    if not statuses:
        return None
    return max(statuses, key=lambda path: path.stat().st_mtime).parent.parent


def _run_status(output_root: Path) -> dict[str, Any]:
    run = _latest_run(output_root)
    if run is None:
        return {"status": "NOT_STARTED", "run": None}
    status_path = run / "config" / "completion_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["run"] = str(run)
    return status


def _accuracy(run: Path) -> float | None:
    files = sorted(run.glob("*_round_metrics.csv"))
    if len(files) != 1:
        return None
    with files[0].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    return float(rows[-1].get("val_acc", rows[-1].get("global_accuracy", "nan")))


def write_status(plans: list[dict[str, Any]]) -> Path:
    rows = []
    for plan in plans:
        for method in plan["methods"]:
            status = _run_status(Path(method["output_root"]))
            run = Path(status["run"]) if status.get("run") else None
            rows.append({
                "scenario": plan["scenario"], "seed": plan["seed"],
                "method": method["method"], "status": status.get("status"),
                "completed_rounds": status.get("completed_rounds"),
                "requested_rounds": status.get("requested_rounds"),
                "final_accuracy": _accuracy(run) if run else None,
                "run": str(run) if run else None,
                "log": method["log"],
            })
    output = _root_from_manifest(plans[0]) / "matrix_status.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    for row in rows:
        print(f"{row['scenario']} seed={row['seed']} {row['method']}: {row['status']} {row['final_accuracy']}")
    return output


def _metrics_summary(run: Path) -> dict[str, Any]:
    path = run / "metrics" / "metrics_summary.json"
    summary: dict[str, Any] = {}
    if path.is_file():
        summary.update(json.loads(path.read_text(encoding="utf-8")))
    metrics_files = sorted(run.glob("*_round_metrics.csv"))
    if len(metrics_files) != 1:
        raise FileNotFoundError(f"Cannot locate unique round metrics under {run}")
    with metrics_files[0].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    accuracy = [float(row["val_acc"]) for row in rows]
    if not accuracy:
        raise ValueError(f"No accuracy rows under {run}")
    # The round curve is the primary learning record.  Never allow an older
    # derived summary to override a completed curve after resume/recalculation.
    summary["final_accuracy"] = accuracy[-1]
    summary["best_accuracy"] = max(accuracy)
    summary["rounds"] = len(accuracy)
    if "consensus_error" in rows[0]:
        consensus = [
            float(row["consensus_error"])
            for row in rows if row.get("consensus_error") not in (None, "")
        ]
        if consensus:
            summary["final_consensus_error"] = consensus[-1]
            summary["mean_consensus_error"] = statistics.fmean(consensus)
    return summary


def _validate_run_against_plan(
    plan: dict[str, Any], method: dict[str, Any], run: Path, status: dict[str, Any]
) -> None:
    """Bind a selected run to its matrix contract before aggregation."""
    requested_rounds = int(status.get("requested_rounds", -1))
    if requested_rounds != int(plan["rounds"]):
        raise ValueError(
            f"Run/plan round mismatch for {method['method']}: "
            f"{requested_rounds}/{plan['rounds']}"
        )
    run_config_dir = run / "config"
    expected_hashes = plan.get("frozen_input_hashes")
    if not isinstance(expected_hashes, dict):
        raise ValueError(f"Missing frozen input hashes for seed={plan['seed']}")
    actual_hashes = _file_hashes(run_config_dir)
    mismatched_inputs = [
        name for name in FROZEN_INPUT_FILES
        if name != "data_partition.json"
        and actual_hashes.get(name) != expected_hashes.get(name)
    ]
    if actual_hashes.get("data_partition.json") != expected_hashes.get("data_partition.json"):
        try:
            expected_partition = json.loads(
                (Path(plan["frozen_inputs_dir"]) / "data_partition.json").read_text(
                    encoding="utf-8"
                )
            )
            actual_partition = json.loads(
                (run_config_dir / "data_partition.json").read_text(encoding="utf-8")
            )
            # local_val_seed only affects HAR's local validation split.  Some
            # runtimes record the default in CIFAR artifacts while others omit
            # it, so it must not invalidate otherwise identical CIFAR inputs.
            if str(expected_partition.get("dataset", "")).lower() != "har":
                expected_partition.pop("local_val_seed", None)
            if str(actual_partition.get("dataset", "")).lower() != "har":
                actual_partition.pop("local_val_seed", None)
            if expected_partition != actual_partition:
                mismatched_inputs.append("data_partition.json")
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            mismatched_inputs.append("data_partition.json")
    if mismatched_inputs:
        raise ValueError(
            f"Run frozen inputs do not match the matrix plan for {method['method']}: "
            f"{', '.join(mismatched_inputs)}"
        )
    experiment_path = run_config_dir / "experiment.json"
    if not experiment_path.is_file():
        raise FileNotFoundError(f"Missing run experiment config: {experiment_path}")
    actual_config = json.loads(experiment_path.read_text(encoding="utf-8"))
    if not isinstance(actual_config, dict):
        raise ValueError(f"Invalid run experiment config: {experiment_path}")
    planned_config = _read_yaml(Path(method["config"]))
    mismatched = [
        key for key in COMMON_CONFIG_KEYS
        if _canonical_config_value(actual_config, key)
        != _canonical_config_value(planned_config, key)
    ]
    if mismatched:
        raise ValueError(
            f"Run training contract mismatch for {method['method']}: {mismatched}"
        )


def aggregate_matrix(
    plans: list[dict[str, Any]], simulation_config: Path | None = None,
    target_accuracy: float | None = None,
    no_failure_control_root: Path | None = None,
    fedavg_edge_server_node: int | None = None,
) -> tuple[Path, Path]:
    records: list[dict[str, Any]] = []
    for plan in plans:
        control_run = None
        for method in plan["methods"]:
            if method["method"] == "fedavg":
                control_status = _run_status(Path(method["output_root"]))
                if control_status.get("status") == "COMPLETE":
                    control_run = Path(control_status["run"])
        for method in plan["methods"]:
            status = _run_status(Path(method["output_root"]))
            record: dict[str, Any] = {
                "scenario": plan["scenario"], "seed": plan["seed"],
                "method": method["method"], "status": status.get("status"),
                "completed_rounds": status.get("completed_rounds"),
                "requested_rounds": status.get("requested_rounds"),
                "valid": False, "run": status.get("run"),
            }
            if status.get("status") != "COMPLETE" or not status.get("run"):
                record["metrics_error"] = "run_not_complete"
                records.append(record)
                continue
            run = Path(status["run"])
            try:
                _validate_run_against_plan(plan, method, run, status)
                no_failure_control = None
                if no_failure_control_root is not None:
                    control_output = (
                        no_failure_control_root / f"seed_{plan['seed']}"
                        / "runs" / method["method"]
                    )
                    control_status = _run_status(control_output)
                    if control_status.get("status") != "COMPLETE" or not control_status.get("run"):
                        raise ValueError(
                            f"Missing COMPLETE no-failure control for seed={plan['seed']} "
                            f"method={method['method']} below {control_output}"
                        )
                    no_failure_control = Path(control_status["run"])
                if (simulation_config is not None or target_accuracy is not None
                        or no_failure_control is not None
                        or (control_run is not None and method["method"] != "fedavg")):
                    command = [sys.executable, "scripts/calculate_offline_metrics.py", "--run", str(run)]
                    if simulation_config is not None:
                        command.extend(["--simulation-config", str(simulation_config)])
                    if control_run is not None and run != control_run:
                        command.extend(["--fedavg-control", str(control_run)])
                    if target_accuracy is not None:
                        command.extend(["--target-accuracy", str(target_accuracy)])
                    if no_failure_control is not None:
                        command.extend(["--no-failure-control", str(no_failure_control)])
                    subprocess.run(command, cwd=REPO_ROOT, check=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                metrics = _metrics_summary(run)
                record.update({
                    key: metrics[key] for key in SUMMARY_METRIC_KEYS if key in metrics
                })
                record["valid"] = (
                    int(status.get("completed_rounds", -1)) == int(status.get("requested_rounds", -2))
                    and int(record.get("rounds", -1)) == int(status.get("requested_rounds", -2))
                    and int(status.get("requested_rounds", -2)) == int(plan["rounds"])
                )
                if (method["method"] == "fedavg" and simulation_config is not None
                        and fedavg_edge_server_node is not None):
                    edge_command = [
                        sys.executable, "scripts/calculate_offline_metrics.py",
                        "--run", str(run), "--simulation-config", str(simulation_config),
                        "--fedavg-edge-server-node", str(fedavg_edge_server_node),
                    ]
                    subprocess.run(
                        edge_command, cwd=REPO_ROOT, check=True,
                        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                    )
                    edge_metrics = json.loads(
                        (run / "metrics" / "fedavg_edge_metrics.json").read_text(encoding="utf-8")
                    )
                    edge_record = dict(record)
                    edge_record["method"] = "fedavg_edge"
                    for key in (
                        "total_simulated_training_time_ms", "mean_round_latency_sim_ms",
                        "p95_round_latency_sim_ms", "time_to_target_sim_ms",
                        "total_messages", "failed_messages", "total_bytes", "wan_bytes",
                        "cross_cluster_bytes", "mean_peak_node_traffic_bytes",
                        "max_peak_node_traffic_bytes", "communication_load_imbalance",
                        "aggregation_load_imbalance",
                        "max_capacity_normalized_load",
                        "capacity_normalized_load_imbalance",
                        "capacity_normalized_load_by_node",
                        "capacity_normalized_load_definition",
                    ):
                        if key in edge_metrics:
                            edge_record[key] = edge_metrics[key]
                    edge_record["system_interpretation"] = "fedavg_edge"
                    edge_record["fedavg_edge_server_node"] = fedavg_edge_server_node
                    records.append(edge_record)
            except (OSError, subprocess.CalledProcessError, TypeError, ValueError, KeyError) as exc:
                record["metrics_error"] = repr(exc)
            records.append(record)

    aggregate: dict[str, Any] = {}
    for method in sorted({record["method"] for record in records}):
        valid = [record for record in records if record["method"] == method and record.get("valid")]
        summary: dict[str, Any] = {"valid_runs": len(valid)}
        for key in AGGREGATE_NUMERIC_KEYS:
            values = [float(item[key]) for item in valid if item.get(key) is not None]
            if values:
                summary[f"{key}_mean"] = statistics.fmean(values)
                summary[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregate[method] = summary
    root = _root_from_manifest(plans[0])
    json_path = root / "matrix_summary.json"
    csv_path = root / "matrix_summary.csv"
    payload = {"scenario": plans[0]["scenario"], "records": records, "aggregate": aggregate}
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = sorted({key for record in records for key in record})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return json_path, csv_path


def _root_from_manifest(plan: dict[str, Any]) -> Path:
    return Path(plan["canonical_config"]).parents[1]


def run_methods(plans: list[dict[str, Any]], resume: bool) -> None:
    for plan in plans:
        for method in plan["methods"]:
            existing = _run_status(Path(method["output_root"]))
            if resume and existing.get("status") == "COMPLETE":
                print(f"SKIP COMPLETE: seed={plan['seed']} {method['method']}")
                continue
            log_path = Path(method["log"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"RUN: seed={plan['seed']} {method['method']} -> {log_path}")
            with log_path.open("w", encoding="utf-8") as handle:
                result = subprocess.run(
                    method["command"], cwd=REPO_ROOT, stdout=handle,
                    stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
            if result.returncode != 0:
                raise RuntimeError(f"{method['method']} failed; see {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--generate-only", action="store_true")
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--run-smoke", action="store_true")
    modes.add_argument("--status", action="store_true")
    modes.add_argument("--aggregate", action="store_true")
    parser.add_argument("--scenario", default="har_matrix_v1")
    parser.add_argument("--output-root", default="outputs/experiment_matrix_v1")
    parser.add_argument("--template-dir", default=None)
    parser.add_argument("--scenario-spec", type=Path, default=None,
                        help="Frozen resource/network/failure scenario JSON")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--dataset", default="har")
    parser.add_argument("--model", default=None,
                        help="Model override; defaults to harmlp for HAR and resnet18 for CIFAR-10")
    parser.add_argument("--input-dim", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument(
        "--intra-cluster-rounds", type=int, default=1,
        help="Client-train/cluster-aggregate cycles before each DRIFT global aggregation",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--learning-rate-min", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-rounds", type=int, default=None)
    parser.add_argument("--learning-rate-schedule-rounds", type=int, default=None)
    parser.add_argument("--cosine-annealing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--client-sample-ratio", type=float, default=1.0)
    parser.add_argument("--client-drop-rate", type=float, default=0.0)
    parser.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--partition", default="dirichlet")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--client-test-partition-mode",
        choices=("global_balanced", "match_train_distribution"),
        default="global_balanced",
        help="Opt-in client test split used by cluster-personalized evaluation",
    )
    parser.add_argument("--drift-jsd-threshold", type=float, default=0.2,
                        help="JSD acceptance threshold for drift_data_aware")
    parser.add_argument("--cluster-diagnostics-interval", type=int, default=1)
    parser.add_argument("--enable-cluster-diagnostics", action="store_true",
                        help="Record identical cluster diagnostics for the base drift control")
    parser.add_argument("--cluster-diagnostics-global-test",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-long", action="store_true",
                        help="Allow rounds > 3; required for formal runs")
    parser.add_argument("--simulation-config", type=Path, default=None)
    parser.add_argument("--target-accuracy", type=float, default=None)
    parser.add_argument(
        "--no-failure-control-scenario", default=None,
        help="Scenario name containing same-method, same-seed no-failure controls",
    )
    parser.add_argument(
        "--fedavg-edge-server-node", type=int, default=None,
        help="Emit a FedAvg-Edge system record using this frozen physical node",
    )
    args = parser.parse_args()
    if args.template_dir is None:
        args.template_dir = f"configs/{args.dataset}"
    if args.rounds <= 0 or args.num_clients <= 0:
        raise ValueError("rounds and num-clients must be positive")
    if args.cluster_diagnostics_interval <= 0:
        raise ValueError("--cluster-diagnostics-interval must be positive")
    if args.intra_cluster_rounds <= 0:
        raise ValueError("--intra-cluster-rounds must be positive")
    if args.run_smoke and args.rounds > 3 and not args.allow_long:
        raise ValueError("--run-smoke above 3 rounds requires --allow-long")
    if args.status:
        plans = load_existing_plans(args)
        write_status(plans)
        return
    if args.aggregate:
        plans = load_existing_plans(args)
        preflight(plans)
        json_path, csv_path = aggregate_matrix(
            plans, simulation_config=args.simulation_config,
            target_accuracy=args.target_accuracy,
            no_failure_control_root=(
                Path(args.output_root).resolve() / args.no_failure_control_scenario
                if args.no_failure_control_scenario else None
            ),
            fedavg_edge_server_node=args.fedavg_edge_server_node,
        )
        print(f"AGGREGATED: {json_path}")
        print(f"AGGREGATED CSV: {csv_path}")
        return
    existing_manifests = list(_root(args).glob("seed_*/manifest.json"))
    if args.preflight and existing_manifests:
        plans = load_existing_plans(args)
        _validate_existing_contract(args, plans)
    elif args.run_smoke and existing_manifests:
        plans = load_existing_plans(args)
        _validate_existing_contract(args, plans)
    elif args.generate_only and existing_manifests:
        raise FileExistsError(
            f"Scenario {_root(args)} already exists; choose a new scenario name "
            "to preserve frozen experiment inputs."
        )
    else:
        plans = prepare_plan(args)
    if args.generate_only:
        generate_frozen_inputs(plans)
        print(f"GENERATED {len(plans)} frozen scenario(s)")
        return
    generate_frozen_inputs(plans)
    preflight(plans)
    if args.preflight:
        print("ALL PREFLIGHT CHECKS PASSED")
        return
    run_methods(plans, resume=args.resume)
    write_status(plans)
    print("MATRIX RUN COMPLETE")


if __name__ == "__main__":
    main()
