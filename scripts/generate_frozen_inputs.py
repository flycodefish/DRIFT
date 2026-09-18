"""Generate a canonical input bundle once, before running any method."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deasfl.config import NewFLConfig
from deasfl.runtime import NewFLRuntime
from fl_core.artifacts import (
    generate_failure_trace, generate_link_profiles, generate_node_resources,
    normalize_physical_graph, write_experiment_inputs,
)
from fl_core.data_loader import (
    build_distribution_matched_test_indices, har_dataset_fingerprint,
    load_cifar10_data, load_har_data,
    resolve_dataset_split_policy, tensor_fingerprint,
)


def validate_scenario_spec(scenario: dict) -> None:
    """Reject misspelled scenario fields before creating immutable inputs."""
    allowed_top = {"name", "purpose", "resource_profile", "network_profile", "failure_profile"}
    unknown = sorted(set(scenario) - allowed_top)
    if unknown:
        raise ValueError(f"Unknown scenario fields: {unknown}")
    allowed_profiles = {
        "resource_profile": {
            "mode", "capacity", "tiers", "reference_cpu_cores",
            "fixed_memory_gb", "fixed_bandwidth_mbps", "fixed_gpu_memory_gb",
            "memory_gb", "bandwidth_mbps", "gpu_memory_gb",
        },
        "network_profile": {
            "mode", "default_bandwidth_mbps", "default_latency_ms", "drop_rate",
            "slow_edge_fraction", "slow_bandwidth_mbps", "slow_latency_ms",
            "slow_drop_rate",
        },
        "failure_profile": {
            "mode", "failure_round", "recovery_round", "selection_rule",
            "failure_count", "max_selection_attempts",
        },
    }
    for key, allowed in allowed_profiles.items():
        profile = scenario.get(key, {})
        if not isinstance(profile, dict):
            raise ValueError(f"{key} must be a JSON object")
        unknown = sorted(set(profile) - allowed)
        if unknown:
            raise ValueError(f"Unknown {key} fields: {unknown}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Dataset root directory (overrides YAML data_dir)")
    parser.add_argument("--scenario-spec", type=Path, default=None,
                        help="Optional frozen resource/network/failure scenario JSON")
    args = parser.parse_args()
    scenario = {}
    if args.scenario_spec:
        scenario = json.loads(args.scenario_spec.read_text(encoding="utf-8"))
        if not isinstance(scenario, dict):
            raise ValueError("scenario spec must be a JSON object")
        validate_scenario_spec(scenario)
    cfg = NewFLConfig.from_yaml(args.config)
    if args.data_dir:
        cfg.data_dir = args.data_dir
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    loader = load_har_data if cfg.dataset == "har" else load_cifar10_data
    if cfg.dataset == "har":
        har_train_data, har_val_data, indices, global_test_data, _ = loader(
            data_dir=cfg.data_dir, num_clients=cfg.num_clients,
            partition_mode=cfg.partition, alpha=cfg.alpha, seed=cfg.seed,
            split_policy=resolve_dataset_split_policy(cfg.dataset, cfg.dataset_split_policy),
            local_val_seed=cfg.har_local_val_seed,
        )
    else:
        cifar_train_data, _, indices, global_test_data, _ = loader(
            data_dir=cfg.data_dir, num_clients=cfg.num_clients,
            partition_mode=cfg.partition, alpha=cfg.alpha, seed=cfg.seed,
        )
    runtime = NewFLRuntime(cfg)
    graph = runtime._build_topology(
        cfg.num_clients, cfg.topology_type, cfg.neighbor_selection,
        cfg.min_neighbors, cfg.max_neighbors,
    )
    graph = normalize_physical_graph(graph, cfg.num_clients)
    torch.manual_seed(cfg.seed)
    initial_model = runtime._create_model().state_dict()
    resources = generate_node_resources(
        cfg.num_clients, cfg.seed, scenario.get("resource_profile")
    )
    partition_metadata = {
        "dataset": cfg.dataset,
            "split_policy": resolve_dataset_split_policy(
                cfg.dataset, getattr(cfg, "dataset_split_policy", None)
            ),
            "local_val_seed": getattr(cfg, "har_local_val_seed", 42),
    }
    if cfg.dataset == "har":
        partition_metadata.update({
            "partition_train_sample_count": sum(len(shard) for shard in indices),
            "official_train_sample_count": (
                sum(len(shard) for shard in indices) +
                sum(len(item["labels"]) for item in har_val_data)
            ),
            "local_validation_sample_count": sum(len(item["labels"]) for item in har_val_data),
            "partition_index_space_size": sum(len(shard) for shard in indices),
            "global_test_sample_count": len(global_test_data["labels"]),
            "num_classes": cfg.num_classes,
            "preprocessing": "official_train_mean_std_v1",
            "dataset_fingerprint": har_dataset_fingerprint(cfg.data_dir),
            "global_test_fingerprint": tensor_fingerprint(
                global_test_data["images"], global_test_data["labels"]
            ),
        })
    elif cfg.dataset == "cifar10" and getattr(
        cfg, "client_test_partition_mode", "global_balanced"
    ) == "match_train_distribution":
        train_labels = torch.empty(50_000, dtype=torch.long)
        for shard, client in zip(indices, cifar_train_data):
            train_labels[torch.as_tensor(shard, dtype=torch.long)] = client["labels"]
        client_test_indices = build_distribution_matched_test_indices(
            train_labels=train_labels,
            client_indices=indices,
            test_labels=global_test_data["labels"],
            num_classes=cfg.num_classes,
            seed=cfg.seed,
        )
        partition_metadata.update({
            "client_test_partition_mode": "match_train_distribution",
            "client_test_indices": client_test_indices,
            "client_test_sample_counts": [len(shard) for shard in client_test_indices],
            "global_test_sample_count": len(global_test_data["labels"]),
        })
    write_experiment_inputs(
        output_dir=args.output, config=cfg, client_indices=indices,
        initial_model=initial_model, physical_graph=graph,
        node_resources=resources,
        link_profiles=generate_link_profiles(
            graph,
            scenario.get("network_profile", {
                "mode": "homogeneous", "drop_rate": cfg.drop_rate,
            }),
            cfg.seed,
        ),
        failure_trace=generate_failure_trace(
            graph, resources, cfg.rounds, scenario.get("failure_profile"), seed=cfg.seed
        ),
        partition_metadata=partition_metadata,
    )
    print(f"Frozen inputs: {Path(args.output) / 'config'}")


if __name__ == "__main__":
    main()
