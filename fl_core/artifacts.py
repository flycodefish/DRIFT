"""Shared, algorithm-neutral experiment artifact helpers."""

from __future__ import annotations

import json
import platform
import random
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

try:
    import ray
except ImportError:  # pragma: no cover - Ray is optional for artifact inspection.
    ray = None


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return value


def _git_value(args: list[str], default: str = "unknown") -> str:
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def normalize_physical_graph(graph: Any, num_clients: int) -> dict[int, list[int]]:
    """Return one connected-semantics, undirected adjacency representation.

    Physical links are bidirectional in Protocol V1.  Closing the adjacency
    list here gives DRIFT and every baseline exactly the same edge set.
    """
    if isinstance(graph, list):
        source = {index: values for index, values in enumerate(graph)}
    elif isinstance(graph, dict):
        source = {int(key): values for key, values in graph.items()}
    else:
        raise ValueError("physical graph must be a list or mapping")
    result = {node: set() for node in range(num_clients)}
    for src, values in source.items():
        if src not in result:
            raise ValueError(f"physical graph contains invalid node {src}")
        for raw_dst in values:
            dst = int(raw_dst)
            if dst not in result:
                raise ValueError(f"physical graph contains invalid neighbor {dst}")
            if dst == src:
                continue
            result[src].add(dst)
            result[dst].add(src)
    return {node: sorted(values) for node, values in result.items()}


def write_experiment_inputs(
    output_dir: str,
    config: Any,
    client_indices: list[list[int]],
    initial_model: dict[str, torch.Tensor],
    physical_graph: Any,
    node_resources: Any,
    link_profiles: Any,
    failure_trace: Any,
    partition_metadata: dict[str, Any] | None = None,
) -> None:
    """Persist the immutable inputs required to reproduce a run."""
    config_dir = Path(output_dir) / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    write_json(config_dir / "experiment.json", config)
    partition_payload = {
        "schema_version": 2,
        "dataset": getattr(config, "dataset", "unknown"),
        "split_policy": getattr(config, "dataset_split_policy", "default"),
        "num_classes": getattr(config, "num_classes", None),
        "client_indices": client_indices,
        "sample_counts": [len(indices) for indices in client_indices],
        "partition_train_sample_count": sum(len(indices) for indices in client_indices),
    }
    if partition_metadata:
        partition_payload.update(partition_metadata)
    write_json(
        config_dir / "data_partition.json",
        partition_payload,
    )
    write_json(
        config_dir / "physical_graph.json",
        normalize_physical_graph(physical_graph, len(client_indices)),
    )
    write_json(config_dir / "node_resources.json", node_resources)
    write_json(config_dir / "link_profiles.json", link_profiles)
    write_json(config_dir / "failure_trace.json", failure_trace)
    torch.save(
        {key: value.detach().cpu().clone() for key, value in initial_model.items()},
        config_dir / "initial_model.pt",
    )

    dirty = bool(_git_value(["status", "--porcelain"], default=""))
    write_json(
        config_dir / "manifest.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_value(["rev-parse", "HEAD"]),
            "git_dirty": dirty,
            "python_version": sys.version,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "ray_version": getattr(ray, "__version__", None),
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ] if torch.cuda.is_available() else [],
        },
    )


def write_completion_status(
    output_dir: str,
    requested_rounds: int,
    completed_rounds: int,
    status: str,
    error: str | None = None,
) -> None:
    write_json(
        Path(output_dir) / "config" / "completion_status.json",
        {
            "status": status,
            "requested_rounds": requested_rounds,
            "completed_rounds": completed_rounds,
            "error": error,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


FROZEN_INPUT_FILES = (
    "data_partition.json", "initial_model.pt", "physical_graph.json",
    "node_resources.json", "link_profiles.json", "failure_trace.json",
)


def generate_node_resources(
    num_clients: int, seed: int, profile: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Generate a reproducible node table, optionally from a frozen profile.

    ``profile`` is an experiment-layer input.  The default keeps the historical
    seeded generator unchanged; tiered profiles make the formal resource
    heterogeneity explicit and method-independent.
    """
    if profile:
        mode = str(profile.get("mode", "seeded_random")).lower()
        if mode == "homogeneous":
            capacity = float(profile.get("capacity", 1.0))
            if capacity <= 0.0:
                raise ValueError("homogeneous capacity must be positive")
            capacities = [capacity] * num_clients
        elif mode == "tiers":
            tiers = profile.get("tiers", [])
            if not isinstance(tiers, list) or not tiers:
                raise ValueError("tiered resource profile requires non-empty tiers")
            fractions = [float(item.get("fraction", 0.0)) for item in tiers]
            if any(fraction < 0.0 for fraction in fractions):
                raise ValueError("tier fractions must be non-negative")
            if abs(sum(fractions) - 1.0) > 1e-9:
                raise ValueError("tier fractions must sum to 1.0")
            if any(float(item.get("capacity", 0.0)) <= 0.0 for item in tiers):
                raise ValueError("tier capacities must be positive")
            raw_counts = [float(item.get("fraction", 0.0)) * num_clients for item in tiers]
            counts = [int(value) for value in raw_counts]
            for index in sorted(
                range(len(counts)),
                key=lambda i: (raw_counts[i] - counts[i], -i),
                reverse=True,
            ):
                if sum(counts) >= num_clients:
                    break
                counts[index] += 1
            while sum(counts) > num_clients:
                index = max(range(len(counts)), key=lambda i: (counts[i], -i))
                counts[index] -= 1
            capacities = []
            for item, count in zip(tiers, counts):
                capacities.extend([float(item["capacity"])] * count)
            rng = random.Random(seed)
            rng.shuffle(capacities)
        else:
            raise ValueError(f"Unsupported resource profile mode: {mode}")
        reference = float(profile.get("reference_cpu_cores", 16.0))
        if reference <= 0.0:
            raise ValueError("reference_cpu_cores must be positive")
        fixed_memory = float(profile.get("fixed_memory_gb", profile.get("memory_gb", 32.0)))
        fixed_bandwidth = float(profile.get("fixed_bandwidth_mbps", profile.get("bandwidth_mbps", 1000.0)))
        fixed_gpu = float(profile.get("fixed_gpu_memory_gb", profile.get("gpu_memory_gb", 0.0)))
        if fixed_memory <= 0.0 or fixed_bandwidth <= 0.0 or fixed_gpu < 0.0:
            raise ValueError("fixed memory/bandwidth must be positive and GPU memory non-negative")
        return [{
            "node_id": node_id,
            "cpu_cores": max(1, int(round(capacity * reference))),
            # Compute tiers intentionally do not alter network or memory
            # capacity.  This keeps the E3 compute-only ablation valid.
            "memory_gb": max(1, int(round(fixed_memory))),
            "bandwidth_mbps": max(1, int(round(fixed_bandwidth))),
            "gpu_memory_gb": max(0, int(round(fixed_gpu))),
            "compute_capacity": capacity,
            "compute_capacity_semantics": "relative_throughput",
            "compute_capacity_reference_cpu_cores": reference,
        } for node_id, capacity in enumerate(capacities)]

    resources = []
    for node_id in range(num_clients):
        rng = random.Random(seed + node_id)
        cpu_cores = rng.randint(2, 16)
        memory_gb = rng.randint(4, 64)
        bandwidth_mbps = rng.randint(50, 1000)
        gpu_memory_gb = rng.choice([0, 4, 8, 16, 32])
        resources.append({
            "node_id": node_id, "cpu_cores": cpu_cores,
            "memory_gb": memory_gb, "bandwidth_mbps": bandwidth_mbps,
            "gpu_memory_gb": gpu_memory_gb,
            # Scenario-level relative throughput.  The 16-core tier is the
            # reference (1.0); this is not a claim about GPU/CPU equivalence.
            "compute_capacity": float(cpu_cores) / 16.0,
            "compute_capacity_semantics": "relative_throughput",
            "compute_capacity_reference_cpu_cores": 16.0,
        })
    return resources


def deterministic_resource_score(resource: dict[str, Any]) -> float:
    """Return the deterministic part of DRIFT's resource score.

    DRIFT's election class adds a small runtime tie-breaking noise term.  A
    frozen failure target cannot depend on that process-local random draw, so
    scenario generation uses the documented deterministic resource terms and
    breaks exact ties by the lowest node id.
    """
    return (
        float(resource.get("cpu_cores", 4)) * 3.0
        + float(resource.get("memory_gb", 8)) * 0.8
        + float(resource.get("bandwidth_mbps", 100)) * 0.02
        + float(resource.get("gpu_memory_gb", 0)) * 1.5
    )


def _physical_edges(graph: dict[int, list[int]]) -> list[tuple[int, int]]:
    return sorted({
        tuple(sorted((int(src), int(dst))))
        for src, values in graph.items()
        for dst in values
        if int(src) != int(dst)
    })


def edge_betweenness_scores(graph: dict[int, list[int]]) -> dict[tuple[int, int], float]:
    """Compute deterministic undirected edge betweenness without extra deps."""
    adjacency = {int(node): sorted(int(neighbor) for neighbor in values)
                 for node, values in graph.items()}
    scores = {edge: 0.0 for edge in _physical_edges(graph)}
    for source in sorted(adjacency):
        stack: list[int] = []
        predecessors = {node: [] for node in adjacency}
        sigma = {node: 0.0 for node in adjacency}
        distance = {node: -1 for node in adjacency}
        sigma[source] = 1.0
        distance[source] = 0
        queue = [source]
        while queue:
            node = queue.pop(0)
            stack.append(node)
            for neighbor in adjacency[node]:
                if distance[neighbor] < 0:
                    distance[neighbor] = distance[node] + 1
                    queue.append(neighbor)
                if distance[neighbor] == distance[node] + 1:
                    sigma[neighbor] += sigma[node]
                    predecessors[neighbor].append(node)
        dependency = {node: 0.0 for node in adjacency}
        while stack:
            node = stack.pop()
            for predecessor in predecessors[node]:
                contribution = (sigma[predecessor] / sigma[node]) * (1.0 + dependency[node])
                edge = tuple(sorted((predecessor, node)))
                scores[edge] += contribution
                dependency[predecessor] += contribution
    return {edge: value / 2.0 for edge, value in scores.items()}


def generate_link_profiles(
    graph: dict[int, list[int]], profile: dict[str, Any] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Create one explicit, shared link profile for a physical graph."""
    profile = profile or {}
    result = {
        "default_bandwidth_mbps": float(profile.get("default_bandwidth_mbps", 100.0)),
        "default_latency_ms": float(profile.get("default_latency_ms", 5.0)),
        "drop_rate": float(profile.get("drop_rate", 0.0)),
        "edge_overrides": {},
    }
    if result["default_bandwidth_mbps"] <= 0 or result["default_latency_ms"] < 0:
        raise ValueError("default link bandwidth must be positive and latency non-negative")
    if not 0.0 <= result["drop_rate"] <= 1.0:
        raise ValueError("link drop_rate must be between 0 and 1")
    mode = str(profile.get("mode", "homogeneous")).lower()
    if mode == "homogeneous":
        return result
    if mode != "medium_heterogeneous":
        raise ValueError(f"Unsupported link profile mode: {mode}")
    slow_fraction = float(profile.get("slow_edge_fraction", 0.2))
    slow_bandwidth = float(profile.get("slow_bandwidth_mbps", 20.0))
    slow_latency = float(profile.get("slow_latency_ms", 20.0))
    slow_drop_rate = float(profile.get("slow_drop_rate", 0.0))
    if not 0.0 <= slow_fraction <= 1.0:
        raise ValueError("slow_edge_fraction must be between 0 and 1")
    if slow_bandwidth <= 0 or slow_latency < 0:
        raise ValueError("slow link bandwidth must be positive and latency non-negative")
    if not 0.0 <= slow_drop_rate <= 1.0:
        raise ValueError("slow_drop_rate must be between 0 and 1")
    edges = sorted({tuple(sorted((int(src), int(dst))))
                    for src, values in graph.items() for dst in values if src != dst})
    rng = random.Random(seed)
    rng.shuffle(edges)
    slow_count = min(len(edges), int(round(len(edges) * slow_fraction)))
    for left, right in edges[:slow_count]:
        result["edge_overrides"][f"{left}->{right}"] = {
            "bandwidth_mbps": slow_bandwidth,
            "latency_ms": slow_latency,
            "drop_rate": slow_drop_rate,
        }
    return result


def generate_failure_trace(
    graph: dict[int, list[int]], resources: list[dict[str, Any]],
    rounds: int, profile: dict[str, Any] | None = None, seed: int = 0,
) -> dict[str, Any]:
    """Generate a deterministic failure/recovery trace for a scenario."""
    profile = profile or {"mode": "none"}
    mode = str(profile.get("mode", "none")).lower()
    if mode == "none":
        return {"schema_version": 1, "events": []}
    if rounds <= 0:
        raise ValueError("failure trace requires a positive round count")
    events: list[dict[str, Any]] = []
    failure_round = int(profile.get("failure_round", rounds // 3))
    if not 0 <= failure_round < rounds:
        raise ValueError(
            f"failure_round {failure_round} must be within the configured 0..{rounds - 1} range"
        )
    if mode == "node_failure":
        selection_rule = str(
            profile.get("selection_rule", "deterministic_resource_score")
        ).lower()
        if selection_rule == "uniform_random_connected":
            node_count = len(graph)
            failure_count = int(profile.get("failure_count", 1))
            if not 0 < failure_count < node_count:
                raise ValueError("failure_count must be between 1 and num_clients - 1")
            max_attempts = int(profile.get("max_selection_attempts", 10000))
            if max_attempts <= 0:
                raise ValueError("max_selection_attempts must be positive")
            rng = random.Random(seed)
            nodes = sorted(graph)
            targets = None
            attempts = 0
            while attempts < max_attempts:
                attempts += 1
                candidate = sorted(rng.sample(nodes, failure_count))
                remaining = set(nodes) - set(candidate)
                reached = set()
                frontier = [min(remaining)]
                while frontier:
                    node = frontier.pop()
                    if node in reached:
                        continue
                    reached.add(node)
                    frontier.extend(
                        neighbor for neighbor in graph[node]
                        if neighbor in remaining and neighbor not in reached
                    )
                if reached == remaining:
                    targets = candidate
                    break
            if targets is None:
                raise ValueError(
                    f"Could not select {failure_count} nodes while preserving connectivity "
                    f"after {max_attempts} attempts"
                )
            target_record = None
        elif selection_rule == "deterministic_resource_score":
            target_record = max(
                resources,
                key=lambda item: (deterministic_resource_score(item), -int(item["node_id"])),
            )
            targets = [int(target_record["node_id"])]
            attempts = 1
        else:
            raise ValueError(f"Unsupported node failure selection_rule: {selection_rule}")
        events.append({"round": failure_round, "event_type": "node_failure", "node_ids": targets})
        recovery_round = int(profile.get("recovery_round", failure_round + 2))
        if recovery_round < 0 or recovery_round >= rounds:
            raise ValueError(
                f"recovery_round {recovery_round} must be within the configured 0..{rounds - 1} range"
            )
        if recovery_round <= failure_round:
            raise ValueError("recovery_round must be greater than failure_round")
        events.append({"round": recovery_round, "event_type": "node_recovery", "node_ids": targets})
    elif mode == "link_failure":
        edges = _physical_edges(graph)
        if not edges:
            raise ValueError("link failure profile requires at least one physical edge")
        scores = edge_betweenness_scores(graph)
        edge = max(edges, key=lambda candidate: (scores[candidate], tuple(-part for part in candidate)))
        events.append({"round": failure_round, "event_type": "link_failure", "edges": [list(edge)]})
        recovery_round = int(profile.get("recovery_round", failure_round + 2))
        if recovery_round < 0 or recovery_round >= rounds:
            raise ValueError(
                f"recovery_round {recovery_round} must be within the configured 0..{rounds - 1} range"
            )
        if recovery_round <= failure_round:
            raise ValueError("recovery_round must be greater than failure_round")
        events.append({"round": recovery_round, "event_type": "link_recovery", "edges": [list(edge)]})
    else:
        raise ValueError(f"Unsupported failure profile mode: {mode}")
    target_event = events[0]
    selection = {
        "rule": (selection_rule if mode == "node_failure" else "maximum_edge_betweenness"),
        "tie_break": (
            "sorted_node_ids" if mode == "node_failure" and target_record is None
            else "lowest_node_id" if mode == "node_failure"
            else "lexicographically_smallest_edge"
        ),
        "target": (target_event.get("node_ids") or target_event.get("edges")),
    }
    if mode == "node_failure":
        if target_record is not None:
            selection["score"] = deterministic_resource_score(target_record)
        else:
            selection.update({
                "seed": int(seed),
                "attempts": attempts,
                "remaining_connected": True,
                "remaining_node_count": len(graph) - len(targets),
            })
    else:
        selection["score"] = scores[edge]
    return {"schema_version": 1, "selection": selection, "events": events}


def failure_state_at_round(
    bundle: dict[str, Any] | None, round_id: int,
) -> tuple[set[int], set[tuple[int, int]]]:
    """Replay a frozen trace through ``round_id`` and return persistent state."""
    if not bundle:
        return set(), set()
    trace = bundle.get("failure_trace.json", {})
    events = trace if isinstance(trace, list) else trace.get("events", [])
    failed_nodes: set[int] = set()
    failed_edges: set[tuple[int, int]] = set()
    for event in sorted(events, key=lambda item: int(item.get("round", -1))):
        if int(event.get("round", -1)) > round_id:
            break
        event_type = str(event.get("event_type", "")).strip().lower()
        if event_type in {"node_failure", "node_recovery", "node_join"}:
            raw = event.get("node_ids", event.get("targets", event.get("target", [])))
            if not isinstance(raw, list):
                raw = [raw]
            nodes = {int(node) for node in raw}
            if event_type == "node_failure":
                failed_nodes.update(nodes)
            else:
                failed_nodes.difference_update(nodes)
        elif event_type in {"link_failure", "link_recovery"}:
            raw_edges = event.get("edges", event.get("targets", event.get("target", [])))
            if raw_edges and isinstance(raw_edges[0], int):
                raw_edges = [raw_edges]
            edges = {tuple(sorted((int(item[0]), int(item[1])))) for item in raw_edges}
            if event_type == "link_failure":
                failed_edges.update(edges)
            else:
                failed_edges.difference_update(edges)
    return failed_nodes, failed_edges


def load_frozen_inputs(input_dir: str) -> dict[str, Any]:
    """Load one immutable input bundle shared by every compared method."""
    root = Path(input_dir)
    missing = [name for name in FROZEN_INPUT_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Frozen input bundle is incomplete: {missing}")
    result = {}
    for name in FROZEN_INPUT_FILES:
        path = root / name
        if name == "initial_model.pt":
            result[name] = torch.load(path, map_location="cpu", weights_only=True)
        else:
            result[name] = json.loads(path.read_text(encoding="utf-8"))
    result["physical_graph.json"] = normalize_physical_graph(
        result["physical_graph.json"],
        len(result["data_partition.json"].get("client_indices", [])),
    )
    return result


def validate_frozen_inputs(bundle: dict[str, Any], num_clients: int,
                           expected_dataset: str | None = None,
                           expected_split_policy: str | None = None,
                           expected_local_val_seed: int | None = None,
                           expected_dataset_fingerprint: str | None = None,
                           strict: bool = False) -> None:
    indices = bundle["data_partition.json"].get("client_indices", [])
    partition = bundle["data_partition.json"]
    if strict:
        required = {
            "schema_version", "dataset", "split_policy", "num_classes",
            "client_indices", "sample_counts", "partition_train_sample_count",
        }
        missing = sorted(key for key in required if key not in partition)
        if missing:
            raise ValueError(f"Strict frozen partition is missing fields: {missing}")
        null_fields = sorted(key for key in required if partition.get(key) is None)
        if null_fields:
            raise ValueError(f"Strict frozen partition has null fields: {null_fields}")
        if int(partition.get("schema_version", 0)) < 2:
            raise ValueError("Strict frozen inputs require data_partition schema_version >= 2")
        if expected_dataset == "har":
            har_required = {
                "local_val_seed", "official_train_sample_count",
                "local_validation_sample_count", "global_test_sample_count",
                "partition_index_space_size", "dataset_fingerprint",
                "global_test_fingerprint",
            }
            missing = sorted(key for key in har_required if key not in partition)
            if missing:
                raise ValueError(f"Strict HAR partition is missing fields: {missing}")
            null_fields = sorted(key for key in har_required if partition.get(key) is None)
            if null_fields:
                raise ValueError(f"Strict HAR partition has null fields: {null_fields}")
    if expected_dataset and (
            partition.get("dataset") != expected_dataset
            if strict else partition.get("dataset") not in (None, "unknown", expected_dataset)
    ):
        raise ValueError(
            f"Frozen partition dataset {partition.get('dataset')!r} does not match {expected_dataset!r}"
        )
    if expected_split_policy and (
            partition.get("split_policy") != expected_split_policy
            if strict else partition.get("split_policy") not in (None, "default", expected_split_policy)
    ):
        raise ValueError(
            f"Frozen partition split policy {partition.get('split_policy')!r} does not match "
            f"{expected_split_policy!r}"
        )
    if (expected_local_val_seed is not None and
            (partition.get("local_val_seed") != expected_local_val_seed
             if strict else partition.get("local_val_seed") not in (None, expected_local_val_seed))):
        raise ValueError(
            f"Frozen partition local validation seed {partition.get('local_val_seed')!r} does not match "
            f"{expected_local_val_seed!r}"
        )
    if (expected_dataset_fingerprint and
            (partition.get("dataset_fingerprint") != expected_dataset_fingerprint
             if strict else partition.get("dataset_fingerprint") not in (None, expected_dataset_fingerprint))):
        raise ValueError("Frozen partition dataset fingerprint does not match the installed dataset")
    if len(indices) != num_clients:
        raise ValueError(
            f"Frozen partition has {len(indices)} clients, expected {num_clients}"
        )
    flattened = [int(index) for shard in indices for index in shard]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Frozen client partitions contain duplicate indices")
    sample_counts = partition.get("sample_counts")
    if sample_counts is not None and list(sample_counts) != [len(shard) for shard in indices]:
        raise ValueError("Frozen sample_counts do not match client_indices")
    partition_count = partition.get("partition_train_sample_count")
    if partition_count is not None and int(partition_count) != len(flattened):
        raise ValueError("Frozen partition_train_sample_count does not match client_indices")
    if strict:
        index_space_size = partition.get("partition_index_space_size")
        if index_space_size is None and expected_dataset == "cifar10":
            index_space_size = 50_000
        if index_space_size is None:
            raise ValueError("Strict frozen partition requires partition_index_space_size")
        index_space_size = int(index_space_size)
        if len(flattened) != index_space_size or set(flattened) != set(range(index_space_size)):
            raise ValueError("Frozen client partitions do not cover the complete index space")
        if expected_dataset == "har":
            official_train = int(partition["official_train_sample_count"])
            local_validation = int(partition["local_validation_sample_count"])
            if official_train != index_space_size + local_validation:
                raise ValueError("Strict HAR sample-count metadata is inconsistent")
        # HARMLP uses BatchNorm1d and therefore needs this constraint.  Other
        # models (for example a linear/synthetic smoke model) can validly
        # train on a one-sample shard, so the artifact contract must not
        # impose a model-specific restriction globally.
        if expected_dataset == "har" and any(len(shard) < 2 for shard in indices):
            raise ValueError("Strict HAR training partitions require at least two samples per client")
    test_indices = partition.get("client_test_indices")
    if test_indices is not None:
        if len(test_indices) != num_clients:
            raise ValueError("Frozen client test partition count does not match num_clients")
        flattened_test = [int(index) for shard in test_indices for index in shard]
        expected_test_count = int(partition.get("global_test_sample_count", len(flattened_test)))
        if (len(flattened_test) != expected_test_count
                or len(set(flattened_test)) != expected_test_count
                or set(flattened_test) != set(range(expected_test_count))):
            raise ValueError("Frozen client test partitions must cover the test index space once")
        test_counts = partition.get("client_test_sample_counts")
        if test_counts is not None and list(test_counts) != [len(shard) for shard in test_indices]:
            raise ValueError("Frozen client_test_sample_counts do not match client_test_indices")
    resources = bundle["node_resources.json"]
    if isinstance(resources, list) and len(resources) not in (0, num_clients):
        raise ValueError(
            f"Frozen resources have {len(resources)} nodes, expected {num_clients}"
        )
    if strict:
        if not isinstance(resources, list) or len(resources) != num_clients:
            raise ValueError("Strict frozen inputs require one resource record per client")
        resource_ids = [item.get("node_id") for item in resources if isinstance(item, dict)]
        if len(resource_ids) != num_clients or set(resource_ids) != set(range(num_clients)):
            raise ValueError("Strict frozen resources must contain unique node_id values 0..num_clients-1")
        links = bundle["link_profiles.json"]
        if not isinstance(links, dict):
            raise ValueError("Strict frozen link profiles must be a mapping")
        for key in ("default_bandwidth_mbps", "default_latency_ms", "drop_rate"):
            if key in links and not isinstance(links[key], (int, float)):
                raise ValueError(f"Strict frozen link profile field {key!r} must be numeric")
        overrides = links.get("edge_overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError("Strict frozen edge_overrides must be a mapping")
        for edge_key, value in overrides.items():
            if not isinstance(edge_key, str) or "->" not in edge_key:
                raise ValueError(f"Strict frozen edge override has invalid key: {edge_key!r}")
            if not isinstance(value, dict):
                raise ValueError(f"Strict frozen edge override must be a mapping: {edge_key!r}")
        trace = bundle["failure_trace.json"]
        events = trace if isinstance(trace, list) else trace.get("events") if isinstance(trace, dict) else None
        if events is None or not isinstance(events, list):
            raise ValueError("Strict frozen failure trace must contain an events list")
        for event in events:
            if not isinstance(event, dict) or "event_type" not in event:
                raise ValueError("Strict frozen failure events require event_type")
            if "round" in event:
                try:
                    if int(event["round"]) < 0:
                        raise ValueError("negative round")
                except (TypeError, ValueError) as exc:
                    raise ValueError("Strict frozen failure event round must be a non-negative integer") from exc
    graph = normalize_physical_graph(bundle["physical_graph.json"], num_clients)
    if strict:
        physical_edges = {
            f"{min(src, dst)}->{max(src, dst)}"
            for src, values in graph.items() for dst in values if src != dst
        }
        for edge_key in bundle["link_profiles.json"].get("edge_overrides", {}):
            if edge_key not in physical_edges:
                raise ValueError(f"Strict frozen edge override is not a physical edge: {edge_key!r}")
    for node, neighbors in graph.items():
        if any(node not in graph[neighbor] for neighbor in neighbors):
            raise ValueError("Frozen physical graph must be undirected")
    if strict:
        reached = set()
        frontier = [0] if num_clients else []
        while frontier:
            node = frontier.pop()
            if node in reached:
                continue
            reached.add(node)
            frontier.extend(neighbor for neighbor in graph[node] if neighbor not in reached)
        if len(reached) != num_clients:
            raise ValueError("Strict frozen physical graph must be connected")


def frozen_failure_events(bundle: dict[str, Any] | None, round_id: int,
                          event_type: str) -> list[dict[str, Any]]:
    if not bundle:
        return []
    trace = bundle.get("failure_trace.json", {})
    events = trace if isinstance(trace, list) else trace.get("events", [])
    return [event for event in events
            if int(event.get("round", -1)) == round_id
            and str(event.get("event_type", "")).lower() == event_type.lower()]
