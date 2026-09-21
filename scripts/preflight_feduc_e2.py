#!/usr/bin/env python3
"""Read-only FedUC-inspired assignment feasibility check for frozen E2 inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torchvision

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.feduc_objective import (
    build_reference_assignment,
    frozen_deadline_ms,
    greedy_assignment,
    label_distribution_proxy,
    local_training_times_ms,
    select_aggregators,
    shortest_latency_path,
    time_sharing_completion_ms,
)


MODEL_BYTES = 44_734_408
REFERENCE_TRAIN_MS = 0.14610237849410623


def _mapping(value: str) -> dict[int, int]:
    result = {}
    for item in value.split(","):
        seed, count = item.split(":", 1)
        result[int(seed)] = int(count)
    return result


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _edge_latency(graph, links):
    default_bandwidth = float(links["default_bandwidth_mbps"])
    default_latency = float(links["default_latency_ms"])
    overrides = links.get("edge_overrides", {})
    result = {}
    for source, values in graph.items():
        source = int(source)
        for destination in values:
            destination = int(destination)
            profile = {
                "bandwidth_mbps": default_bandwidth,
                "latency_ms": default_latency,
            }
            profile.update(overrides.get(f"{source}->{destination}", {}))
            result[(source, destination)] = (
                float(profile["latency_ms"])
                + MODEL_BYTES * 8.0 / (float(profile["bandwidth_mbps"]) * 1_000_000.0) * 1000.0
            )
    return result


def _label_counts(data_dir: Path, client_indices, num_classes=10):
    dataset = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=True, download=False, transform=None
    )
    labels = torch.tensor(dataset.targets, dtype=torch.long)
    return {
        node: torch.bincount(
            labels[torch.tensor(indices, dtype=torch.long)], minlength=num_classes
        ).tolist()
        for node, indices in enumerate(client_indices)
    }


def inspect_seed(seed_dir: Path, aggregator_count: int, data_dir: Path, slack: float):
    frozen = seed_dir / "frozen_inputs" / "config"
    partition = _load(frozen / "data_partition.json")
    graph = _load(frozen / "physical_graph.json")
    resources = _load(frozen / "node_resources.json")
    links = _load(frozen / "link_profiles.json")
    workers = list(range(len(partition["client_indices"])))
    aggregators = sorted(select_aggregators(resources, aggregator_count))
    edge_latency = _edge_latency(graph, links)
    paths, upload = {}, {}
    normalized_graph = {int(node): [int(value) for value in values]
                        for node, values in graph.items()}
    for worker in workers:
        for aggregator in aggregators:
            path, latency = shortest_latency_path(
                normalized_graph, worker, aggregator, edge_latency
            )
            paths[(worker, aggregator)] = path
            upload[(worker, aggregator)] = latency
    samples = {node: int(value) for node, value in enumerate(partition["sample_counts"])}
    train = local_training_times_ms(
        samples, resources, local_epochs=2,
        reference_ms_per_sample_epoch=REFERENCE_TRAIN_MS,
    )
    reference = build_reference_assignment(workers, aggregators, upload)
    deadline = frozen_deadline_ms(reference, train, upload, slack=slack)
    labels = _label_counts(data_dir, partition["client_indices"])
    groups, decisions = greedy_assignment(
        workers, aggregators, samples, train, upload, deadline,
        lambda value: label_distribution_proxy(value, labels),
    )
    return {
        "seed": int(seed_dir.name.removeprefix("seed_")),
        "objective_mode": "label_distribution_proxy",
        "reporting_name": "FedUC-inspired-CenSyn-I5",
        "num_aggregators": len(aggregators),
        "aggregator_ids": aggregators,
        "deadline_ms": deadline,
        "reference_max_completion_ms": max(
            time_sharing_completion_ms(members, head, train, upload)
            for head, members in reference.items()
        ),
        "final_max_completion_ms": max(
            time_sharing_completion_ms(members, head, train, upload)
            for head, members in groups.items()
        ),
        "cluster_sizes": {str(head): len(members) for head, members in groups.items()},
        "objective": label_distribution_proxy(groups, labels),
        "assignment": {str(head): members for head, members in groups.items()},
        "decision_count": len(decisions),
        "covered_workers": len({node for members in groups.values() for node in members}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--e2-root", type=Path,
        default=REPO_ROOT / "outputs/paper_results_archive_v2/E2_static_tradeoff/source_data",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("/home/yjq/data"))
    parser.add_argument("--aggregators", default="42:9,43:3,44:8")
    parser.add_argument("--deadline-slack", type=float, default=1.25)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    counts = _mapping(args.aggregators)
    records = [
        inspect_seed(args.e2_root / f"seed_{seed}", count, args.data_dir, args.deadline_slack)
        for seed, count in sorted(counts.items())
    ]
    text = json.dumps({"status": "PASS", "records": records}, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
