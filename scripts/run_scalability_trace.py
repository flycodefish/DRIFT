"""Generate training-free FedAvg-Edge, Resource-Tree and DRIFT traces."""

from __future__ import annotations

import argparse
import math
import json
import shutil
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fl_core.scalability_trace import drift_trace, fedavg_edge_trace, resource_tree_trace
from fl_core.simulation import simulate_round
from fl_core.system_metrics import (
    build_node_round_stats, capacity_normalized_load_metrics, dump_system_metrics,
)


def _load(path: Path, name: str):
    return json.loads((path / name).read_text(encoding="utf-8"))


def _model_bytes(path: Path) -> int:
    state = torch.load(path / "initial_model.pt", map_location="cpu", weights_only=True)
    return sum(value.numel() * value.element_size() for value in state.values())


def _assumptions(path: Path, resources: list[dict]) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if str(result.get("status", "")).lower() != "frozen":
        raise ValueError("simulation configuration must have status=frozen")
    capacities, network = {}, {}
    for position, item in enumerate(resources):
        params = item.get("resource_params", item); node = str(item.get("node_id", position))
        capacities[node] = float(params["compute_capacity"])
        network[node] = float(params["bandwidth_mbps"])
    result.update(compute_capacity_by_node=capacities,
                  aggregation_capacity_by_node=dict(capacities),
                  network_capacity_by_node=network)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--simulation-config", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--methods", default="fedavg_edge,resource_tree,drift")
    args = parser.parse_args()
    if args.rounds < 1 or args.rounds > 3:
        raise ValueError("scalability trace rounds must be in [1, 3]")
    input_dir = args.input_dir.resolve()
    experiment = _load(input_dir, "experiment.json")
    graph = {int(k): list(v) for k, v in _load(input_dir, "physical_graph.json").items()}
    resources, links = _load(input_dir, "node_resources.json"), _load(input_dir, "link_profiles.json")
    partition = _load(input_dir, "data_partition.json")
    model_bytes, seed = _model_bytes(input_dir), int(experiment["seed"])
    assumptions = _assumptions(args.simulation_config.resolve(), resources)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = sorted(set(methods) - {"fedavg_edge", "resource_tree", "drift"})
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")
    summaries = {}
    for method in methods:
        run = args.output_dir.resolve() / f"seed_{seed}" / method
        run.mkdir(parents=True, exist_ok=True)
        exp_id = f"scalability_n{len(graph)}_seed{seed}_{method}"
        metadata = {}
        if method == "fedavg_edge":
            events, contexts, server = fedavg_edge_trace(
                graph, resources, links, model_bytes, seed, args.rounds, exp_id)
            metadata["edge_server"] = server
        elif method == "resource_tree":
            events, contexts = resource_tree_trace(
                graph, resources, links, model_bytes, seed, args.rounds, exp_id)
        else:
            events, contexts, metadata = drift_trace(
                graph, resources, links, model_bytes, seed, args.rounds, exp_id, experiment)
        node_rows = build_node_round_stats(events, contexts)
        simulated_rounds = []
        for context in contexts:
            round_id = int(context["round"])
            round_events = [event for event in events if int(event["round"]) == round_id]
            round_nodes = [row for row in node_rows if int(row["round"]) == round_id]
            simulated = simulate_round(round_events, round_nodes, partition["sample_counts"],
                                       int(experiment["local_epochs"]), assumptions, context)
            context["round_latency_sim_ms"] = simulated["round_latency_sim_ms"]
            simulated_rounds.append({"round": round_id, **simulated})
        summary = dump_system_metrics(str(run), events, contexts)
        node_rows = build_node_round_stats(events, contexts)
        summary.update(capacity_normalized_load_metrics(
            node_rows, assumptions["compute_capacity_by_node"]))
        summary.update({"trace_only": True, "training_performed": False,
                        "num_clients": len(graph), "rounds": args.rounds,
                        "model_bytes": model_bytes, "protocol_metadata": metadata,
                        "simulated_rounds": simulated_rounds,
                        "mean_round_latency_sim_ms": sum(
                            item["round_latency_sim_ms"] for item in simulated_rounds
                        ) / len(simulated_rounds),
                        "p95_round_latency_sim_ms": sorted(
                            item["round_latency_sim_ms"] for item in simulated_rounds
                        )[math.ceil(0.95 * len(simulated_rounds)) - 1]})
        metrics_path = run / "metrics" / "metrics_summary.json"
        metrics_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        config_dir = run / "config"; config_dir.mkdir(exist_ok=True)
        for name in ("experiment.json", "physical_graph.json", "node_resources.json",
                     "link_profiles.json", "data_partition.json", "manifest.json"):
            source = input_dir / name
            if source.exists(): shutil.copy2(source, config_dir / name)
        (config_dir / "completion_status.json").write_text(json.dumps({
            "status": "COMPLETE", "requested_rounds": args.rounds,
            "completed_rounds": args.rounds, "trace_only": True,
        }, indent=2), encoding="utf-8")
        summaries[method] = summary
        print(f"{method}: {summary['total_messages']} hops, "
              f"peak={summary['max_peak_node_traffic_bytes']} bytes, "
              f"latency={summary['per_round'][0]['round_latency_sim_ms']:.2f} ms")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"seed_{seed}_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
