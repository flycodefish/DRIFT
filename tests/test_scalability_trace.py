from contextlib import redirect_stdout
from io import StringIO

from fl_core.scalability_trace import drift_trace, fedavg_edge_trace, resource_tree_trace


def _inputs(n=5):
    graph = {i: [((i - 1) % n), ((i + 1) % n)] for i in range(n)}
    resources = [{"node_id": i, "cpu_cores": i + 1, "memory_gb": 8,
                  "bandwidth_mbps": 100, "gpu_memory_gb": 0,
                  "compute_capacity": float(i + 1)} for i in range(n)]
    links = {"default_bandwidth_mbps": 100, "default_latency_ms": 5,
             "drop_rate": 0, "edge_overrides": {}}
    return graph, resources, links


def test_resource_tree_has_two_transfers_per_tree_edge():
    graph, resources, links = _inputs()
    events, contexts = resource_tree_trace(graph, resources, links, 100, 42, 2, "x")
    assert len(events) == 2 * 2 * (len(graph) - 1)
    assert len(contexts) == 2


def test_fedavg_edge_routes_each_client_both_directions():
    graph, resources, links = _inputs()
    events, contexts, server = fedavg_edge_trace(graph, resources, links, 100, 42, 1, "x")
    assert server == 4
    assert len(events) >= 2 * (len(graph) - 1)
    assert contexts[0]["aggregation_inputs_by_phase"]["MODEL_UPLOAD"][str(server)] == len(graph)


def test_drift_trace_scales_without_actors_and_preserves_physical_edges():
    graph, resources, links = _inputs(50)
    config = {"cluster_max_child": 3, "cluster_max_retry": 3,
              "cluster_request_timeout": 2.0, "cluster_offset": 30.0,
              "election_base_trigger_time": 10.0, "election_trigger_factor": 0.1,
              "distribution_similarity_threshold": 0.0, "resource_awareness": True}
    with redirect_stdout(StringIO()):
        events, contexts, metadata = drift_trace(
            graph, resources, links, 100, 42, 3, "x", config)
    edges = {tuple(sorted((src, dst))) for src, values in graph.items() for dst in values}
    assert events and len(contexts) == 3 and metadata["num_clusters"] > 0
    assert all(tuple(sorted((event["src"], event["dst"]))) in edges for event in events)
