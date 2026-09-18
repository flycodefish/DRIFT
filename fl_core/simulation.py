"""Explicit, replayable V1 simulated-time model.

The model is intentionally opt-in: no paper latency is emitted without a
frozen assumptions JSON file.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any
from copy import copy


def transfer_latency_ms(event: dict[str, Any], assumptions: dict[str, Any]) -> float:
    bandwidth = event.get("bandwidth_mbps") or assumptions["default_bandwidth_mbps"]
    propagation = event.get("latency_ms")
    if propagation is None:
        propagation = assumptions["default_link_latency_ms"]
    one_attempt = (float(propagation)
                   + int(event.get("payload_bytes", 0)) * 8
                   / max(float(bandwidth), 1e-9) / 1000)
    return (one_attempt * int(event.get("attempt_count", 1) or 1)
            + float(event.get("backoff_ms", 0) or 0))


def _dag_critical_path(events: list[dict[str, Any]], assumptions: dict[str, Any]) -> float:
    if not events:
        return 0.0
    edges = {}
    nodes = set()
    for event in events:
        src, dst = str(event["src"]), str(event["dst"])
        if src == dst:
            continue
        nodes.update((src, dst))
        edges[(src, dst)] = max(edges.get((src, dst), 0.0),
                                transfer_latency_ms(event, assumptions))
    outgoing = defaultdict(list)
    indegree = {node: 0 for node in nodes}
    for (src, dst), latency in edges.items():
        outgoing[src].append((dst, latency))
        indegree[dst] += 1
    queue = deque(node for node, degree in indegree.items() if degree == 0)
    distance = {node: 0.0 for node in nodes}
    visited = 0
    while queue:
        src = queue.popleft()
        visited += 1
        for dst, latency in outgoing[src]:
            distance[dst] = max(distance[dst], distance[src] + latency)
            indegree[dst] -= 1
            if indegree[dst] == 0:
                queue.append(dst)
    if visited != len(nodes):
        # Synchronous peer exchange can contain cycles; under the frozen
        # unlimited-independent-link assumption it completes at the slowest edge.
        return max(edges.values(), default=0.0)
    return max(distance.values(), default=0.0)


def _aggregation_duration_ms(inputs: int, node_id: str, model_mb: float,
                             assumptions: dict[str, Any],
                             capacities: dict[str, float]) -> float:
    """Duration of one actual weighted aggregation operation."""
    if inputs <= 0:
        return 0.0
    fixed = float(assumptions.get("aggregation_fixed_ms", 0.0))
    rate = float(assumptions.get("aggregation_ms_per_model_mb", 0.0))
    default = float(assumptions.get(
        "default_aggregation_capacity",
        assumptions.get("default_compute_capacity", 1.0),
    ))
    aggregation_capacities = {
        str(key): float(value) for key, value in
        assumptions.get("aggregation_capacity_by_node", {}).items()
    }
    key = str(node_id)
    plain_key = key.removeprefix("client_")
    capacity = max(aggregation_capacities.get(
        key, aggregation_capacities.get(
            plain_key, capacities.get(key, capacities.get(plain_key, default))
        )
    ), 1e-9)
    return (fixed + inputs * model_mb * rate) / capacity


def _dag_critical_path_with_aggregations(
    events: list[dict[str, Any]],
    aggregation_inputs: dict[str, int],
    model_mb: float,
    assumptions: dict[str, Any],
    capacities: dict[str, float],
) -> float:
    """Critical path with an aggregation operation between receives/sends.

    Each node is split into ``in`` and ``out`` vertices.  Incoming transfers
    end at ``in``; one aggregation edge then leads to ``out``.  This makes a
    parent wait for all child messages before sending its aggregate upward,
    while retaining parallel transfers on independent links.
    """
    if not events:
        return 0.0
    edges: dict[tuple[str, str], float] = {}
    nodes: set[str] = set()

    def in_node(node: str) -> str:
        return f"{node}:in"

    def out_node(node: str) -> str:
        return f"{node}:out"

    endpoints = {str(value) for event in events for value in
                 (event["src"], event["dst"])}
    endpoints.update(str(value) for value in aggregation_inputs)
    for node in endpoints:
        left, right = in_node(node), out_node(node)
        duration = _aggregation_duration_ms(
            int(aggregation_inputs.get(node, 0)), node, model_mb,
            assumptions, capacities,
        )
        edges[(left, right)] = duration
        nodes.update((left, right))

    for event in events:
        src, dst = str(event["src"]), str(event["dst"])
        if src == dst:
            continue
        key = (out_node(src), in_node(dst))
        edges[key] = max(edges.get(key, 0.0),
                         transfer_latency_ms(event, assumptions))
        nodes.update(key)

    outgoing = defaultdict(list)
    indegree = {node: 0 for node in nodes}
    for (src, dst), latency in edges.items():
        outgoing[src].append((dst, latency))
        indegree[dst] += 1
    queue = deque(node for node, degree in indegree.items() if degree == 0)
    distance = {node: 0.0 for node in nodes}
    visited = 0
    while queue:
        src = queue.popleft()
        visited += 1
        for dst, latency in outgoing[src]:
            distance[dst] = max(distance[dst], distance[src] + latency)
            indegree[dst] -= 1
            if indegree[dst] == 0:
                queue.append(dst)
    if visited != len(nodes):
        # Preserve the old safe behavior for cyclic peer-like logs.
        return max(edges.values(), default=0.0)
    return max(distance.values(), default=0.0)


def _event_phase(event: dict[str, Any]) -> str:
    """Return an explicit phase, with compatibility inference for old logs."""
    if event.get("phase"):
        return str(event["phase"])
    message_type = str(event.get("message_type", "")).upper()
    hop_count = int(event.get("physical_hop_count", 1) or 1)
    if hop_count > 1 and message_type == "AGGREGATED_MODEL":
        return "INTER_CLUSTER_UPLOAD"
    if hop_count > 1 and message_type == "GLOBAL_MODEL":
        return "INTER_CLUSTER_DOWNLOAD"
    return {
        "AGGREGATED_MODEL": "INTRA_CLUSTER_UPLOAD",
        "ROUTED_MODEL": "INTER_CLUSTER_UPLOAD",
        "GLOBAL_MODEL": "MODEL_DISTRIBUTION",
        "MODEL_UPLOAD": "MODEL_UPLOAD",
        "UPLOAD": "MODEL_UPLOAD",
        "BROADCAST": "MODEL_DISTRIBUTION",
        "MODEL_DOWNLOAD": "MODEL_DISTRIBUTION",
        "GOSSIP": "PEER_EXCHANGE",
        "GOSSIP_PUSH": "PEER_EXCHANGE",
        "PEER_MODEL": "PEER_EXCHANGE",
    }.get(str(event.get("message_type", "")).upper(), "CONTROL")


def _resource_scheduled_time(events: list[dict[str, Any]],
                             assumptions: dict[str, Any]) -> float:
    """FIFO schedule for shared node interfaces and directed links."""
    if not events:
        return 0.0
    node_capacity = {
        str(k): float(v) for k, v in
        assumptions.get("network_capacity_by_node", {}).items()
    }
    tx_ready: dict[str, float] = defaultdict(float)
    rx_ready: dict[str, float] = defaultdict(float)
    link_ready: dict[tuple[str, str], float] = defaultdict(float)
    finish = 0.0
    for event in events:  # input order is the stable message-id order
        scheduled = copy(event)
        src, dst = str(event["src"]), str(event["dst"])
        link_bandwidth = float(event.get("bandwidth_mbps") or
                               assumptions["default_bandwidth_mbps"])
        link_bandwidth = min(link_bandwidth,
                             node_capacity.get(src, link_bandwidth),
                             node_capacity.get(dst, link_bandwidth))
        scheduled["bandwidth_mbps"] = max(link_bandwidth, 1e-9)
        duration = transfer_latency_ms(scheduled, assumptions)
        key = (src, dst)
        start = max(tx_ready[src], rx_ready[dst], link_ready[key])
        end = start + duration
        tx_ready[src], rx_ready[dst], link_ready[key] = end, end, end
        finish = max(finish, end)
    return finish


def _joint_scheduled_time(events: list[dict[str, Any]],
                          aggregation_inputs: dict[str, int],
                          model_mb: float,
                          assumptions: dict[str, Any],
                          capacities: dict[str, float]) -> float:
    """List-schedule communication and aggregation with shared resources."""
    if not events:
        return 0.0
    tasks: dict[str, dict[str, Any]] = {}
    incoming: dict[str, list[str]] = defaultdict(list)
    outgoing: dict[str, list[str]] = defaultdict(list)
    for index, event in enumerate(events):
        task_id = f"event:{index}"
        src, dst = str(event["src"]), str(event["dst"])
        bandwidth = float(event.get("bandwidth_mbps") or
                          assumptions["default_bandwidth_mbps"])
        node_capacity = assumptions.get("network_capacity_by_node", {})
        bandwidth = min(
            bandwidth,
            float(node_capacity.get(src, bandwidth)),
            float(node_capacity.get(dst, bandwidth)),
        )
        transfer = dict(event)
        transfer["bandwidth_mbps"] = max(bandwidth, 1e-9)
        tasks[task_id] = {
            "kind": "transfer", "event": transfer, "duration": transfer_latency_ms(transfer, assumptions),
            "resources": (f"tx:{src}", f"rx:{dst}", f"link:{src}->{dst}"),
            "order": index,
        }

    # One operation per configured destination.  All inbound model transfers
    # in this phase must complete before it becomes ready.
    operation_ids = {}
    for offset, (node, count) in enumerate(sorted(aggregation_inputs.items())):
        task_id = f"aggregate:{node}"
        operation_ids[node] = task_id
        tasks[task_id] = {
            "kind": "aggregation",
            "duration": _aggregation_duration_ms(
                int(count), node, model_mb, assumptions, capacities
            ),
            "resources": (f"compute:{node}",),
            "order": len(events) + offset,
        }

    for index, event in enumerate(events):
        task_id = f"event:{index}"
        src, dst = str(event["src"]), str(event["dst"])
        if dst in operation_ids:
            incoming[operation_ids[dst]].append(task_id)
            outgoing[task_id].append(operation_ids[dst])
        if src in operation_ids:
            incoming[task_id].append(operation_ids[src])
            outgoing[operation_ids[src]].append(task_id)

    # Multi-hop route messages must preserve hop order.  Intra-cluster flow
    # IDs deliberately do not serialize sibling tree edges.
    by_flow: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for index, event in enumerate(events):
        flow = event.get("flow_id")
        if flow and str(event.get("phase", "")).startswith("INTER_CLUSTER"):
            by_flow[str(flow)].append((int(event.get("physical_hop_index", 1) or 1), f"event:{index}"))
    for route in by_flow.values():
        route.sort()
        for (_, previous), (_, current) in zip(route, route[1:]):
            incoming[current].append(previous)
            outgoing[previous].append(current)

    indegree = {task_id: len(incoming[task_id]) for task_id in tasks}
    ready = [task_id for task_id, degree in indegree.items() if degree == 0]
    ready.sort(key=lambda task_id: tasks[task_id]["order"])
    resource_ready: dict[str, float] = defaultdict(float)
    finish: dict[str, float] = {}
    while ready:
        task_id = ready.pop(0)
        task = tasks[task_id]
        start = max((finish[parent] for parent in incoming[task_id]), default=0.0)
        start = max(start, *(resource_ready[item] for item in task["resources"]))
        end = start + float(task["duration"])
        finish[task_id] = end
        for resource in task["resources"]:
            resource_ready[resource] = end
        for child in outgoing[task_id]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
        ready.sort(key=lambda item: tasks[item]["order"])
    if len(finish) != len(tasks):
        return max((float(task["duration"]) for task in tasks.values()), default=0.0)
    return max(finish.values(), default=0.0)


def _peer_exchange_scheduled_time(
    events: list[dict[str, Any]],
    aggregation_inputs: dict[str, int],
    model_mb: float,
    assumptions: dict[str, Any],
    capacities: dict[str, float],
) -> tuple[float, float]:
    """Schedule frozen-model peer sends followed by receiver-side averaging.

    Unlike tree aggregation, a node's outgoing model is frozen before peer
    exchange and therefore does not wait for its same-round averaging.  Only
    hops within one routed overlay flow are serialized; receiver averaging
    waits for the final hops of all successfully received logical messages.
    """
    if not events:
        return 0.0, 0.0
    tasks: dict[str, dict[str, Any]] = {}
    incoming: dict[str, list[str]] = defaultdict(list)
    outgoing: dict[str, list[str]] = defaultdict(list)
    for index, event in enumerate(events):
        task_id = f"event:{index}"
        src, dst = str(event["src"]), str(event["dst"])
        bandwidth = float(
            event.get("bandwidth_mbps") or assumptions["default_bandwidth_mbps"]
        )
        node_capacity = assumptions.get("network_capacity_by_node", {})
        bandwidth = min(
            bandwidth,
            float(node_capacity.get(src, bandwidth)),
            float(node_capacity.get(dst, bandwidth)),
        )
        transfer = dict(event)
        transfer["bandwidth_mbps"] = max(bandwidth, 1e-9)
        tasks[task_id] = {
            "kind": "transfer",
            "duration": transfer_latency_ms(transfer, assumptions),
            "resources": (f"tx:{src}", f"rx:{dst}", f"link:{src}->{dst}"),
            "order": index,
        }

    by_flow: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for index, event in enumerate(events):
        flow = event.get("flow_id") or f"event:{index}"
        by_flow[str(flow)].append(
            (
                int(event.get("physical_hop_index", 1) or 1),
                int(event.get("physical_hop_count", 1) or 1),
                f"event:{index}",
            )
        )
    final_transfer_ids: list[str] = []
    for route in by_flow.values():
        route.sort()
        last_hop_index, expected_hops, last_task_id = route[-1]
        if last_hop_index == expected_hops:
            final_transfer_ids.append(last_task_id)
        for (_, _, previous), (_, _, current) in zip(route, route[1:]):
            incoming[current].append(previous)
            outgoing[previous].append(current)

    operation_ids: dict[str, str] = {}
    for offset, (node, count) in enumerate(sorted(aggregation_inputs.items())):
        if int(count) <= 0:
            continue
        task_id = f"aggregate:{node}"
        operation_ids[str(node)] = task_id
        tasks[task_id] = {
            "kind": "aggregation",
            "duration": _aggregation_duration_ms(
                int(count), str(node), model_mb, assumptions, capacities
            ),
            "resources": (f"compute:{node}",),
            "order": len(events) + offset,
        }

    for transfer_id in final_transfer_ids:
        index = int(transfer_id.split(":", 1)[1])
        destination = str(events[index]["dst"])
        operation_id = operation_ids.get(destination)
        if operation_id:
            incoming[operation_id].append(transfer_id)
            outgoing[transfer_id].append(operation_id)

    indegree = {task_id: len(incoming[task_id]) for task_id in tasks}
    ready = sorted(
        (task_id for task_id, degree in indegree.items() if degree == 0),
        key=lambda task_id: tasks[task_id]["order"],
    )
    resource_ready: dict[str, float] = defaultdict(float)
    finish: dict[str, float] = {}
    while ready:
        task_id = ready.pop(0)
        task = tasks[task_id]
        start = max((finish[parent] for parent in incoming[task_id]), default=0.0)
        start = max(start, *(resource_ready[item] for item in task["resources"]))
        end = start + float(task["duration"])
        finish[task_id] = end
        for resource in task["resources"]:
            resource_ready[resource] = end
        for child in outgoing[task_id]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
        ready.sort(key=lambda item: tasks[item]["order"])
    if len(finish) != len(tasks):
        raise ValueError("peer-exchange dependency graph contains a cycle")
    communication = max(
        (finish[task_id] for task_id, task in tasks.items()
         if task["kind"] == "transfer"),
        default=0.0,
    )
    return communication, max(finish.values(), default=0.0)


def simulate_round(events: list[dict[str, Any]], node_rows: list[dict[str, Any]],
                   sample_counts: list[int], local_epochs: int,
                   assumptions: dict[str, Any],
                   round_context: dict[str, Any] | None = None) -> dict[str, float]:
    upload_types = {"UPLOAD", "MODEL_UPLOAD", "AGGREGATED_MODEL", "ROUTED_MODEL"}
    download_types = {"BROADCAST", "MODEL_DOWNLOAD", "GLOBAL_MODEL"}
    peer_types = {"GOSSIP", "GOSSIP_PUSH", "PEER_MODEL"}
    successful = [event for event in events if event.get("success", False)]
    phases = defaultdict(list)
    for event in successful:
        phases[_event_phase(event)].append(event)
    capacities = {str(key): float(value)
                  for key, value in assumptions.get("compute_capacity_by_node", {}).items()}
    model_mb = max((int(event.get("payload_bytes", 0)) for event in successful), default=0) / 1_000_000

    def operation_inputs(name: str) -> dict[str, int]:
        configured = (round_context or {}).get("aggregation_inputs_by_phase", {})
        if name in configured:
            endpoints = {
                str(value) for event in phases[name]
                for value in (event["src"], event["dst"])
            }

            def endpoint_key(node: Any) -> str:
                value = str(node)
                candidate = f"client_{value}"
                return candidate if value not in endpoints and candidate in endpoints else value

            return {
                endpoint_key(node): int(count)
                for node, count in configured[name].items()
                if int(count) > 0
            }
        inputs: dict[str, int] = defaultdict(int)
        for event in phases[name]:
            message_type = str(event.get("message_type", "")).upper()
            is_aggregate = (
                name == "INTRA_CLUSTER_UPLOAD" and message_type == "AGGREGATED_MODEL"
            ) or (
                name == "INTER_CLUSTER_UPLOAD" and
                message_type == "AGGREGATED_MODEL" and
                int(event.get("physical_hop_index", 1) or 1) ==
                int(event.get("physical_hop_count", 1) or 1)
            ) or (
                name == "MODEL_UPLOAD" and
                message_type in {"UPLOAD", "MODEL_UPLOAD"}
            )
            if is_aggregate:
                inputs[str(event["dst"])] += 1
        return dict(inputs)

    operation_durations: dict[str, float] = {}

    def phase_time(name: str) -> tuple[float, float]:
        phase_events = phases[name]
        inputs = operation_inputs(name)
        for node, count in inputs.items():
            operation_durations[f"{name}:{node}"] = _aggregation_duration_ms(
                count, node, model_mb, assumptions, capacities
            )
        communication = max(_dag_critical_path(phase_events, assumptions),
                            _resource_scheduled_time(phase_events, assumptions))
        dag = (_dag_critical_path_with_aggregations(
            phase_events, inputs, model_mb, assumptions, capacities
        ) if inputs else _dag_critical_path(phase_events, assumptions))
        effective = (_joint_scheduled_time(
            phase_events, inputs, model_mb, assumptions, capacities
        ) if inputs else communication)
        return communication, max(dag, effective)

    intra_upload, intra_upload_effective = phase_time("INTRA_CLUSTER_UPLOAD")
    inter_upload, inter_upload_effective = phase_time("INTER_CLUSTER_UPLOAD")
    model_upload, model_upload_effective = phase_time("MODEL_UPLOAD")
    intra_download, intra_download_effective = phase_time("INTRA_CLUSTER_DOWNLOAD")
    inter_download, inter_download_effective = phase_time("INTER_CLUSTER_DOWNLOAD")
    model_download, model_download_effective = phase_time("MODEL_DISTRIBUTION")
    peer_inputs = operation_inputs("PEER_EXCHANGE")
    for node, count in peer_inputs.items():
        operation_durations[f"PEER_EXCHANGE:{node}"] = _aggregation_duration_ms(
            count, node, model_mb, assumptions, capacities
        )
    peer, peer_effective = _peer_exchange_scheduled_time(
        phases["PEER_EXCHANGE"], peer_inputs, model_mb, assumptions, capacities
    )
    upload = intra_upload + inter_upload + model_upload
    download = intra_download + inter_download + model_download
    upload_effective = (intra_upload_effective + inter_upload_effective
                        + model_upload_effective)
    download_effective = (intra_download_effective + inter_download_effective
                          + model_download_effective)
    default_capacity = float(assumptions.get("default_compute_capacity", 1.0))
    reference = float(assumptions["reference_train_ms_per_sample_epoch"])
    train_times = [
        count * local_epochs * reference / max(capacities.get(str(node), default_capacity), 1e-9)
        for node, count in enumerate(sample_counts)
    ]
    aggregation = max(
        upload_effective - upload + download_effective - download
        + peer_effective - peer,
        0.0,
    )
    return {
        "train_critical_path_ms": max(train_times, default=0.0),
        "upload_critical_path_ms": upload,
        "download_critical_path_ms": download,
        "peer_exchange_critical_path_ms": peer,
        "aggregation_critical_path_ms": aggregation,
        "intra_cluster_upload_ms": intra_upload,
        "inter_cluster_upload_ms": inter_upload,
        "inter_cluster_download_ms": inter_download,
        "intra_cluster_download_ms": intra_download,
        "round_latency_sim_ms": (max(train_times, default=0.0)
                                  + upload_effective + download_effective
                                  + peer_effective),
    }
