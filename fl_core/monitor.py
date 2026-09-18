from __future__ import annotations

import csv
import json
from pathlib import Path

from .system_metrics import dump_system_metrics


class Monitor:
    """Collect round metrics and communication stats and dump json/csv."""

    def __init__(self) -> None:
        self.round_metrics: list[dict] = []
        self.comm_events: list[dict] = []
        self.round_contexts: list[dict] = []
        self.topology_events: list[dict] = []
        self.experiment_id = ""
        self.method = ""
        self.seed = 0

    def configure(self, experiment_id: str, method: str, seed: int) -> None:
        self.experiment_id = experiment_id
        self.method = method
        self.seed = seed

    def record_round_context(self, **context) -> None:
        self.round_contexts.append(context)

    def record_topology_event(self, **event) -> None:
        self.topology_events.append(event)

    def record_decentralized_metrics(self, round_id: int, **metrics) -> None:
        for row in reversed(self.round_metrics):
            if row["round_id"] == round_id:
                row.update(metrics)
                return
        raise ValueError(f"Round {round_id} must be recorded before diagnostics")

    def record_round(
        self,
        round_id: int,
        avg_train_loss: float,
        val_loss: float,
        val_acc: float,
        selected_count: int,
        dropped_count: int,
        comm_success_count: int,
        comm_failure_count: int,
        comm_timeout_count: int,
        broadcast_success_count: int,
        broadcast_failure_count: int,
        upload_success_count: int,
        upload_failure_count: int,
        upload_storage_blocked_count: int,
        gossip_success_count: int,
        gossip_failure_count: int,
        round_wall_clock_seconds: float | None = None,
    ) -> None:
        self.round_metrics.append(
            {
                "round_id": round_id,
                "avg_train_loss": avg_train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "selected_count": selected_count,
                "dropped_count": dropped_count,
                "comm_success_count": comm_success_count,
                "comm_failure_count": comm_failure_count,
                "comm_timeout_count": comm_timeout_count,
                "broadcast_success_count": broadcast_success_count,
                "broadcast_failure_count": broadcast_failure_count,
                "upload_success_count": upload_success_count,
                "upload_failure_count": upload_failure_count,
                "upload_storage_blocked_count": upload_storage_blocked_count,
                "gossip_success_count": gossip_success_count,
                "gossip_failure_count": gossip_failure_count,
                "round_wall_clock_seconds": round_wall_clock_seconds,
            }
        )

    def record_comm(
        self,
        round_id: int,
        src: str,
        dst: str,
        payload_bytes: int,
        kind: str,
        simulated_latency_ms: float,
        bandwidth_mbps: float,
        drop_rate: float,
        delivered: bool,
        retries_used: int,
        timeout_occurred: bool,
        total_backoff_ms: float,
        link_latency_ms: float | None = None,
        message_type: str | None = None,
        phase: str | None = None,
        flow_id: str | None = None,
        logical_parent_message_id: str | None = None,
        link_type: str | None = None,
        physical_hop_index: int = 1,
        physical_hop_count: int = 1,
    ) -> None:
        self.comm_events.append(
            {
                "round_id": round_id,
                "src": src,
                "dst": dst,
                "payload_bytes": payload_bytes,
                "kind": kind,
                "simulated_latency_ms": simulated_latency_ms,
                "bandwidth_mbps": bandwidth_mbps,
                "drop_rate": drop_rate,
                "delivered": delivered,
                "retries_used": retries_used,
                "timeout_occurred": timeout_occurred,
                "total_backoff_ms": total_backoff_ms,
                "link_latency_ms": link_latency_ms,
                "message_type": message_type,
                "phase": phase,
                "flow_id": flow_id,
                "logical_parent_message_id": logical_parent_message_id,
                "link_type": link_type,
                "physical_hop_index": physical_hop_index,
                "physical_hop_count": physical_hop_count,
            }
        )

    def dump(self, output_dir: str, run_name: str) -> None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / f"{run_name}_metrics.json"
        csv_round_path = out_dir / f"{run_name}_round_metrics.csv"
        csv_comm_path = out_dir / f"{run_name}_comm_events.csv"

        payload = {
            "round_metrics": self.round_metrics,
            "comm_events": self.comm_events,
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        with csv_round_path.open("w", newline="", encoding="utf-8") as fp:
            base_fields = [
                    "round_id",
                    "avg_train_loss",
                    "val_loss",
                    "val_acc",
                    "selected_count",
                    "dropped_count",
                    "comm_success_count",
                    "comm_failure_count",
                    "comm_timeout_count",
                    "broadcast_success_count",
                    "broadcast_failure_count",
                    "upload_success_count",
                    "upload_failure_count",
                    "upload_storage_blocked_count",
                    "gossip_success_count",
                    "gossip_failure_count",
                    "round_wall_clock_seconds",
                ]
            extra_fields = sorted({key for row in self.round_metrics for key in row} - set(base_fields))
            writer = csv.DictWriter(fp, fieldnames=base_fields + extra_fields)
            writer.writeheader()
            writer.writerows(self.round_metrics)

        with csv_comm_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=[
                    "round_id",
                    "src",
                    "dst",
                    "payload_bytes",
                    "kind",
                    "simulated_latency_ms",
                    "bandwidth_mbps",
                    "drop_rate",
                    "delivered",
                    "retries_used",
                    "timeout_occurred",
                    "total_backoff_ms",
                    "link_latency_ms",
                ],
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(self.comm_events)

        normalized_events = []
        for index, event in enumerate(self.comm_events):
            normalized_events.append({
                "experiment_id": self.experiment_id or run_name,
                "method": self.method,
                "seed": self.seed,
                "round": event["round_id"],
                "timestamp_start": None,
                "timestamp_end": None,
                "src": event["src"],
                "dst": event["dst"],
                "message_type": event.get("message_type") or str(event["kind"]).upper(),
                "message_id": f"r{event['round_id']}-m{index}",
                "phase": event.get("phase") or {
                    "MODEL_UPLOAD": "MODEL_UPLOAD",
                    "UPLOAD": "MODEL_UPLOAD",
                    "GLOBAL_MODEL": "MODEL_DISTRIBUTION",
                    "BROADCAST": "MODEL_DISTRIBUTION",
                }.get(str(event["kind"]).upper(), "UNCLASSIFIED"),
                "flow_id": event.get("flow_id") or f"round:{event['round_id']}:{str(event['kind']).lower()}",
                "logical_parent_message_id": event.get("logical_parent_message_id"),
                "payload_bytes": event["payload_bytes"],
                "bandwidth_mbps": event["bandwidth_mbps"],
                "latency_ms": event.get("link_latency_ms"),
                "drop_rate": event["drop_rate"],
                "link_type": event.get("link_type") or "DIRECT",
                "success": event["delivered"],
                "physical_hop_index": int(event.get("physical_hop_index", 1) or 1),
                "physical_hop_count": int(event.get("physical_hop_count", 1) or 1),
                "attempt_count": int(event.get("retries_used", 0)) + 1,
                "backoff_ms": event.get("total_backoff_ms", 0.0),
            })
        if not self.round_contexts:
            nodes = sorted({
                str(value).removeprefix("client_")
                for event in self.comm_events
                for value in (event["src"], event["dst"])
            })
            for metric in self.round_metrics:
                self.round_contexts.append({
                    "experiment_id": self.experiment_id or run_name,
                    "method": self.method,
                    "seed": self.seed,
                    "round": metric["round_id"],
                    "node_ids": nodes,
                    "roles": {node: ("ROOT" if node == "server" else "CLIENT") for node in nodes},
                    "active_nodes": nodes,
                    "failed_nodes": [],
                    "eligible_node_ratio": 1.0,
                    "reachable_node_ratio": 1.0,
                    "eligible_sample_ratio": 1.0,
                    "aggregated_sample_ratio": (
                        metric["selected_count"] - metric["dropped_count"]
                    ) / max(1, metric["selected_count"]),
                    "round_completion_ratio": 1.0,
                    "round_wall_clock_seconds": metric.get("round_wall_clock_seconds"),
                    "round_latency_sim_ms": None,
                })
        dump_system_metrics(
            output_dir,
            normalized_events,
            self.round_contexts,
            self.topology_events,
        )
