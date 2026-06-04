from __future__ import annotations

import csv
import json
from pathlib import Path


class Monitor:
    """Collect round metrics and communication stats and dump json/csv."""

    def __init__(self) -> None:
        self.round_metrics: list[dict] = []
        self.comm_events: list[dict] = []

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
            writer = csv.DictWriter(
                fp,
                fieldnames=[
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
                ],
            )
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
                ],
            )
            writer.writeheader()
            writer.writerows(self.comm_events)
