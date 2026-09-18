"""Verify that FedAvg and DRIFT preserve the same first ten learning rounds."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_prefix(path: str, rounds: int = 10) -> list[dict[str, float]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < rounds:
        raise ValueError(f"{path} contains {len(rows)} rounds; {rounds} required")
    return [{key: float(row[key]) for key in ("avg_train_loss", "val_loss", "val_acc")}
            for row in rows[:rounds]]


def compare_prefixes(fedavg_path: str, drift_path: str, rounds: int = 10,
                     accuracy_tolerance: float = 1e-12,
                     loss_tolerance: float = 2e-5) -> dict:
    fedavg, drift = read_prefix(fedavg_path, rounds), read_prefix(drift_path, rounds)
    differences = []
    for round_id, (left, right) in enumerate(zip(fedavg, drift)):
        differences.append({
            "round": round_id,
            "accuracy_difference": abs(left["val_acc"] - right["val_acc"]),
            "validation_loss_difference": abs(left["val_loss"] - right["val_loss"]),
            "training_loss_difference": abs(left["avg_train_loss"] - right["avg_train_loss"]),
        })
    max_accuracy = max(item["accuracy_difference"] for item in differences)
    max_loss = max(item["validation_loss_difference"] for item in differences)
    passed = max_accuracy <= accuracy_tolerance and max_loss <= loss_tolerance
    return {"passed": passed, "rounds": rounds,
            "max_accuracy_difference": max_accuracy,
            "max_validation_loss_difference": max_loss,
            "differences": differences}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fedavg", required=True)
    parser.add_argument("--drift", required=True)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--accuracy-tolerance", type=float, default=1e-12)
    parser.add_argument("--loss-tolerance", type=float, default=2e-5)
    args = parser.parse_args()
    result = compare_prefixes(
        args.fedavg, args.drift, args.rounds,
        args.accuracy_tolerance, args.loss_tolerance,
    )
    print(f"rounds={result['rounds']}")
    print(f"max_accuracy_difference={result['max_accuracy_difference']:.12g}")
    print(f"max_validation_loss_difference={result['max_validation_loss_difference']:.12g}")
    print("prefix_alignment=" + ("PASSED" if result["passed"] else "FAILED"))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
