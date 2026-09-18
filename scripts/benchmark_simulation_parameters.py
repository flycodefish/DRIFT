"""Measure reference training and aggregation costs for simulation calibration.

This is an isolated microbenchmark; it does not launch Ray or federated training.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn

# Make ``python scripts/benchmark_simulation_parameters.py`` resolve this
# checkout's packages, independently of any editable installation elsewhere.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deasfl.aggregation import sample_weighted_average
from fl_core.actors import ResNet18
from fl_core.data_loader import CNNModel, HARMLP


def timed(fn, repeats: int, device: torch.device) -> list[float]:
    values = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        values.append((time.perf_counter() - start) * 1000.0)
    return values


def stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "median_ms": statistics.median(ordered),
        "p25_ms": ordered[max(0, int(len(ordered) * 0.25) - 1)],
        "p75_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.75))],
        "min_ms": min(ordered), "max_ms": max(ordered),
        "repeats": len(ordered),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("cnn", "resnet18", "harmlp"), default="resnet18")
    parser.add_argument("--input-dim", type=int, default=561,
                        help="Feature dimension for vector models such as HARMLP")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--steps-per-measurement", type=int, default=1,
                        help="Run this many consecutive training steps per timing sample")
    parser.add_argument("--aggregation-repeats", type=int, default=20)
    parser.add_argument("--aggregation-steps-per-measurement", type=int, default=1,
                        help="Run this many aggregation operations per timing sample")
    parser.add_argument("--aggregation-warmup", type=int, default=5)
    parser.add_argument("--aggregation-inputs", default="2,4,8,16,30")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    if args.steps_per_measurement <= 0 or args.aggregation_steps_per_measurement <= 0:
        raise ValueError("measurement step counts must be positive")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    torch.set_num_threads(max(1, args.cpu_threads))
    torch.manual_seed(1234)

    if args.model == "resnet18":
        model = ResNet18(args.num_classes)
    elif args.model == "harmlp":
        model = HARMLP(input_dim=args.input_dim, num_classes=args.num_classes)
    else:
        model = CNNModel(num_classes=args.num_classes)
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    if args.model == "harmlp":
        images = torch.randn(args.batch_size, args.input_dim, device=device)
    else:
        images = torch.randn(args.batch_size, 3, 32, 32, device=device)
    labels = torch.randint(0, args.num_classes, (args.batch_size,), device=device)

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

    def train_measurement():
        for _ in range(args.steps_per_measurement):
            train_step()

    for _ in range(args.warmup):
        train_measurement()
    raw_train_times = timed(train_measurement, args.repeats, device)
    train_times = [value / args.steps_per_measurement for value in raw_train_times]
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model_bytes = sum(value.numel() * value.element_size() for value in state.values())

    aggregation = {}
    cpu = torch.device("cpu")
    for input_count in [int(item) for item in args.aggregation_inputs.split(",") if item.strip()]:
        states = [state for _ in range(input_count)]
        counts = [1] * input_count
        for _ in range(args.aggregation_warmup):
            for _ in range(args.aggregation_steps_per_measurement):
                sample_weighted_average(states, counts)
        def aggregation_measurement():
            for _ in range(args.aggregation_steps_per_measurement):
                sample_weighted_average(states, counts)
        raw_values = timed(aggregation_measurement, args.aggregation_repeats, cpu)
        values = [value / args.aggregation_steps_per_measurement for value in raw_values]
        aggregation[str(input_count)] = stats(values)

    output_dir = Path(args.output_dir or
                      f"outputs/calibration/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1, "status": "candidate",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "input_dim": args.input_dim,
        "num_classes": args.num_classes,
        "batch_size": args.batch_size, "device": str(device),
        "cpu_threads": torch.get_num_threads(),
        "model_bytes": model_bytes,
        "training_step": stats(train_times),
        "training_measurement_steps": args.steps_per_measurement,
        "aggregation_measurement_steps": args.aggregation_steps_per_measurement,
        "training_ms_per_sample_epoch": statistics.median(train_times) / args.batch_size,
        "aggregation_ms": aggregation,
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "notes": "Candidate microbenchmark; freeze only after review and repeated runs.",
    }
    path = output_dir / "benchmark.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Output: {path}")
    print(json.dumps({"device": str(device), "model_bytes": model_bytes,
                      "training_step": payload["training_step"],
                      "training_ms_per_sample_epoch": payload["training_ms_per_sample_epoch"]},
                     indent=2))


if __name__ == "__main__":
    main()
