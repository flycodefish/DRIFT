#!/usr/bin/env python3
"""Offline diagnosis for the fixed-size JSD clustering experiment.

This script does not modify or invoke the training protocol.  The optional
gradient statistic is a deterministic classifier-gradient proxy evaluated at
the shared frozen initial model; historical per-round client updates were not
stored by the training runs and therefore cannot be reconstructed exactly.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fl_core.actors import ResNet18
from fl_core.data_loader import load_cifar10_data


METHODS = tuple(f"drift_jsd_q{value:02d}" for value in (0, 25, 50, 75, 100))


def _latest(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No {name} under {root}")
    return matches[-1]


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    total = float(values.sum())
    if total <= 0:
        raise ValueError("Distribution has no mass")
    return values / total


def _jsd(left: np.ndarray, right: np.ndarray) -> float:
    epsilon = 1e-12
    p = _normalize(left) + epsilon
    q = _normalize(right) + epsilon
    p /= p.sum()
    q /= q.sum()
    midpoint = (p + q) / 2.0
    return float(
        (np.sum(p * np.log(p / midpoint)) + np.sum(q * np.log(q / midpoint))) / 2.0
    )


def _entropy_stats(probabilities: np.ndarray) -> tuple[float, float, int]:
    probabilities = _normalize(probabilities)
    positive = probabilities[probabilities > 0]
    entropy = float(-np.sum(positive * np.log(positive)))
    normalized = entropy / math.log(len(probabilities))
    return normalized, float(math.exp(entropy)), int(np.count_nonzero(probabilities))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0 else float("nan")


def _classifier_gradient_proxies(
    train_data: list[dict], initial_model_path: Path, device: torch.device,
    batch_size: int,
) -> list[np.ndarray]:
    model = ResNet18(10).to(device)
    model.load_state_dict(torch.load(initial_model_path, map_location="cpu", weights_only=True))
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("model.fc."))
    model.eval()
    proxies = []
    for client in train_data:
        model.zero_grad(set_to_none=True)
        images = client["images"]
        labels = client["labels"]
        for start in range(0, len(labels), batch_size):
            batch_images = images[start:start + batch_size].to(device)
            batch_labels = labels[start:start + batch_size].to(device)
            loss = F.cross_entropy(model(batch_images), batch_labels, reduction="sum")
            loss.backward()
        gradient = torch.cat([
            model.model.fc.weight.grad.detach().flatten(),
            model.model.fc.bias.grad.detach().flatten(),
        ]) / max(1, len(labels))
        proxies.append(gradient.cpu().numpy().astype(np.float64))
    return proxies


def _correlations(frame: pd.DataFrame, metrics: list[str], target: str) -> list[dict]:
    rows = []
    for metric in metrics:
        pair = frame[[metric, target]].dropna()
        rows.append({
            "scope": "tier_mean",
            "predictor": metric,
            "target": target,
            "n": len(pair),
            "pearson": float(pair[metric].corr(pair[target], method="pearson")),
            "spearman": float(pair[metric].corr(pair[target], method="spearman")),
        })
    return rows


def _markdown_table(frame: pd.DataFrame) -> str:
    def render(value) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6f}"
        return str(value)

    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(render(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _residual(values: np.ndarray, controls: pd.DataFrame) -> np.ndarray:
    matrix = np.column_stack([np.ones(len(controls)), controls.to_numpy(dtype=float)])
    return values - matrix @ np.linalg.lstsq(matrix, values, rcond=None)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--gradient-device", default="cuda")
    parser.add_argument("--gradient-batch-size", type=int, default=256)
    parser.add_argument("--skip-gradient-proxy", action="store_true")
    args = parser.parse_args()

    root = args.experiment_root.resolve()
    seed_dir = root / "seed_42"
    frozen = seed_dir / "frozen_inputs" / "config"
    partition = json.loads((frozen / "data_partition.json").read_text(encoding="utf-8"))
    indices = partition["client_indices"]
    train_data, val_data, _, global_test, client_distributions = load_cifar10_data(
        data_dir=args.data_dir,
        num_clients=len(indices),
        partition_mode="dirichlet",
        alpha=0.5,
        seed=42,
        client_indices=indices,
    )
    client_counts = np.asarray([len(item["labels"]) for item in train_data], dtype=np.int64)
    client_label_counts = np.stack([
        np.bincount(item["labels"].numpy(), minlength=10).astype(np.float64)
        for item in train_data
    ])
    global_train = client_label_counts.sum(axis=0)
    global_test_counts = np.bincount(global_test["labels"].numpy(), minlength=10)

    gradient_proxies = None
    gradient_definition = "not computed"
    if not args.skip_gradient_proxy:
        device = torch.device(
            args.gradient_device if args.gradient_device != "cuda" or torch.cuda.is_available()
            else "cpu"
        )
        gradient_proxies = _classifier_gradient_proxies(
            train_data, frozen / "initial_model.pt", device, args.gradient_batch_size
        )
        gradient_definition = (
            "mean pairwise cosine of per-client final-classifier gradients on each complete "
            "local training shard, at the shared frozen initial model, eval mode, no augmentation"
        )

    cluster_rows = []
    weight_audit = []
    for method in METHODS:
        diagnostic_path = _latest(seed_dir / "runs" / method, "cluster_round_metrics.csv")
        diagnostics = pd.read_csv(diagnostic_path)
        final_round = diagnostics[diagnostics["round_id"] == diagnostics["round_id"].max()]
        for _, row in final_round.iterrows():
            members = [int(value) for value in str(row["member_ids"]).split(",")]
            pooled_counts = client_label_counts[members].sum(axis=0)
            p_cluster = _normalize(pooled_counts)
            entropy, effective_classes, nonzero_classes = _entropy_stats(p_cluster)
            pair_cosines = []
            if gradient_proxies is not None:
                for index, left in enumerate(members):
                    for right in members[index + 1:]:
                        pair_cosines.append(_cosine(gradient_proxies[left], gradient_proxies[right]))
            expected_samples = int(client_counts[members].sum())
            recorded_samples = int(row["train_samples"])
            weight_audit.append({
                "method": method, "cluster_head": int(row["cluster_head"]),
                "expected_train_samples": expected_samples,
                "recorded_train_samples": recorded_samples,
                "exact_match": expected_samples == recorded_samples,
            })
            result = {
                "method": method,
                "target_quantile": int(method.removeprefix("drift_jsd_q")) / 100.0,
                "cluster_head": int(row["cluster_head"]),
                "members": ",".join(map(str, members)),
                "num_members": len(members),
                "train_samples": expected_samples,
                "mean_pairwise_client_label_jsd": float(row["mean_pairwise_label_jsd"]),
                "cluster_label_entropy": entropy,
                "effective_class_count": effective_classes,
                "nonzero_class_coverage": nonzero_classes,
                "jsd_cluster_to_global_train": _jsd(p_cluster, global_train),
                "jsd_cluster_to_global_test": _jsd(p_cluster, global_test_counts),
                "largest_class_share": float(p_cluster.max()),
                "mean_client_gradient_cosine_proxy": (
                    float(np.nanmean(pair_cosines)) if pair_cosines else None
                ),
                "client_gradient_dispersion_proxy": (
                    float(1.0 - np.nanmean(pair_cosines)) if pair_cosines else None
                ),
                "cluster_val_accuracy": float(row["cluster_val_accuracy"]),
                "cluster_global_accuracy": float(row["cluster_global_accuracy"]),
                "cluster_validation_samples": int(row["validation_samples"]),
            }
            result.update({f"p_class_{index}": float(value) for index, value in enumerate(p_cluster)})
            cluster_rows.append(result)

    clusters = pd.DataFrame(cluster_rows)
    audit = pd.DataFrame(weight_audit)
    tier_rows = []
    for method, frame in clusters.groupby("method", sort=False):
        sample_weights = frame["train_samples"].astype(float)
        pair_weights = frame["num_members"] * (frame["num_members"] - 1) / 2.0
        pair_total = max(1.0, float(pair_weights.sum()))
        total = max(1.0, float(sample_weights.sum()))
        tier_rows.append({
            "method": method,
            "target_quantile": float(frame["target_quantile"].iloc[0]),
            "pair_weighted_client_label_jsd": float(
                (frame["mean_pairwise_client_label_jsd"] * pair_weights).sum() / pair_total
            ),
            "sample_weighted_cluster_label_entropy": float(
                (frame["cluster_label_entropy"] * sample_weights).sum() / total
            ),
            "sample_weighted_effective_class_count": float(
                (frame["effective_class_count"] * sample_weights).sum() / total
            ),
            "sample_weighted_nonzero_class_coverage": float(
                (frame["nonzero_class_coverage"] * sample_weights).sum() / total
            ),
            "sample_weighted_jsd_to_global_train": float(
                (frame["jsd_cluster_to_global_train"] * sample_weights).sum() / total
            ),
            "sample_weighted_jsd_to_global_test": float(
                (frame["jsd_cluster_to_global_test"] * sample_weights).sum() / total
            ),
            "pair_weighted_gradient_cosine_proxy": (
                float((frame["mean_client_gradient_cosine_proxy"] * pair_weights).sum() / pair_total)
                if frame["mean_client_gradient_cosine_proxy"].notna().any() else None
            ),
            "pair_weighted_gradient_dispersion_proxy": (
                float((frame["client_gradient_dispersion_proxy"] * pair_weights).sum() / pair_total)
                if frame["client_gradient_dispersion_proxy"].notna().any() else None
            ),
            "sample_weighted_cluster_val_accuracy": float(
                (frame["cluster_val_accuracy"] * frame["cluster_validation_samples"]).sum()
                / max(1.0, float(frame["cluster_validation_samples"].sum()))
            ),
            "sample_weighted_cluster_global_accuracy": float(
                (frame["cluster_global_accuracy"] * sample_weights).sum() / total
            ),
        })
    tiers = pd.DataFrame(tier_rows).sort_values("target_quantile")

    predictors = [
        "pair_weighted_client_label_jsd",
        "sample_weighted_cluster_label_entropy",
        "sample_weighted_effective_class_count",
        "sample_weighted_jsd_to_global_train",
        "sample_weighted_jsd_to_global_test",
        "pair_weighted_gradient_cosine_proxy",
        "pair_weighted_gradient_dispersion_proxy",
    ]
    correlations = pd.DataFrame(
        _correlations(tiers, predictors, "sample_weighted_cluster_val_accuracy")
        + _correlations(tiers, predictors, "sample_weighted_cluster_global_accuracy")
    )
    cluster_correlations = []
    cluster_predictors = [
        "mean_pairwise_client_label_jsd", "cluster_label_entropy",
        "effective_class_count", "nonzero_class_coverage",
        "jsd_cluster_to_global_train", "jsd_cluster_to_global_test",
        "mean_client_gradient_cosine_proxy", "client_gradient_dispersion_proxy",
        "train_samples", "num_members",
    ]
    for target in ("cluster_val_accuracy", "cluster_global_accuracy"):
        for metric in cluster_predictors:
            pair = clusters[[metric, target]].dropna()
            controls = clusters.loc[pair.index, ["train_samples", "num_members"]]
            predictor_residual = _residual(pair[metric].to_numpy(dtype=float), controls)
            target_residual = _residual(pair[target].to_numpy(dtype=float), controls)
            cluster_correlations.append({
                "scope": "all_clusters", "predictor": metric, "target": target,
                "n": len(pair),
                "pearson": float(pair[metric].corr(pair[target], method="pearson")),
                "spearman": float(pair[metric].corr(pair[target], method="spearman")),
                "partial_pearson_controlling_samples_members": float(
                    np.corrcoef(predictor_residual, target_residual)[0, 1]
                ),
            })
    cluster_correlations = pd.DataFrame(cluster_correlations)

    output = (args.output_dir or root / "jsd_diagnostic_report").resolve()
    output.mkdir(parents=True, exist_ok=True)
    clusters.to_csv(output / "cluster_statistics.csv", index=False)
    tiers.to_csv(output / "tier_statistics.csv", index=False)
    correlations.to_csv(output / "tier_correlations.csv", index=False)
    cluster_correlations.to_csv(output / "cluster_correlations.csv", index=False)
    audit.to_csv(output / "sample_weight_audit.csv", index=False)

    strongest = cluster_correlations[
        (cluster_correlations["target"] == "cluster_global_accuracy")
        & (~cluster_correlations["predictor"].isin(["train_samples", "num_members"]))
    ].dropna(subset=["partial_pearson_controlling_samples_members"]).assign(
        absolute=lambda x: x["partial_pearson_controlling_samples_members"].abs()
    ).sort_values(
        "absolute", ascending=False
    ).iloc[0]
    report = [
        "# JSD cluster-effect diagnostic report",
        "",
        "## Scope and provenance",
        "",
        "- Offline analysis only; the training and aggregation mechanisms were not modified.",
        f"- Experiment root: `{root}`",
        "- Label statistics use the exact frozen CIFAR-10 client indices and official labels.",
        f"- Gradient statistic: {gradient_definition}.",
        "- Historical per-round client updates were not archived, so the gradient metric is a proxy, not the actual training update cosine.",
        "",
        "## Tier summary (round 30)",
        "",
        _markdown_table(tiers),
        "",
        "## Correlations across five tier means",
        "",
        _markdown_table(correlations),
        "",
        "## Correlations across all 45 clusters",
        "",
        _markdown_table(cluster_correlations),
        "",
        "## Integrity audit",
        "",
        f"- Sample-weight matches: `{int(audit['exact_match'].sum())}/{len(audit)}` clusters.",
        f"- Total frozen training samples: `{int(client_counts.sum())}`.",
        f"- Official global test samples: `{len(global_test['labels'])}`.",
        "- Cluster-local validation is the union of fixed client-specific slices of the official test set; membership changes this subset.",
        "- `cluster_global_accuracy` evaluates every cluster model on the same complete official 10,000-image test set and is therefore the safer cross-tier check.",
        "",
        "## Automated diagnostic",
        "",
        f"- Strongest all-cluster association with cluster-global accuracy after controlling "
        f"for training samples and member count: `{strongest['predictor']}` "
        f"(partial Pearson `{strongest['partial_pearson_controlling_samples_members']:.6f}`, "
        f"n={int(strongest['n'])}).",
        "- Interpret n=5 correlations descriptively; they are not inferential evidence.",
        "- See `cluster_statistics.csv` for every P_C vector and per-cluster result.",
    ]
    (output / "diagnostic_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Cluster statistics: {output / 'cluster_statistics.csv'}")
    print(f"Tier statistics: {output / 'tier_statistics.csv'}")
    print(f"Correlations: {output / 'tier_correlations.csv'}")
    print(f"Cluster correlations: {output / 'cluster_correlations.csv'}")
    print(f"Weight audit: {output / 'sample_weight_audit.csv'}")
    print(f"Report: {output / 'diagnostic_report.md'}")


if __name__ == "__main__":
    main()
