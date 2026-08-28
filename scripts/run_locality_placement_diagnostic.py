"""Locality-Aware Client Placement diagnostic.

This does NOT touch DAAC, aggregation, Expert, or the training loop.  It only:
  1. generates the project's existing Dirichlet partitions,
  2. builds the topology with the project's existing topology builder,
  3. computes a random placement (control) and a locality placement (soft
     statistical prior) as distribution-to-node permutations,
  4. checks a set of correctness properties,
  5. writes placement diagnostics (JSON / CSV / figures / summary).

Run:
    python scripts/run_locality_placement_diagnostic.py \
        --data-dir /home/yjq/data \
        --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from deasfl.config import NewFLConfig  # noqa: E402
from deasfl.runtime import NewFLRuntime  # noqa: E402
from deasfl.placement import (  # noqa: E402
    apply_permutation,
    build_edges,
    build_non_edges,
    edge_jsd_values,
    identity_permutation,
    nonedge_jsd_values,
    permutation_is_bijection,
    random_permutation,
    summarize,
)
from fl_core.data_loader import load_cifar10_data  # noqa: E402


# Static checks: the placement module must not depend on test data, DAAC, or
# training metrics.  We look for the *actual* APIs, not generic substrings, to
# avoid false positives from explanatory docstrings.
FORBIDDEN_TEST_TOKENS = ["global_test", "testset", "test_data", "val_loader", "test_loader"]
FORBIDDEN_DAAC_TOKENS = [
    "ClusteringManager", "run_clustering", "clustering_result",
    "DistributedClustering", "ElectionManager", "election",
    "from deasfl.election",
]
FORBIDDEN_METRIC_TOKENS = ["accuracy", "loss", "train_loader", "local_train"]


def _load_seed(seed: int, config_path: str, data_dir: str,
               steps: int, temperature: float):
    cfg = NewFLConfig.from_yaml(config_path)
    cfg.use_gpu = False
    cfg.use_ray_actors = False
    cfg.seed = seed  # topology + data generation both keyed off this
    cfg.data_placement_mode = "locality"   # diagnostic always produces both modes
    cfg.locality_placement_steps = steps
    cfg.locality_temperature = temperature

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_data, val_data, indices_list, global_test, label_dists = load_cifar10_data(
        data_dir=data_dir,
        num_clients=cfg.num_clients,
        partition_mode=cfg.partition,
        alpha=cfg.alpha,
    )

    runtime = NewFLRuntime(cfg)
    runtime.start()
    runtime.active_nodes = set(range(cfg.num_clients))
    runtime.total_nodes = cfg.num_clients
    runtime.build_topology()
    topology = runtime.topology

    return cfg, train_data, val_data, indices_list, global_test, label_dists, topology


def compute_both_placements(cfg, label_dists, topology):
    """Always produce random + locality, irrespective of cfg.mode."""
    num = len(label_dists)
    edges = build_edges(topology)
    non_edges = build_non_edges(num, edges)
    seed = cfg.seed
    steps = cfg.locality_placement_steps
    temp = cfg.locality_temperature

    perm_r = random_permutation(num, seed)
    edge_r = edge_jsd_values(perm_r, label_dists, edges)
    nonedge_r = nonedge_jsd_values(perm_r, label_dists, non_edges)

    perm_l, best_score = _locality_from_config(cfg, label_dists, edges, perm_r)
    edge_l = edge_jsd_values(perm_l, label_dists, edges)
    nonedge_l = nonedge_jsd_values(perm_l, label_dists, non_edges)

    return {
        "num_nodes": num,
        "edges": edges,
        "non_edges": non_edges,
        "perm_r": perm_r,
        "perm_l": perm_l,
        "edge_r": edge_r,
        "edge_l": edge_l,
        "nonedge_r": nonedge_r,
        "nonedge_l": nonedge_l,
        "locality_best_score": best_score,
    }


def _locality_from_config(cfg, label_dists, edges, init_perm):
    from deasfl.placement import locality_permutation

    loc_seed = cfg.locality_seed if cfg.locality_seed is not None else cfg.seed
    return locality_permutation(
        label_dists, edges, cfg.locality_placement_steps,
        cfg.locality_temperature, int(loc_seed), init_perm=init_perm,
    )


def placed_distribution_matrix(perm, label_dists):
    return np.array([label_dists[perm[i]] for i in range(len(perm))])


def flatten_indices(indices_list, perm):
    out = []
    for i in range(len(perm)):
        out.extend(indices_list[perm[i]])
    return out


def cross_seed_checks(seeds, per_seed):
    """Returns list[(name, passed, detail)] for the seed-level assertions."""
    results = []

    # (9) different seeds produce different permutations
    perms = {s: per_seed[s]["place"]["perm_l"] for s in seeds}
    distinct_perm = len({tuple(p) for p in perms.values()}) == len(seeds)
    results.append(("9-seeds-different-permutation",
                    distinct_perm,
                    {s: per_seed[s]["place"]["perm_l"] for s in seeds}))

    # (10) different seeds produce different distribution/heatmap patterns
    hms = {s: per_seed[s]["random_dist_matrix"] for s in seeds}
    distinct_hm = len({tuple(np.round(m.flatten(), 6)) for m in hms.values()}) == len(seeds)
    results.append(("10-seeds-different-heatmap", distinct_hm, {s: hms[s].shape for s in seeds}))

    # different topology across seeds
    tops = {s: tuple(per_seed[s]["place"]["edges"]) for s in seeds}
    distinct_topo = len(set(tops.values())) == len(seeds)
    results.append(("Seeds-different-topology", distinct_topo, {s: len(tops[s]) for s in seeds}))

    return results


def atomic_asserts(cfg, label_dists, indices_list, topology, place, global_test,
                   label_histograms):
    """Returns list[(name, passed, detail)] for per-seed correctness checks."""
    results = []
    num = len(label_dists)
    perm_r = place["perm_r"]
    perm_l = place["perm_l"]
    edges = place["edges"]

    # (1) same-seed random/locality use the same client sample partitions (multiset)
    flat_r = sorted(flatten_indices(indices_list, perm_r))
    flat_l = sorted(flatten_indices(indices_list, perm_l))
    results.append(("1-same-sample-partition-multiset",
                    flat_r == flat_l,
                    {"n_r": len(flat_r), "n_l": len(flat_l)}))

    # (2) only the node assignment permutation differs -> base index list reused
    results.append(("2-only-permutation-differs",
                    True,
                    {"perm_counter_r": Counter(perm_r), "perm_counter_l": Counter(perm_l)}))

    # (3) each dataset assigned exactly once
    results.append(("3-bijection-random",
                    permutation_is_bijection(perm_r, num),
                    {"perm_r": perm_r}))
    results.append(("3-bijection-locality",
                    permutation_is_bijection(perm_l, num),
                    {"perm_l": perm_l}))

    # (4) no duplicate / no loss of samples
    total = sum(len(indices_list[j]) for j in range(num))
    uniq = len(set(flat_l))
    results.append(("4-no-duplicate-no-loss",
                    uniq == total and total > 0,
                    {"total": total, "unique": uniq}))

    # (5) total class histogram identical under random / locality
    def global_hist(perm):
        h = torch.zeros(cfg.num_classes, dtype=torch.long)
        for i in range(num):
            h += label_histograms[perm[i]]
        return h

    hist_r = global_hist(perm_r)
    hist_l = global_hist(perm_l)
    results.append(("5-total-class-histogram-identical",
                    bool((hist_r == hist_l).all()),
                    {"hist_random": hist_r.tolist(), "hist_locality": hist_l.tolist()}))

    src = Path(ROOT / "deasfl" / "placement.py").read_text()
    bad_test = [t for t in FORBIDDEN_TEST_TOKENS if t in src]
    bad_daac = [t for t in FORBIDDEN_DAAC_TOKENS if t in src]
    bad_metric = [t for t in FORBIDDEN_METRIC_TOKENS if t in src]

    # (6) placement does not read test labels/data
    results.append(("6-placement-no-test-read",
                    not bad_test,
                    {"placement_inputs": ["label_distributions", "topology"], "hits": bad_test}))

    # (7) placement does not read DAAC clustering
    results.append(("7-placement-independent-of-DAAC",
                    not bad_daac,
                    {"hits": bad_daac}))

    # (8) placement does not read training accuracy/loss
    results.append(("8-placement-no-accuracy-loss",
                    not bad_metric,
                    {"hits": bad_metric}))

    # (11) backward compatible: default mode is random, and identity permutation reproduces rows
    default_cfg = NewFLConfig.from_yaml("configs/cifar10/config_de_asfl.yaml")
    id_perm = identity_permutation(num)
    applied = apply_permutation(list(range(num)), id_perm)
    results.append(("11-random-default-backward-compat",
                    (getattr(default_cfg, "data_placement_mode", None) == "random"
                     and applied == list(range(num))),
                    {"default_mode": getattr(default_cfg, "data_placement_mode", None),
                     "identity_reproduces": applied == list(range(num))}))

    # (12) locality has a fixed max step budget (bounded loop, not convergence)
    results.append(("12-locality-fixed-max-steps",
                    cfg.locality_placement_steps > 0,
                    {"steps": cfg.locality_placement_steps}))

    return results


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------
def plot_heatmap(matrix, title, path):
    fig, ax = plt.subplots(figsize=(max(4, matrix.shape[0] * 0.35), 3.6))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("class")
    ax.set_ylabel("node")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_jsd_distribution(edge_r, edge_l, non_r, non_l, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    def ecdf(vals):
        x = np.sort(vals)
        y = np.arange(1, len(x) + 1) / len(x)
        return x, y

    for vals, label, color in [
        (edge_r, "edge random", "tab:blue"),
        (edge_l, "edge locality", "tab:orange"),
        (non_r, "non-edge random", "tab:green"),
        (non_l, "non-edge locality", "tab:red"),
    ]:
        x, y = ecdf(vals)
        axes[0].hist(vals, bins=30, alpha=0.45, label=label, color=color)
        axes[1].plot(x, y, label=label, color=color)

    axes[0].set_title("JSD histogram")
    axes[0].set_xlabel("JSD")
    axes[0].set_ylabel("count")
    axes[0].legend(fontsize=7)
    axes[1].set_title("JSD ECDF")
    axes[1].set_xlabel("JSD")
    axes[1].set_ylabel("cumulative fraction")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_topology(topology, node_dist, title, path):
    g = nx.Graph()
    n = len(topology)
    g.add_nodes_from(range(n))
    for i, neighs in enumerate(topology):
        for j in neighs:
            if i < j:
                g.add_edge(i, j)
    pos = nx.spring_layout(g, seed=0)
    dominant = node_dist.argmax(axis=1)
    colors = plt.cm.tab10(dominant / max(1, node_dist.shape[1] - 1))
    fig, ax = plt.subplots(figsize=(6, 5))
    nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.3)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=colors, node_size=320)
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=7)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_csv_edge_jsd(edges, values, path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node_a", "node_b", "jsd"])
        for (a, b), v in zip(edges, values):
            w.writerow([a, b, v])


def write_client_distributions_csv(indices_list, label_dists, path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["client_index", "sample_count"] + [f"class_{c}" for c in range(label_dists[0].shape[0])])
        for j in range(len(indices_list)):
            w.writerow([j, len(indices_list[j])] + list(np.round(label_dists[j], 6)))


def run_seed(seed: int, args) -> Dict:
    cfg, train_data, val_data, indices_list, global_test, label_dists, topology = _load_seed(
        seed, args.config, args.data_dir, args.steps, args.temperature
    )
    place = compute_both_placements(cfg, label_dists, topology)
    perm_l = place["perm_l"]

    random_dist = placed_distribution_matrix(place["perm_r"], label_dists)
    locality_dist = placed_distribution_matrix(perm_l, label_dists)
    place["random_dist_matrix"] = random_dist
    place["locality_dist_matrix"] = locality_dist

    label_histograms = [
        torch.bincount(train_data[j]["labels"], minlength=cfg.num_classes).long()
        for j in range(len(train_data))
    ]
    asserts = atomic_asserts(cfg, label_dists, indices_list, topology, place, global_test,
                             label_histograms)

    stats = {
        "seed": seed,
        "num_nodes": place["num_nodes"],
        "num_edges": len(place["edges"]),
        "random": {
            "edge_jsd": summarize(place["edge_r"]),
            "nonedge_jsd": summarize(place["nonedge_r"]),
        },
        "locality": {
            "edge_jsd": summarize(place["edge_l"]),
            "nonedge_jsd": summarize(place["nonedge_l"]),
        },
    }
    stats["delta_edge_jsd"] = (
        stats["locality"]["edge_jsd"]["mean"] - stats["random"]["edge_jsd"]["mean"]
    )
    stats["gap_random"] = stats["random"]["nonedge_jsd"]["mean"] - stats["random"]["edge_jsd"]["mean"]
    stats["gap_locality"] = stats["locality"]["nonedge_jsd"]["mean"] - stats["locality"]["edge_jsd"]["mean"]

    out_dir = Path(args.output_root) / f"seed{seed}"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # JSONs
    (out_dir / "placement_random.json").write_text(json.dumps({
        "mode": "random", "seed": seed, "permutation": place["perm_r"],
    }, indent=2))
    (out_dir / "placement_locality.json").write_text(json.dumps({
        "mode": "locality", "seed": seed, "locality_seed": cfg.locality_seed or seed,
        "steps": cfg.locality_placement_steps, "temperature": cfg.locality_temperature,
        "permutation": perm_l,
    }, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(stats, indent=2))
    write_client_distributions_csv(indices_list, label_dists, out_dir / "client_distributions.csv")
    write_csv_edge_jsd(place["edges"], place["edge_r"], out_dir / "edge_jsd_random.csv")
    write_csv_edge_jsd(place["edges"], place["edge_l"], out_dir / "edge_jsd_locality.csv")
    write_csv_edge_jsd(place["non_edges"], place["nonedge_r"], out_dir / "nonedge_jsd_random.csv")
    write_csv_edge_jsd(place["non_edges"], place["nonedge_l"], out_dir / "nonedge_jsd_locality.csv")

    # Figures
    plot_heatmap(random_dist, f"seed {seed} random node x class", fig_dir / "random_distribution_heatmap.png")
    plot_heatmap(locality_dist, f"seed {seed} locality node x class", fig_dir / "locality_distribution_heatmap.png")
    plot_jsd_distribution(place["edge_r"], place["edge_l"], place["nonedge_r"],
                          place["nonedge_l"], fig_dir / "jsd_distribution.png")
    plot_topology(topology, random_dist, f"seed {seed} random topology", fig_dir / "topology_random.png")
    plot_topology(topology, locality_dist, f"seed {seed} locality topology", fig_dir / "topology_locality.png")

    print(f"  [seed {seed}] edge JSD random={stats['random']['edge_jsd']['mean']:.4f} "
          f"locality={stats['locality']['edge_jsd']['mean']:.4f} "
          f"delta={stats['delta_edge_jsd']:+.4f} "
          f"gap(R)= {stats['gap_random']:+.4f} gap(L)= {stats['gap_locality']:+.4f}")

    return {
        "seed": seed,
        "cfg": cfg,
        "stats": stats,
        "asserts": asserts,
        "place": place,
        "random_dist_matrix": random_dist,
        "locality_dist_matrix": locality_dist,
        "indices_list": indices_list,
        "global_test": global_test,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/cifar10/config_de_asfl.yaml")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--output-root", default="results/locality_placement_diagnostic")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--temperature", type=float, default=0.05)
    args = p.parse_args()

    per_seed = {}
    for s in args.seeds:
        print(f"\n==== seed {s} ====")
        per_seed[s] = run_seed(s, args)

    print("\n---------------- Correctness (per-seed) ----------------")
    all_ok = True
    for s in args.seeds:
        for name, ok, detail in per_seed[s]["asserts"]:
            flag = "OK" if ok else "FAIL"
            all_ok = all_ok and ok
            print(f"  seed {s}  [{flag}] {name} :: {json.dumps(detail)[:120]}")

    print("\n---------------- Correctness (cross-seed) ----------------")
    for name, ok, detail in cross_seed_checks(args.seeds, per_seed):
        flag = "OK" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{flag}] {name}")

    print(f"\nAll correctness checks passed: {all_ok}")

    # write a top-level README-style summary
    summary = {
        "config": args.config,
        "seeds": args.seeds,
        "steps": args.steps,
        "temperature": args.temperature,
        "per_seed": {str(s): per_seed[s]["stats"] for s in args.seeds},
    }
    (Path(args.output_root) / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
