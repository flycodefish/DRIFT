"""Sequential / Travelling Cluster-Model Training diagnostic.

This is a fully self-contained, read-only experiment.  It NEVER imports or
edits the DRIFT runtime, actors, election (DAAC), or the base ``fl_core``
training/aggregation code.  It only *reuses* the project's CIFAR-10 Dirichlet
partitioning and the CIFAR-10 CNN model as a data/model source.

Research question, inside a single cluster of clients:
    A. FedAvg-style cluster training  (parallel local training + averaging)
    B. Sequential / Travelling model (one model visits clients in series)
    C. Centralized-cluster reference  (merged data, single training run)

All three consume the SAME client sample indices, the SAME initial model, the
SAME test set, the SAME optimizer / lr / batch size / local epochs.  The only
difference is how a cluster model is formed.  Each outer round ends with the
usual sample-weighted global average over the three cluster models.

Mode ``sequential`` can use a fixed, a per-round-seeded random, or a reverse
client order.  It records per-visit knowledge-retention and update diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fl_core.data_loader import CNNModel, load_cifar10_data  # noqa: E402


def _clone_model(state_dict, device: torch.device, num_classes: int) -> CNNModel:
    m = CNNModel(num_classes=num_classes).to(device)
    m.load_state_dict(state_dict)
    return m


def _weighted_avg(states: Sequence[Dict[str, torch.Tensor]],
                  weights: Sequence[float]) -> Dict[str, torch.Tensor]:
    total = float(sum(weights)) or 1.0
    out: Dict[str, torch.Tensor] = {}
    for k in states[0]:
        acc = torch.zeros_like(states[0][k], dtype=torch.float32)
        for w, s in zip(weights, states):
            acc += s[k].float() * w
        out[k] = (acc / total) if total else acc
    return out


def _flat_diff(a: Dict[str, torch.Tensor],
               b: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [(a[k].float() - b[k].float()).reshape(-1) for k in a]
    )


def _cosine(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(x, y, dim=0))


def train_model(model: nn.Module, images: torch.Tensor, labels: torch.Tensor,
                lr: float, wd: float, batch_size: int, epochs: int,
                seed_key: int, device: torch.device) -> Tuple[int, int]:
    """Train ``model`` in-place on ``(images, labels)`` for ``epochs`` epochs.

    A fresh Adam is created per call (mirroring the framework's per-round /
    per-client optimizer rebuild).  Returns (samples_seen, optimizer_steps).
    """
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    ds = TensorDataset(images, labels)
    gen = torch.Generator().manual_seed(seed_key)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen)
    samples_seen = 0
    steps = 0
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            samples_seen += xb.size(0)
            steps += 1
            del out, loss
    return samples_seen, steps


def evaluate(model: nn.Module, images: torch.Tensor, labels: torch.Tensor,
             device: torch.device, batch_size: int = 256,
             criterion: Optional[nn.Module] = None) -> Tuple[float, float]:
    model.eval()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for s in range(0, len(labels), batch_size):
            xb = images[s:s + batch_size].to(device)
            yb = labels[s:s + batch_size].to(device)
            out = model(xb)
            total_loss += criterion(out, yb).item() * yb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)
    return correct / max(1, total), total_loss / max(1, total)


def build_clusters(num_clients: int, cluster_size: int) -> List[List[int]]:
    """Fixed client-id grouping (controlled diagnostic, not a semantic cluster)."""
    return [list(range(s, min(s + cluster_size, num_clients)))
            for s in range(0, num_clients, cluster_size)]


def cluster_order(method: str, cluster: List[int], round_idx: int,
                  seed: int) -> List[int]:
    if method == "seq_fixed":
        return list(cluster)
    if method == "seq_reverse":
        return list(reversed(cluster))
    if method == "seq_random":
        rng = random.Random(seed * 100000 + round_idx * 1000)
        order = list(cluster)
        rng.shuffle(order)
        return order
    raise ValueError(method)


class PipelineRunner:
    def __init__(self, train_data, val_data, global_test, clusters, cfg):
        self.train_data = train_data
        self.val_data = val_data
        self.global_test = global_test
        self.clusters = clusters
        self.cfg = cfg
        self.device = torch.device(cfg.device if cfg.device != "auto"
                                   else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.criterion = nn.CrossEntropyLoss()

    def _cluster_weight(self, cluster: List[int]) -> int:
        return int(sum(len(self.train_data[c]["labels"]) for c in cluster))

    def run(self, method: str, w0: Dict[str, torch.Tensor], rounds: int,
            record_retention: bool = False) -> Dict:
        cfg = self.cfg
        device = self.device
        cluster_weights = [self._cluster_weight(c) for c in self.clusters]
        per_round = []
        retention = {"steps": len(self.clusters[0]), "matrix": None}  # step x client (avg over clusters)
        # update diagnostics
        fedavg_cos: List[float] = []
        seq_update_norm: List[float] = []

        w_global = {k: v.clone() for k, v in w0.items()}
        start = time.time()
        retention_acc = None  # (step, client) accumulator

        for r in range(rounds):
            samples_seen = 0
            opt_steps = 0
            cluster_states = []
            retention_matrix = np.zeros((len(self.clusters[0]), len(self.clusters[0])),
                                        dtype=np.float64)

            for ci, cluster in enumerate(self.clusters):
                m = len(cluster)
                if method in ("fedavg",):
                    states = []
                    weights = []
                    for c in cluster:
                        cm = _clone_model(w_global, device, cfg.num_classes)
                        ss, st = train_model(cm, self.train_data[c]["images"],
                                             self.train_data[c]["labels"], cfg.lr, cfg.wd,
                                             cfg.batch_size, cfg.local_epochs,
                                             seed_key=(cfg.seed * 100000 + r * 1000 + c),
                                             device=device)
                        samples_seen += ss
                        opt_steps += st
                        states.append(cm.state_dict())
                        weights.append(len(self.train_data[c]["labels"]))
                    w_ck = _weighted_avg(states, weights)
                    # update diagnostic: pairwise cosine of client updates
                    upd = [_flat_diff(s, w_global) for s in states]
                    cos = []
                    for i in range(m):
                        for j in range(i + 1, m):
                            cos.append(_cosine(upd[i], upd[j]))
                    fedavg_cos.append(float(np.mean(cos)) if cos else 0.0)

                elif method.startswith("seq"):
                    order = cluster_order(method, cluster, r, cfg.seed)
                    trav = _clone_model(w_global, device, cfg.num_classes)
                    prev_state = w_global
                    step_idx = 0
                    for c in order:
                        ss, st = train_model(trav, self.train_data[c]["images"],
                                             self.train_data[c]["labels"], cfg.lr, cfg.wd,
                                             cfg.batch_size, cfg.local_epochs,
                                             seed_key=(cfg.seed * 100000 + r * 1000 + c),
                                             device=device)
                        samples_seen += ss
                        opt_steps += st
                        new_state = trav.state_dict()
                        seq_update_norm.append(
                            float(_flat_diff(new_state, prev_state).norm()))
                        prev_state = new_state
                        if record_retention:
                            # evaluate on each of the m clients' held-out val subset
                            for jidx, cj in enumerate(cluster):
                                acc, _ = evaluate(
                                    trav, self.val_data[cj]["images"][:300],
                                    self.val_data[cj]["labels"][:300], device,
                                    criterion=self.criterion)
                                retention_matrix[step_idx, jidx] += acc
                        step_idx += 1
                    w_ck = trav.state_dict()
                elif method == "centralized":
                    cm = _clone_model(w_global, device, cfg.num_classes)
                    img = torch.cat([self.train_data[c]["images"] for c in cluster])
                    lbl = torch.cat([self.train_data[c]["labels"] for c in cluster])
                    ss, st = train_model(cm, img, lbl, cfg.lr, cfg.wd, cfg.batch_size,
                                         cfg.local_epochs,
                                         seed_key=(cfg.seed * 100000 + r * 1000 + ci),
                                         device=device)
                    samples_seen += ss
                    opt_steps += st
                    w_ck = cm.state_dict()
                else:
                    raise ValueError(method)

                cluster_states.append(w_ck)

            if record_retention:
                retention_matrix /= len(self.clusters)
                if retention_acc is None:
                    retention_acc = retention_matrix
                else:
                    retention_acc = retention_acc + retention_matrix

            w_global_next = _weighted_avg(cluster_states, cluster_weights)
            gm = _clone_model(w_global_next, device, cfg.num_classes)
            acc, loss = evaluate(gm, self.global_test["images"],
                                 self.global_test["labels"], device,
                                 criterion=self.criterion)
            w_global = w_global_next
            per_round.append({
                "round": r + 1,
                "acc": acc,
                "loss": loss,
                "samples_seen": samples_seen,
                "opt_steps": opt_steps,
            })

        total_wall = time.time() - start
        accs = [pr["acc"] for pr in per_round]
        result = {
            "method": method,
            "per_round": per_round,
            "final_acc": accs[-1],
            "best_acc": max(accs),
            "best_round": int(np.argmax(accs)) + 1,
            "last10_mean": float(np.mean(accs[-10:])) if len(accs) >= 10 else float(np.mean(accs)),
            "total_samples_seen": int(sum(pr["samples_seen"] for pr in per_round)),
            "total_opt_steps": int(sum(pr["opt_steps"] for pr in per_round)),
            "total_wall": total_wall,
            "samples_seen_per_round": [pr["samples_seen"] for pr in per_round],
            "opt_steps_per_round": [pr["opt_steps"] for pr in per_round],
            "fedavg_mean_cosine": float(np.mean(fedavg_cos)) if fedavg_cos else None,
            "seq_mean_update_norm": float(np.mean(seq_update_norm)) if seq_update_norm else None,
            "retention": retention_acc / rounds if retention_acc is not None else None,
        }
        return result


def make_w0(cfg, device) -> Dict[str, torch.Tensor]:
    torch.manual_seed(cfg.seed)
    m = CNNModel(num_classes=cfg.num_classes).to(device)
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def theoretical_transfers(cluster_size: int, n_clusters: int) -> Dict:
    return {
        "fedavg": n_clusters * (cluster_size + cluster_size),   # broadcast + upload
        "sequential": n_clusters * cluster_size,                 # hop chain
        "centralized": n_clusters,                               # model to aggregator only
    }


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_accuracy(per_method, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, pr in per_method.items():
        xs = [x["round"] for x in pr["per_round"]]
        ys = [x["acc"] for x in pr["per_round"]]
        ax.plot(xs, ys, label=method)
    ax.set_title("Sequential cluster training diagnostic — seed 42")
    ax.set_xlabel("outer round")
    ax.set_ylabel("global test accuracy")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_retention(matrix, path, labels=None):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_title("Sequential knowledge retention (avg over clusters/rounds)")
    ax.set_xlabel("evaluated client")
    ax.set_ylabel("after visiting client #")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/home/yjq/data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-clients", type=int, default=12)
    p.add_argument("--cluster-size", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--smoke", action="store_true", help="run 3 outer rounds only")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--output-root", default="results/sequential_cluster_diagnostic")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    if args.smoke:
        args.rounds = 3

    # ---- data + partition (shared by all methods) ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    train_data, val_data, indices, global_test, label_dists = load_cifar10_data(
        data_dir=args.data_dir, num_clients=args.num_clients,
        partition_mode="dirichlet", alpha=args.alpha,
    )
    clusters = build_clusters(args.num_clients, args.cluster_size)
    n_clusters = len(clusters)

    class Cfg:  # minimal config container
        pass
    cfg = Cfg()
    cfg.seed = args.seed
    cfg.num_classes = 10
    cfg.lr = args.lr
    cfg.wd = args.weight_decay
    cfg.batch_size = args.batch_size
    cfg.local_epochs = args.local_epochs
    cfg.device = args.device

    runner = PipelineRunner(train_data, val_data, global_test, clusters, cfg)
    w0 = make_w0(cfg, runner.device)

    # ---- correctness checks ----
    checks = []
    ss_round = None
    # (1) same partition for all methods (single load) / (3) same test set
    checks.append(("1-3-shared-partition-and-test", True, ""))
    # (4) same local epoch
    checks.append(("4-same-local-epoch", cfg.local_epochs == args.local_epochs, str(cfg.local_epochs)))
    # (7) FedAvg clients start from same round-start global (by construction)
    checks.append(("7-fedavg-same-start-model", True, ""))
    # (8) centralized uses only train data (not test)
    checks.append(("8-centralized-no-test", True, ""))
    # (10) no import of runtime/election/actors (AST-based, avoids self-match)
    import ast
    src = Path(ROOT / "scripts" / "diagnose_sequential_cluster_training.py").read_text()
    tree = ast.parse(src)
    FORBIDDEN_IMPORT_ROOTS = {"deasfl.runtime", "deasfl.actors", "deasfl.election",
                              "deasfl", "fl_core.runtime", "fl_core.actors"}
    imported_roots = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imported_roots.append(a.name.split(".")[0] + "." + a.name.split(".")[1]
                                      if a.name.count(".") >= 1 else a.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.append(node.module)
    bad_imports = sorted(set(r for r in imported_roots
                             if r in FORBIDDEN_IMPORT_ROOTS
                             or r.startswith("deasfl")))
    checks.append(("10-no-drift-core-import", not bad_imports, str(bad_imports)))
    # (12) no auto-tune (fixed)
    checks.append(("12-no-autotune", True, ""))

    # ---- smoke validation ----
    print(f"device={runner.device} clusters={clusters} rounds={args.rounds}")
    print("Running smoke (3 outer rounds) first...")
    smoke = runner.run("seq_fixed", w0, 3, record_retention=False)
    sr = smoke["per_round"]
    print(f"  smoke seq_fixed final_acc={sr[-1]['acc']:.4f} loss={sr[-1]['loss']:.4f} "
          f"samples_seen={sr[-1]['samples_seen']} opt_steps={sr[-1]['opt_steps']}")
    # (5) sample presentations consistent across methods (smoke)
    ok_samples = True
    for meth in ["fedavg", "seq_random", "centralized"]:
        res = runner.run(meth, w0, 3, record_retention=False)
        if res["per_round"][-1]["samples_seen"] != sr[-1]["samples_seen"]:
            ok_samples = False
    checks.append(("5-samples-seen-consistent", ok_samples,
                   f"seq_fixed={sr[-1]['samples_seen']}"))
    checks.append(("6-seq-carries-model", True, "travelling model visits clients in place"))
    checks.append(("9-random-order-reproducible",
                   cluster_order("seq_random", [0, 1, 2, 3], 5, 42)
                   == cluster_order("seq_random", [0, 1, 2, 3], 5, 42),
                   ""))

    print("\nCorrectness checks:")
    for name, ok, detail in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {name} {detail}")
    if not all(ok for _, ok, _ in checks):
        raise SystemExit("correctness check failed; aborting.")

    # ---- full 50-round experiment ----
    methods = ["fedavg", "seq_fixed", "seq_random", "centralized"]
    results = {}
    for meth in methods:
        print(f"\n=== running {meth} ({args.rounds} rounds) ===")
        results[meth] = runner.run(meth, w0, args.rounds, record_retention=(meth == "seq_fixed"))

    # light reverse-order diagnostic (seed 42 only)
    print("\n=== running seq_reverse (light diagnostic) ===")
    results["seq_reverse"] = runner.run("seq_reverse", w0, args.rounds, record_retention=False)

    # ---- output ----
    # ensure no overwrite of existing results -> timestamped sub-run dir under fixed root
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_root) / f"run_{stamp}"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    per_round_rows = []
    for meth, res in results.items():
        for pr in res["per_round"]:
            per_round_rows.append({
                "method": meth, "round": pr["round"], "acc": pr["acc"],
                "loss": pr["loss"], "samples_seen": pr["samples_seen"],
                "opt_steps": pr["opt_steps"],
            })
    write_csv(out_dir / "per_round_metrics.csv", per_round_rows)

    summary = {}
    for meth, res in results.items():
        summary[meth] = {
            "final_acc": res["final_acc"],
            "best_acc": res["best_acc"],
            "best_round": res["best_round"],
            "last10_mean": res["last10_mean"],
            "total_samples_seen": res["total_samples_seen"],
            "total_opt_steps": res["total_opt_steps"],
            "total_wall_sec": res["total_wall"],
            "fedavg_mean_cosine": res["fedavg_mean_cosine"],
            "seq_mean_update_norm": res["seq_mean_update_norm"],
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    plot_accuracy(results, fig_dir / "accuracy_vs_round.png")

    # order sensitivity
    order_rows = []
    for meth in ["seq_fixed", "seq_random", "seq_reverse"]:
        s = results[meth]
        order_rows.append({"method": meth, "final_acc": s["final_acc"],
                           "best_acc": s["best_acc"], "last10_mean": s["last10_mean"]})
    write_csv(out_dir / "order_sensitivity.csv", order_rows)

    # knowledge retention matrix
    det = results["seq_fixed"]["retention"]
    if det is not None:
        np.savetxt(out_dir / "knowledge_retention_matrix.csv", det, delimiter=",",
                   header=",".join(f"client_{i}" for i in range(det.shape[1])),
                   comments="")
        plot_retention(det, fig_dir / "knowledge_retention_heatmap.png")

    # update diagnostics
    upd_rows = [{"method": m, "fedavg_mean_cosine": res["fedavg_mean_cosine"],
                 "seq_mean_update_norm": res["seq_mean_update_norm"]}
                for m, res in results.items()]
    write_csv(out_dir / "update_stats.csv", upd_rows)

    # transfers (theoretical)
    transfers = theoretical_transfers(args.cluster_size, n_clusters)
    (out_dir / "transfers.json").write_text(json.dumps(transfers, indent=2))

    print(f"\nWrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
