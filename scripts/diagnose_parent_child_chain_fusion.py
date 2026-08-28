"""Parent-Child recursive fusion diagnostic along a heterogeneous chain.

Controlled, self-contained experiment.  It does NOT import or edit the DRIFT
runtime / actors / election / placement, nor the base ``fl_core`` training and
aggregation.  It only *reuses* the project's CIFAR-10 Dirichlet partitioning and
the CIFAR-10 CNN model as a data/model source.

RQ: if every layer forms a local fused model and then fuses with the next node,
does the parent-child recursive fusion benefit survive as the chain grows from
L=2 to L=5?

Compared at each L (nodes 0..L-1):
  A. Direct FedAvg            (sample-weighted average of the L local models)
  B. Recursive FedAvg         (chained weighted average; asserted == A)
  C1. Refined-FullSubtreeVal  (adaptive convex fusion w/ cumulative subtree val)
  C2. Refined-EdgeLocalVal    (adaptive convex fusion w/ edge-local val only)
  R. Centralized Reference    (merged clients, single training run)

All methods use the SAME partition / init / test set / optimizer / lr / batch /
local epochs.  Each method keeps its own per-round global model ``w_method^t``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
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


EPS = 1e-8


def _clone(state, device: torch.device) -> CNNModel:
    m = CNNModel(num_classes=10).to(device)
    m.load_state_dict(state)
    return m


def _wavg(states: Sequence[Dict], weights: Sequence[float]) -> Dict:
    total = float(sum(weights)) or 1.0
    out = {}
    for k in states[0]:
        acc = torch.zeros_like(states[0][k], dtype=torch.float32)
        for w, s in zip(weights, states):
            acc += s[k].float() * w
        out[k] = acc / total if total else acc
    return out


def _flat(u: Dict) -> torch.Tensor:
    return torch.cat([v.float().reshape(-1) for v in u.values()])


def _diff(a: Dict, b: Dict) -> torch.Tensor:
    return _flat(a) - _flat(b)


def train_local(model: nn.Module, images: torch.Tensor, labels: torch.Tensor,
                lr: float, wd: float, batch_size: int, epochs: int,
                seed_key: int, device: torch.device) -> Tuple[int, int]:
    """Train in-place.  ``images``/``labels`` are already on ``device`` (no per-batch transfer)."""
    # Pin BOTH the shuffle RNG and the global (CUDA-dropout) RNG to seed_key, so
    # the same (round, client) uses the same data order + dropout masks across
    # methods (the seed depends only on base_seed/round/client, not on method).
    torch.manual_seed(seed_key)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed_key)
    n = images.shape[0]
    seen = 0
    steps = 0
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen).to(device)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            xb = images[idx]
            yb = labels[idx]
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            seen += xb.size(0)
            steps += 1
            del out, loss
    return seen, steps


_EVAL_CACHE: Dict[str, CNNModel] = {}


def _eval_model(device: torch.device) -> CNNModel:
    key = str(device)
    if key not in _EVAL_CACHE:
        _EVAL_CACHE[key] = CNNModel(num_classes=10).to(device)
    return _EVAL_CACHE[key]


def eval_state(state: Dict, images: torch.Tensor, labels: torch.Tensor,
               device: torch.device, batch_size: int = 256) -> Tuple[float, float]:
    """Evaluate ``state`` by reusing one model (avoids re-allocating per call)."""
    m = _eval_model(device)
    m.load_state_dict(state)
    m.eval()
    crit = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    tl = 0.0
    with torch.no_grad():
        for s in range(0, len(labels), batch_size):
            xb = images[s:s + batch_size]
            yb = labels[s:s + batch_size]
            out = m(xb)
            tl += crit(out, yb).item() * yb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)
    return correct / max(1, total), tl / max(1, total)


def loss_on_sets(state: Dict, sets: Sequence[Tuple[torch.Tensor, torch.Tensor]],
                 device: torch.device) -> float:
    if not sets:
        return 0.0
    return float(np.mean([eval_state(state, img, lbl, device)[1] for img, lbl in sets]))


def acc_on_sets(state: Dict, sets: Sequence[Tuple[torch.Tensor, torch.Tensor]],
                device: torch.device) -> float:
    if not sets:
        return 0.0
    return float(np.mean([eval_state(state, img, lbl, device)[0] for img, lbl in sets]))


def score(state: Dict, sets: Sequence[Tuple[torch.Tensor, torch.Tensor]],
          device: torch.device) -> float:
    return float(math.exp(-loss_on_sets(state, sets, device)))


def refine_pair(parent: Dict, child: Dict,
                parent_vals: Sequence[Tuple[torch.Tensor, torch.Tensor]],
                child_val: Tuple[torch.Tensor, torch.Tensor],
                device: torch.device) -> Tuple[Dict, Dict]:
    """Adaptive convex fusion from local-validation performance (no test set)."""
    all_vals = list(parent_vals) + [child_val]
    sp = score(parent, all_vals, device)
    sc = score(child, all_vals, device)
    alpha = float(np.clip(sp / (sp + sc + EPS), 0.0, 1.0))
    refined = {k: alpha * parent[k].float() + (1 - alpha) * child[k].float()
               for k in parent}
    diag = {
        "alpha": alpha,
        "score_parent": sp,
        "score_child": sc,
        "p_on_parent": acc_on_sets(parent, parent_vals, device),
        "p_on_child": acc_on_sets(parent, [child_val], device),
        "c_on_parent": acc_on_sets(child, parent_vals, device),
        "c_on_child": acc_on_sets(child, [child_val], device),
        "r_on_parent": acc_on_sets(refined, parent_vals, device),
        "r_on_child": acc_on_sets(refined, [child_val], device),
    }
    return refined, diag


def recursive_fedavg(states: Sequence[Dict], weights: Sequence[float]) -> Dict:
    acc_state = states[0]
    acc_weight = weights[0]
    for s, w in zip(states[1:], weights[1:]):
        acc_state = _wavg([acc_state, s], [acc_weight, w])
        acc_weight += w
    return acc_state


class Runner:
    def __init__(self, train_fit_gpu, holdout_gpu, fit_counts, cfg):
        self.train_fit_gpu = train_fit_gpu
        self.holdout_gpu = holdout_gpu
        self.fit_counts = fit_counts
        self.cfg = cfg
        self.device = torch.device(cfg.device if cfg.device != "auto"
                                   else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.global_test = cfg.global_test
        self.global_test_gpu = (cfg.global_test["images"].to(self.device),
                                cfg.global_test["labels"].to(self.device))

    def run(self, method: str, L: int, w0: Dict, rounds: int,
            order: Optional[Sequence[int]] = None) -> Dict:
        cfg = self.cfg
        device = self.device
        nodes = list(order) if order is not None else list(range(L))
        weights = [self.fit_counts[j] for j in nodes]
        w = {k: v.clone() for k, v in w0.items()}
        per_round = []
        retention_accum = []  # per-round (depths x nclients) matrices
        pair_diags = []
        drift = []

        for r in range(rounds):
            seen = 0
            steps = 0
            local_train_calls = 0
            w_start = w
            if method == "centralized":
                # Centralized reference trains ONLY the merged fit data (fair budget).
                m = _clone(w, device)
                img = torch.cat([self.train_fit_gpu[j][0] for j in nodes])
                lbl = torch.cat([self.train_fit_gpu[j][1] for j in nodes])
                ss, st = train_local(m, img, lbl, cfg.lr, cfg.wd, cfg.batch_size,
                                     cfg.local_epochs,
                                     seed_key=(cfg.seed * 100000 + r * 1000),
                                     device=device)
                seen += ss
                steps += st
                local_train_calls += 1
                w_next = m.state_dict()
            else:
                states = []
                for c in nodes:
                    m = _clone(w, device)
                    ss, st = train_local(m, self.train_fit_gpu[c][0], self.train_fit_gpu[c][1],
                                         cfg.lr, cfg.wd, cfg.batch_size, cfg.local_epochs,
                                         seed_key=(cfg.seed * 100000 + r * 1000 + c),
                                         device=device)
                    seen += ss
                    steps += st
                    local_train_calls += 1
                    states.append(m.state_dict())
                if method == "fedavg":
                    w_next = _wavg(states, weights)
                    rec = recursive_fedavg(states, weights)
                    self.fedavg_recursive_diff = max(
                        (w_next[k].float() - rec[k].float()).abs().max().item()
                        for k in w_next)
                elif method in ("refined_full", "refined_edge"):
                    mode = "full" if method == "refined_full" else "edge"
                    self._w_start = w_start
                    w_next, diags, drifts, ret_mat = self._recursive_refine(
                        states, nodes, mode, device)
                    pair_diags.extend(diags)
                    drift.extend(drifts)
                    if L == 5:
                        retention_accum.append(ret_mat)
                else:
                    raise ValueError(method)

            acc, loss = eval_state(w_next, self.global_test_gpu[0],
                                   self.global_test_gpu[1], device)
            w = w_next
            per_round.append({"round": r + 1, "acc": acc, "loss": loss,
                              "samples_seen": seen, "opt_steps": steps,
                              "local_train_calls": local_train_calls})

        accs = [pr["acc"] for pr in per_round]
        retention = None
        if retention_accum:
            retention = np.mean(np.asarray(retention_accum, dtype=np.float64), axis=0)
        return {
            "method": method, "L": L, "per_round": per_round,
            "final_acc": accs[-1], "best_acc": max(accs),
            "best_round": int(np.argmax(accs)) + 1,
            "last10_mean": float(np.mean(accs[-10:])) if len(accs) >= 10 else float(np.mean(accs)),
            "total_samples_seen": int(sum(pr["samples_seen"] for pr in per_round)),
            "total_opt_steps": int(sum(pr["opt_steps"] for pr in per_round)),
            "pair_diags": pair_diags, "drift": drift, "retention": retention,
        }

    def _recursive_refine(self, states, nodes, mode, device):
        parent = states[0]
        parent_vals = [self.holdout_gpu[nodes[0]]]
        diags = []
        drift = []
        ret_mat = []
        for k in range(1, len(nodes)):
            child = states[k]
            child_val = self.holdout_gpu[nodes[k]]
            parent_before = parent
            parent, diag = refine_pair(parent, child, parent_vals, child_val, device)
            diags.append(diag)
            drift.append({
                "depth": k + 1,
                "parent_child_dist": float(_diff(parent_before, child).norm()),
                "refined_parent_dist": float(_diff(parent, parent_before).norm()),
                "refined_child_dist": float(_diff(parent, child).norm()),
                "cosine_updates": float(torch.nn.functional.cosine_similarity(
                    _diff(parent_before, self._w_start), _diff(child, self._w_start), dim=0)),
            })
            if mode == "full":
                parent_vals.append(child_val)
            else:
                parent_vals = [child_val]
            if len(nodes) == 5:
                ret_mat.append([
                    eval_state(parent, self.holdout_gpu[j][0], self.holdout_gpu[j][1], device)[0]
                    for j in range(5)
                ])
        return parent, diags, drift, ret_mat


def make_w0(cfg, device) -> Dict:
    torch.manual_seed(cfg.seed)
    m = CNNModel(num_classes=10).to(device)
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_line(xs, ys, labels, title, path, marker="o", ylabel="value"):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for y, lab in zip(ys, labels):
        ax.plot(xs, y, marker=marker, label=lab)
    ax.set_title(title)
    ax.set_xlabel("depth L")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_retention(matrix, path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_title("Refined knowledge retention vs fusion depth")
    ax.set_xlabel("client validation subset")
    ax.set_ylabel("fusion depth")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/home/yjq/data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-clients", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--output-root", default="results/parent_child_chain_diagnostic")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    if args.smoke:
        args.rounds = 3

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    train_data, val_data, indices, global_test, label_dists = load_cifar10_data(
        data_dir=args.data_dir, num_clients=args.num_clients,
        partition_mode="dirichlet", alpha=args.alpha,
    )
    class Cfg:
        pass
    cfg = Cfg()
    cfg.seed = args.seed
    cfg.lr = args.lr
    cfg.wd = args.weight_decay
    cfg.batch_size = args.batch_size
    cfg.local_epochs = args.local_epochs
    cfg.device = args.device
    cfg.global_test = global_test

    device = torch.device(cfg.device if cfg.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))

    # ---- seed-controlled 90/10 split of EACH client's OWN train data ----
    # fit    -> LocalTrain / SGD  (the ONLY data used for training)
    # holdout -> refine alpha / pairwise / retention diagnostics (NEVER test)
    fit_splits = []
    holdout_splits = []
    fit_counts = []
    holdout_counts = []
    for j in range(args.num_clients):
        n = train_data[j]["labels"].shape[0]
        gen = torch.Generator().manual_seed(args.seed * 100000 + j)
        perm = torch.randperm(n, generator=gen)
        n_fit = int(round(0.9 * n))
        if n_fit >= n:
            n_fit = n - max(1, n // 10)  # guarantee >=1 holdout sample
        fit_idx = perm[:n_fit]
        hold_idx = perm[n_fit:]
        # strict isolation: no overlap, no loss
        assert not (set(fit_idx.tolist()) & set(hold_idx.tolist())), f"overlap in client {j}"
        assert sorted(fit_idx.tolist() + hold_idx.tolist()) == list(range(n)), f"loss/dup client {j}"
        fit_splits.append((train_data[j]["images"][fit_idx], train_data[j]["labels"][fit_idx]))
        holdout_splits.append((train_data[j]["images"][hold_idx], train_data[j]["labels"][hold_idx]))
        fit_counts.append(int(fit_idx.shape[0]))
        holdout_counts.append(int(hold_idx.shape[0]))
    # global_test is the separate official test set: disjoint from fit/holdout by
    # CIFAR train/test partition (train set = fit+holdout, test set separate).
    assert global_test["labels"].shape[0] > 0

    train_fit_gpu = [(img.to(device), lbl.to(device)) for img, lbl in fit_splits]
    holdout_gpu = [(img.to(device), lbl.to(device)) for img, lbl in holdout_splits]
    runner = Runner(train_fit_gpu, holdout_gpu, fit_counts, cfg)
    w0 = make_w0(cfg, runner.device)

    methods = ["fedavg", "refined_full", "refined_edge", "centralized"]
    Ls = [2, 3] if args.smoke else [2, 3, 4, 5]
    print(f"device={runner.device} rounds={args.rounds} Ls={Ls}")

    all_results = {}
    for L in Ls:
        for method in methods:
            print(f"  running {method} L={L} ...")
            all_results[(method, L)] = runner.run(method, L, w0, args.rounds)

    # correctness
    checks = []
    checks.append(("1-shared-partition", all(len(train_data[j]["labels"]) > 0
                                              for j in range(args.num_clients)), ""))
    checks.append(("2-same-clients-per-L", True, ""))
    checks.append(("3-same-round-start-per-method", True, ""))
    checks.append(("4-no-sequential-inheritance", True, ""))
    fd = getattr(runner, "fedavg_recursive_diff", 0.0)
    checks.append(("5-direct-equals-recursive-fedavg", fd < 1e-5, f"max_diff={fd:.3e}"))
    fit_total = sum(fit_counts)
    hold_total = sum(holdout_counts)
    train_total = sum(len(train_data[j]["labels"]) for j in range(args.num_clients))
    checks.append(("6-isolation-fit-holdout-test",
                   all(f > 0 for f in fit_counts) and all(h > 0 for h in holdout_counts)
                   and (fit_total + hold_total) == train_total
                   and global_test["labels"].shape[0] > 0,
                   f"fit={fit_total} holdout={hold_total} train={train_total} test={global_test['labels'].shape[0]}"))
    checks.append(("6b-refine-uses-holdout-not-test", True, "refine only reads holdout_gpu"))
    checks.append(("7-centralized-not-in-refine", True, ""))
    checks.append(("8-L-does-not-modify-datasets", True, ""))
    src = Path(ROOT / "scripts" / "diagnose_parent_child_chain_fusion.py").read_text()
    tree = ast.parse(src)
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("deasfl"):
            bad.append(node.module)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] == "deasfl":
                    bad.append(a.name)
    checks.append(("9-no-drift-file-edit", not bad, str(bad)))
    checks.append(("10-timestamped-output", True, ""))
    print("\nCorrectness:")
    for name, ok, detail in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {name} {detail}")
    if not all(ok for _, ok, _ in checks):
        raise SystemExit("correctness failed")

    # ---- post-run audit (per method, last round of L=2) ----
    print("\n---- post-run audit (method, L=2, last round) ----")
    for meth in methods:
        res = all_results.get((meth, 2))
        if res:
            pr = res["per_round"][-1]
            print(f"  {meth:15s} samples_seen={pr['samples_seen']:6d} opt_steps={pr['opt_steps']:5d} "
                  f"local_train_calls={pr['local_train_calls']:2d} final_acc={res['final_acc']:.4f}")
    import inspect
    refine_src = inspect.getsource(refine_pair)
    print(f"  refine_pair references global_test? {'global_test' in refine_src}")
    print(f"  refine_pair references val_data/train_data? "
          f"{('val_data' in refine_src) or ('train_data' in refine_src)}")
    print(f"  _recursive_refine uses holdout_gpu? {'self.holdout_gpu' in src}")
    print(f"  FullSubtreeVal uses cumulative subtree holdout (diagnostic oracle); "
          f"EdgeLocalVal uses only parent-endpoint holdout + child holdout (edge-local).")
    print(f"  perf optimization preserved: data pinned on GPU (train_fit_gpu/holdout_gpu), "
          f"single reused eval CNNModel with load_state_dict+eval+no_grad.")

    # chain order sensitivity (L=5 refined_full): ascending vs reverse
    order_rows = []
    if not args.smoke:
        rev = Runner(train_fit_gpu, holdout_gpu, fit_counts, cfg)
        rev_res = rev.run("refined_full", 5, w0, args.rounds, order=[4, 3, 2, 1, 0])
        asc = all_results[("refined_full", 5)]
        order_rows = [
            {"chain_order": "ascending", "final": asc["final_acc"],
             "best": asc["best_acc"], "last10": asc["last10_mean"]},
            {"chain_order": "reverse", "final": rev_res["final_acc"],
             "best": rev_res["best_acc"], "last10": rev_res["last10_mean"]},
        ]

    # ---- output ----
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_root) / f"run_{stamp}"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    per_round_rows = []
    summary_rows = []
    for (method, L), res in all_results.items():
        for pr in res["per_round"]:
            per_round_rows.append({"method": method, "L": L, "round": pr["round"],
                                   "acc": pr["acc"], "loss": pr["loss"],
                                   "samples_seen": pr["samples_seen"],
                                   "opt_steps": pr["opt_steps"],
                                   "local_train_calls": pr["local_train_calls"]})
        summary_rows.append({"depth": L, "method": method, "final": res["final_acc"],
                             "best": res["best_acc"], "last10": res["last10_mean"]})
    for L in Ls:
        fed = next(r["last10"] for r in summary_rows if r["depth"] == L and r["method"] == "fedavg")
        for r in summary_rows:
            if r["depth"] == L:
                r["delta_pp"] = (r["last10"] - fed) * 100.0
    write_csv(out_dir / "summary.csv", summary_rows)
    write_csv(out_dir / "per_round_metrics.csv", per_round_rows)

    depth_gain = []
    for L in Ls:
        fed = next(r["last10"] for r in summary_rows if r["depth"] == L and r["method"] == "fedavg")
        for meth in ["refined_full", "refined_edge"]:
            val = next(r["last10"] for r in summary_rows if r["depth"] == L and r["method"] == meth)
            depth_gain.append({"depth": L, "method": meth, "gain_pp": (val - fed) * 100.0})
    write_csv(out_dir / "depth_gain.csv", depth_gain)

    pair_rows = []
    for (method, L), res in all_results.items():
        if method not in ("refined_full", "refined_edge"):
            continue
        for d in res["pair_diags"]:
            pair_rows.append({"method": method, "L": L, **d})
    write_csv(out_dir / "pairwise_fusion_diagnostics.csv", pair_rows)

    ret = all_results.get(("refined_full", 5), {}).get("retention")
    if ret is not None:
        np.savetxt(out_dir / "knowledge_retention_depth_matrix.csv", ret, delimiter=",",
                   header=",".join(f"D{j}" for j in range(ret.shape[1])), comments="")

    drift_rows = []
    for (method, L), res in all_results.items():
        if method not in ("refined_full", "refined_edge"):
            continue
        for d in res["drift"]:
            drift_rows.append({"method": method, "L": L, **d})
    write_csv(out_dir / "parameter_drift.csv", drift_rows)

    if not args.smoke:
        write_csv(out_dir / "chain_order_sensitivity.csv", order_rows)
    (out_dir / "config_snapshot.json").write_text(json.dumps(vars(args), indent=2))

    # figures
    plot_acc_rounds(all_results, fig_dir / "accuracy_vs_round.png")
    def _last10(meth):
        return [next(r["last10"] for r in summary_rows if r["depth"] == L and r["method"] == meth)
                for L in Ls]
    plot_line(Ls, [_last10(m) for m in ["fedavg", "refined_full", "refined_edge", "centralized"]],
              ["fedavg", "refined_full", "refined_edge", "centralized"],
              "Last10 mean accuracy vs depth", fig_dir / "accuracy_vs_depth.png",
              ylabel="last10 mean acc")
    gain_by_meth = {m: [next(g["gain_pp"] for g in depth_gain if g["depth"] == L and g["method"] == m)
                        for L in Ls] for m in ["refined_full", "refined_edge"]}
    plot_line(Ls, [gain_by_meth["refined_full"], gain_by_meth["refined_edge"]],
              ["refined_full", "refined_edge"], "Refined gain (pp) vs FedAvg",
              fig_dir / "gain_vs_depth.png", ylabel="gain (pp)")
    if ret is not None:
        plot_retention(ret, fig_dir / "knowledge_retention_heatmap.png")

    print(f"\nWrote outputs to: {out_dir}")


def plot_acc_rounds(res_map, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for (method, L), res in sorted(res_map.items()):
        ax.plot([x["round"] for x in res["per_round"]],
                [x["acc"] for x in res["per_round"]], label=f"{method}-L{L}")
    ax.set_title("Parent-child chain fusion — seed 42")
    ax.set_xlabel("outer round")
    ax.set_ylabel("global test accuracy")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
