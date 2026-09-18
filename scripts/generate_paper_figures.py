"""Generate publication-ready TMC figures from completed experiment artifacts.

This script is deliberately read-only with respect to training artifacts.  It
creates a separate paper-figures directory containing PDF, 300-DPI PNG, and
the exact CSV values used for every figure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


SINGLE = (3.5, 2.65)
DOUBLE = (7.16, 3.75)
FONT = "Times New Roman"
COLORS = {
    "fedavg": "#3B6FB6", "fedavg_edge": "#343A40", "dpsgd": "#E69F00",
    "dynamic_dpsgd": "#3B6FB6", "fixed_hierarchical": "#8C8C8C",
    "resource_tree": "#009E73", "epidemic_learning": "#8E6C8A",
    "drift_no_resource_awareness": "#9C755F", "drift": "#D1495B",
}
LABELS = {
    "fedavg": "FedAvg", "fedavg_edge": "FedAvg-Edge", "dpsgd": "D-PSGD",
    "dynamic_dpsgd": "Dynamic D-PSGD", "fixed_hierarchical": "Fixed HierFL",
    "resource_tree": "Resource-Tree", "epidemic_learning": "Epidemic Learning",
    "drift_no_resource_awareness": "DRIFT-NoResource", "drift": "DRIFT",
}
MARKERS = {
    "fedavg": "o", "fedavg_edge": "s", "dpsgd": "^", "dynamic_dpsgd": "^",
    "fixed_hierarchical": "D", "resource_tree": "P",
    "epidemic_learning": "X", "drift_no_resource_awareness": "v", "drift": "o",
}
LINESTYLES = {
    "fedavg": "--", "fedavg_edge": "-", "dpsgd": "-.", "dynamic_dpsgd": "-.",
    "fixed_hierarchical": (0, (5, 2)), "resource_tree": (0, (2, 1)),
    "epidemic_learning": (0, (1, 1)), "drift_no_resource_awareness": "--", "drift": "-",
}


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": [FONT, "DejaVu Serif"],
        "mathtext.fontset": "stix", "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
        "lines.linewidth": 1.7, "axes.linewidth": 0.8, "grid.linewidth": 0.5,
        "axes.edgecolor": "#4A4A4A", "axes.labelcolor": "#222222",
        "axes.facecolor": "#F7F8FA", "figure.facecolor": "white",
        "xtick.color": "#333333", "ytick.color": "#333333",
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
    })


def polish(axis: plt.Axes, grid_axis: str = "y") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis=grid_axis, color="white", linewidth=.9, alpha=1.0)
    axis.set_axisbelow(True)


def save_figure(fig: plt.Figure, output: Path, data: list[dict[str, Any]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    if data:
        fields = list(dict.fromkeys(key for row in data for key in row))
        with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(data)
    plt.close(fig)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def curve(run: Path, field: str = "val_acc") -> list[float]:
    files = list(run.glob("*_round_metrics.csv"))
    if len(files) != 1:
        raise ValueError(f"Expected one round CSV below {run}, found {len(files)}")
    return [float(row[field]) for row in read_csv(files[0])]


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    values = list(values)
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def compact_number(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.1f}"
    return f"{value:.2f}"


def matrix_runs(root: Path, methods: Iterable[str]) -> dict[str, dict[int, Path]]:
    result: dict[str, dict[int, Path]] = {method: {} for method in methods}
    for seed_dir in sorted(root.glob("seed_*")):
        seed = int(seed_dir.name.split("_")[-1])
        for method in methods:
            candidates = list((seed_dir / "runs" / method).glob("*"))
            complete = [p for p in candidates if (p / "config" / "completion_status.json").is_file()]
            if complete:
                result[method][seed] = max(complete, key=lambda p: p.stat().st_mtime)
    return result


def plot_e1(root: Path, out: Path) -> None:
    status = json.loads((root / "matrix_status.json").read_text(encoding="utf-8"))
    fig, (ax, delta_ax) = plt.subplots(
        1, 2, figsize=DOUBLE, gridspec_kw={"width_ratios": (1.7, 1)},
    )
    data = []
    curves = {}
    for item in status:
        method, run = item["method"], Path(item["run"])
        values = curve(run); curves[method] = values
        data.extend({"method": method, "round": i, "accuracy": value}
                    for i, value in enumerate(values, 1))
    common = curves["fedavg"]
    ax.plot(range(1, len(common) + 1), common, color="#2F5597", marker="o",
            markevery=2, markersize=3.2, label="All four methods overlap")
    ax.set(xlabel="Communication round", ylabel="Global test accuracy", title="(a) Shared learning trajectory")
    polish(ax); ax.legend(loc="lower right", frameon=False)
    methods = ("resource_tree", "fixed_hierarchical", "drift")
    max_diffs = [max(abs(a-b) for a,b in zip(curves[m], common)) * 100 for m in methods]
    positions = list(range(len(methods)))
    delta_ax.axvline(0, color="#777777", linewidth=.8, zorder=1)
    for y, method, value in zip(positions, methods, max_diffs):
        delta_ax.scatter(value, y, s=34, color=COLORS[method], marker=MARKERS[method], zorder=3)
        delta_ax.annotate(f"{value:.1f}", (value, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=7.5)
        for row in data:
            if row["method"] == method:
                row["max_abs_difference_to_fedavg_pp"] = value
    delta_ax.set_yticks(positions, [LABELS[m] for m in methods]); delta_ax.set_xlim(-.005, .055)
    delta_ax.set(xlabel="Max. absolute difference (pp)", title="(b) Numerical equivalence")
    delta_ax.invert_yaxis(); polish(delta_ax, "x")
    fig.tight_layout(w_pad=1.5)
    save_figure(fig, out / "e1_learning_semantics", data)


def load_analysis_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("records", [])


def plot_e2(root: Path, out: Path) -> None:
    analysis = root / "analysis" / "analysis_summary.json"
    payload = json.loads(analysis.read_text(encoding="utf-8"))
    records = payload["records"]
    grouped: dict[str, list[list[float]]] = {}
    by_method_seed: dict[str, dict[int, list[float]]] = {}
    data = []
    for item in records:
        method, run = item["method"], Path(item["run"])
        values = curve(run); grouped.setdefault(method, []).append(values)
        by_method_seed.setdefault(method, {})[int(item["seed"])] = values
    fig, (ax, gap_ax) = plt.subplots(1, 2, figsize=DOUBLE, gridspec_kw={"width_ratios": (1.15, 1)})
    for method in ("fedavg", "dpsgd", "epidemic_learning"):
        values = grouped[method]
        means = [statistics.fmean(x) for x in zip(*values)]
        stds = [statistics.stdev(x) if len(x) > 1 else 0 for x in zip(*values)]
        rounds = list(range(1, len(means) + 1)); color = COLORS[method]
        label = "FedAvg / Fixed HierFL / Resource-Tree / DRIFT" if method == "fedavg" else LABELS[method]
        ax.plot(rounds, means, color=color, label=label, linestyle=LINESTYLES[method],
                marker=MARKERS[method], markevery=10, markersize=3)
        data.extend({"method": method, "round": r, "mean_accuracy": m, "std_accuracy": s}
                    for r, m, s in zip(rounds, means, stds))
    ax.set(xlabel="Communication round", ylabel="Global test accuracy", title="(a) Full learning trajectory")
    polish(ax); ax.legend(loc="lower right", frameon=False, fontsize=6.6)
    for method in ("dpsgd", "epidemic_learning", "drift"):
        gaps = []
        for seed, values in sorted(by_method_seed[method].items()):
            reference = by_method_seed["fedavg"][seed]
            gaps.append([100 * (a - b) for a, b in zip(reference, values)])
        means = [statistics.fmean(x) for x in zip(*gaps)]
        rounds = list(range(1, len(means) + 1))
        gap_ax.plot(rounds, means, color=COLORS[method], label=LABELS[method],
                    linestyle=LINESTYLES[method], marker=MARKERS[method],
                    markevery=10, markersize=3)
        for row in data:
            if row["method"] == method:
                row["mean_gap_to_fedavg_pp"] = means[int(row["round"]) - 1]
    gap_ax.axhline(0, color="#555555", linewidth=.8)
    gap_ax.set(xlabel="Communication round", ylabel="Gap to FedAvg (pp)", title="(b) Accuracy difference")
    polish(gap_ax); gap_ax.legend(frameon=False, fontsize=6.8)
    fig.tight_layout(w_pad=1.1)
    save_figure(fig, out / "e2_accuracy", data)
    methods = ["fedavg_edge", "dpsgd", "fixed_hierarchical", "resource_tree", "epidemic_learning", "drift"]
    panels = [("mean_round_latency_sim_ms", 1e-3, "Relative latency (%)"),
              ("max_peak_node_traffic_bytes", 1e-9, "Relative peak traffic (%)"),
              ("total_bytes", 1e-9, "Relative total traffic (%)")]
    fig, axes = plt.subplots(1, 3, figsize=DOUBLE)
    rows = []
    for axis, (field, scale, ylabel) in zip(axes, panels):
        ratios = []
        raw_means = []
        for method in methods:
            values = []
            raw = []
            for seed in sorted({int(item["seed"]) for item in records}):
                own = next(item for item in records if item["method"] == method and int(item["seed"]) == seed)
                edge = next(item for item in records if item["method"] == "fedavg_edge" and int(item["seed"]) == seed)
                raw.append(float(own[field]) * scale)
                values.append(100 * float(own[field]) / float(edge[field]))
            ratios.append(statistics.fmean(values)); raw_means.append(statistics.fmean(raw))
        y = list(range(len(methods)))
        bars = axis.barh(y, ratios, color=[COLORS[m] for m in methods],
                         edgecolor="white", linewidth=.8, height=.68)
        axis.bar_label(bars, labels=[f"{value:.0f}%" for value in ratios],
                       padding=2, fontsize=6.5)
        axis.set_yticks(y, [LABELS[m] for m in methods] if axis is axes[0] else [])
        axis.invert_yaxis(); axis.axvline(100, color="#555555", linestyle=":", linewidth=.9)
        axis.set_xlabel(ylabel); polish(axis, "x")
        rows.extend({"method":m, "metric":field, "relative_to_fedavg_edge_percent":v,
                     "raw_mean":raw} for m,v,raw in zip(methods,ratios,raw_means))
    fig.tight_layout(w_pad=1.0)
    save_figure(fig, out / "e2_system_tradeoff", rows)


def plot_e3(root: Path, out: Path) -> None:
    payload = json.loads((root / "e3_resource_awareness" / "analysis" / "analysis_summary.json").read_text(encoding="utf-8"))
    summary = payload["aggregate"]
    methods = ("fixed_hierarchical", "resource_tree", "drift_no_resource_awareness", "drift")
    metrics = [("mean_round_latency_sim_ms", 1e-3, "Mean round latency (s)"),
              ("max_capacity_normalized_load", 1, "Max normalized load"),
              ("capacity_normalized_load_imbalance", 1, "Load imbalance")]
    fig, axes = plt.subplots(1, 3, figsize=DOUBLE); data=[]
    for ax,(metric,scale,ylabel) in zip(axes,metrics):
        for method in methods:
            rows=[r for r in summary if r["method"]==method]
            x=[r["heterogeneity"] for r in rows]; y=[r[f"{metric}_mean"]*scale for r in rows]
            e=[r[f"{metric}_std"]*scale for r in rows]
            ax.plot(x,y,marker=MARKERS[method], label=LABELS[method],
                    color=COLORS[method],linestyle=LINESTYLES[method],markersize=4)
            data.extend({"method":method,"H":a,"mean":b,"std":c} for a,b,c in zip(x,y,e))
        ax.set(xlabel="Compute heterogeneity H",ylabel=ylabel); ax.xaxis.set_major_locator(MaxNLocator(integer=True)); polish(ax)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(.5, 1.02), columnspacing=1.2, handlelength=2.4)
    fig.tight_layout(rect=(0, 0, 1, .91), w_pad=1.0)
    save_figure(fig,out/"e3_resource_awareness",data)
    # Relative improvement over DRIFT-NoResource for the two resource-sensitive metrics.
    fig, ax = plt.subplots(figsize=SINGLE); rows=[]
    for H in (1,5,10):
        base=next(r for r in summary if r["heterogeneity"]==H and r["method"]=="drift_no_resource_awareness")
        drift=next(r for r in summary if r["heterogeneity"]==H and r["method"]=="drift")
        for metric,label in (("mean_round_latency_sim_ms","Latency"),("max_capacity_normalized_load","Max load")):
            value=100*(base[f"{metric}_mean"]-drift[f"{metric}_mean"])/base[f"{metric}_mean"]
            rows.append({"H":H,"metric":label,"improvement_percent":value})
    for metric,color in (("Latency",COLORS["drift"]),("Max load",COLORS["resource_tree"])):
        vals=[r["improvement_percent"] for r in rows if r["metric"]==metric]
        ax.plot((1,5,10),vals,marker="o",label=metric,color=color,markersize=4)
    ax.axhline(0,color="#555",linewidth=.8); ax.set(xlabel="Compute heterogeneity H",ylabel="Improvement over DRIFT-NoResource (%)",title="E3: Resource-awareness gain")
    polish(ax); ax.legend(frameon=False); fig.tight_layout(); save_figure(fig,out/"e3_resource_gain",rows)


def e4_curve_paths(root: Path, seed: int, method: str) -> tuple[Path,Path]:
    fail_base = root / ("pilots/e4_multi_node_failure_v1/e4_multi_node_failure_pilot" if seed==42 else "formal_v1/e4_multi_node_failure_remaining_v1/e4_multi_node_failure_pilot")
    ctrl_base = root / (f"formal_v1/e2_static_tradeoff/seed_{seed}/runs" if method=="fixed_hierarchical" else f"formal_v1/e4_no_failure_control/seed_{seed}/runs")
    def one(base: Path) -> Path:
        dirs=list((base / f"seed_{seed}/runs/{method}" if "seed_" not in str(base) else base / method).glob("*"))
        complete=[p for p in dirs if (p/"config/completion_status.json").is_file()]
        return max(complete,key=lambda p:p.stat().st_mtime)
    return one(fail_base), one(ctrl_base)


def plot_e4(root: Path, out: Path) -> None:
    methods=("fixed_hierarchical","dynamic_dpsgd","epidemic_learning","resource_tree","drift")
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=DOUBLE); data=[]
    for method in methods:
        curves=[]; gaps=[]
        for seed in (42,43,44):
            fail,ctrl=e4_curve_paths(root,seed,method); f=curve(fail); c=curve(ctrl)
            curves.append([100*x for x in f]); gaps.append([100*(c[i]-f[i]) for i in range(min(len(c),len(f)))])
        means=[statistics.fmean(x) for x in zip(*curves)]; stds=[statistics.stdev(x) for x in zip(*curves)]
        gmeans=[statistics.fmean(x) for x in zip(*gaps)]; gstds=[statistics.stdev(x) for x in zip(*gaps)]
        x=list(range(len(means))); color=COLORS[method]; ls=LINESTYLES[method]
        ax1.plot(x,means,color=color,linestyle=ls,label=LABELS[method],linewidth=2 if method=="drift" else 1.4)
        z=list(range(18,min(51,len(gmeans)))); ax2.plot(z,[gmeans[i] for i in z],color=color,linestyle=ls,label=LABELS[method],linewidth=2 if method=="drift" else 1.4)
        data.extend({"method":method,"round":i,"accuracy_mean":m,"accuracy_std":s,"gap_mean":gmeans[i],"gap_std":gstds[i]} for i,(m,s) in enumerate(zip(means,stds)))
    for ax in (ax1,ax2):
        ax.axvspan(20,30,color="#D9D9D9",alpha=.3); ax.axvline(20,color="#555",lw=.7); ax.axvline(30,color="#555",lw=.7); polish(ax)
    ax1.set(xlabel="Round",ylabel="Global test accuracy (%)",title="(a) Transient node failure")
    ax2.axhline(0,color="#333",lw=.7); ax2.set(xlabel="Round",ylabel="Accuracy deficit (pp)",title="(b) Paired deficit",xlim=(18,50))
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(.5,1.02))
    fig.tight_layout(rect=(0,0,1,.91),w_pad=1.2); save_figure(fig,out/"e4_failure_recovery",data)


def plot_e5(root: Path, out: Path) -> None:
    payload=json.loads((root/"e5_scalability/scalability_summary.json").read_text(encoding="utf-8")); records=payload["records"]
    methods=("fedavg_edge","resource_tree","drift"); metrics=(("total_bytes",1e-9,"Total traffic (GB)"),("max_peak_node_traffic_bytes",1e-9,"Peak node traffic (GB)"),("max_capacity_normalized_load",1,"Max normalized load"))
    fig,axes=plt.subplots(1,3,figsize=DOUBLE); data=[]
    for ax,(metric,scale,ylabel) in zip(axes,metrics):
        for method in methods:
            for n in sorted({int(r["num_clients"]) for r in records}):
                vals=[float(r[metric])*scale for r in records if r["method"]==method and int(r["num_clients"])==n]
                mean,std=mean_std(vals); data.append({"method":method,"num_clients":n,"metric":metric,"mean":mean,"std":std})
            selected=[r for r in data if r["method"]==method and r["metric"]==metric]; selected.sort(key=lambda r:r["num_clients"])
            ax.plot([r["num_clients"] for r in selected],[r["mean"] for r in selected],marker=MARKERS[method],label=LABELS[method],color=COLORS[method],linestyle=LINESTYLES[method],markersize=4)
        ax.set(xlabel="Number of clients",ylabel=ylabel); polish(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5,1.02))
    fig.tight_layout(rect=(0,0,1,.92),w_pad=1.0); save_figure(fig,out/"e5_scalability",data)


def plot_e6(root: Path, out: Path) -> None:
    payload=json.loads((root/"e6_har_generalization/matrix_summary.json").read_text(encoding="utf-8")); records=payload["records"]
    methods=("fedavg","dpsgd","resource_tree","epidemic_learning","drift")
    runs=matrix_runs(root/"e6_har_generalization",methods)
    fig,(ax,gap_ax)=plt.subplots(1,2,figsize=DOUBLE,gridspec_kw={"width_ratios":(1.15,1)}); data=[]
    curve_cache: dict[str, dict[int, list[float]]] = {}
    for method in methods:
        curve_cache[method]={seed:curve(run) for seed,run in runs[method].items()}
        curves=list(curve_cache[method].values()); means=[statistics.fmean(x) for x in zip(*curves)]; stds=[statistics.stdev(x) for x in zip(*curves)]
        x=list(range(1,len(means)+1)); ax.plot(x,means,color=COLORS[method],label=LABELS[method],linestyle=LINESTYLES[method],marker=MARKERS[method],markevery=4,markersize=3)
        data.extend({"method":method,"round":i,"accuracy_mean":m,"accuracy_std":s} for i,(m,s) in enumerate(zip(means,stds),1))
    ax.set(xlabel="Communication round",ylabel="HAR test accuracy",title="(a) Full learning trajectory"); polish(ax); ax.legend(frameon=False,ncol=2,loc="lower right",fontsize=6.5)
    for method in ("dpsgd","epidemic_learning","drift"):
        gaps=[]
        for seed,values in sorted(curve_cache[method].items()):
            reference=curve_cache["fedavg"][seed]
            gaps.append([100*(a-b) for a,b in zip(reference,values)])
        means=[statistics.fmean(x) for x in zip(*gaps)]; x=list(range(1,len(means)+1))
        gap_ax.plot(x,means,color=COLORS[method],label=LABELS[method],linestyle=LINESTYLES[method],marker=MARKERS[method],markevery=4,markersize=3)
        for row in data:
            if row["method"]==method: row["mean_gap_to_fedavg_pp"]=means[int(row["round"])-1]
    gap_ax.axhline(0,color="#555555",linewidth=.8)
    gap_ax.set(xlabel="Communication round",ylabel="Gap to FedAvg (pp)",title="(b) Accuracy difference"); polish(gap_ax); gap_ax.legend(frameon=False,fontsize=6.8)
    fig.tight_layout(w_pad=1.1); save_figure(fig,out/"e6_har_accuracy",data)
    # System metrics use the non-edge records for methods and edge record as the central reference.
    selected=[r for r in records if r["method"] in methods or r["method"]=="fedavg_edge"]
    labels=("fedavg_edge","dpsgd","resource_tree","epidemic_learning","drift"); metrics=(("total_bytes",1e-9,"Relative total traffic (%)"),("max_peak_node_traffic_bytes",1e-9,"Relative peak traffic (%)"),("mean_round_latency_sim_ms",1e-3,"Relative latency (%)"))
    fig,axes=plt.subplots(1,3,figsize=DOUBLE); rows=[]
    for ax,(metric,scale,ylabel) in zip(axes,metrics):
        vals=[]; raw_means=[]
        for method in labels:
            ratios=[]; raw=[]
            for seed in (42,43,44):
                own=next(r for r in selected if r["method"]==method and int(r["seed"])==seed)
                edge=next(r for r in selected if r["method"]=="fedavg_edge" and int(r["seed"])==seed)
                raw.append(float(own[metric])*scale); ratios.append(100*float(own[metric])/float(edge[metric]))
            vals.append(statistics.fmean(ratios)); raw_means.append(statistics.fmean(raw))
        y=list(range(len(labels)))
        bars=ax.barh(y,vals,color=[COLORS.get(m,"#555") for m in labels],edgecolor="white",linewidth=.8,height=.68)
        ax.bar_label(bars,labels=[f"{value:.0f}%" for value in vals],
                     padding=2,fontsize=6.5)
        ax.set_yticks(y,[LABELS[m] for m in labels] if ax is axes[0] else []); ax.invert_yaxis(); ax.axvline(100,color="#555555",linestyle=":",linewidth=.9); ax.set_xlabel(ylabel); polish(ax,"x")
        rows.extend({"method":method,"metric":metric,"relative_to_fedavg_edge_percent":ratio,
                     "raw_mean":raw} for method,ratio,raw in zip(labels,vals,raw_means))
    fig.tight_layout(w_pad=1.0); save_figure(fig,out/"e6_har_system_tradeoff",rows)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root",type=Path,default=Path("outputs/formal_v1"))
    parser.add_argument("--output-dir",type=Path,default=Path("outputs/formal_v1/paper_figures"))
    args=parser.parse_args(); configure_style(); root=args.formal_root.resolve(); out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    plot_e1(root/"e1_learning_preservation",out); plot_e2(root/"e2_static_tradeoff",out); plot_e3(root,out); plot_e4(root.parent,out); plot_e5(root,out); plot_e6(root,out)
    print(f"Paper figures written to: {out}")


if __name__ == "__main__":
    main()
