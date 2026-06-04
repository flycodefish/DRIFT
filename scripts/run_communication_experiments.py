#!/usr/bin/env python3
"""
通信成本对比实验 — 批量运行并生成对比图表

运行 DE-ASFL (hierarchical + flat)、FedAvg、Gossip，统计每轮通信量。

用法:
    python run_communication_experiments.py              # 全部运行（~2小时）
    python run_communication_experiments.py --list       # 仅列出计划
    python run_communication_experiments.py --from-existing  # 从已有实验目录读取
    python run_communication_experiments.py --only deasfl  # 仅DE-ASFL
"""

import os
import sys
import subprocess
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.absolute()

# ============================================================
# 已有实验目录（数据异构实验 alpha=0.1）
# ============================================================
EXISTING_DIRS = {
    "DE-ASFL (分层)": "outputs/de_asfl_mesh_30c_100r_20260507_093405",
    "FedAvg": "outputs/fedavg_cifar10_20260507_104846",
    "Gossip": "outputs/gossip_cifar10_20260507_121046",
}

PARAMS = {
    "num_clients": 30,
    "rounds": 100,
    "model_bytes": 44734408,  # ResNet18 state_dict size in bytes
}

# ============================================================
# DE-ASFL 实验定义
# ============================================================
DEASFL_EXPERIMENTS = [
    {
        "config": "config_de_asfl.yaml",
        "label": "DE-ASFL (分层)",
        "desc": "选举+聚类+分层聚合",
    },
    {
        "config": "config_de_asfl_no_cluster.yaml",
        "label": "DE-ASFL (扁平, 有选举)",
        "desc": "仅选举+扁平聚合",
    },
    {
        "config": "config_de_asfl_no_election_no_cluster.yaml",
        "label": "DE-ASFL (扁平, 无选举)",
        "desc": "client 0 扁平聚合",
    },
]


def run_one(config_path: str, label: str) -> dict:
    """运行单个 DE-ASFL 实验"""
    start = time.time()
    print(f"\n{'='*70}")
    print(f"运行: {label}")
    print(f"配置: {config_path}")
    print(f"{'='*70}")

    cmd = [sys.executable, str(BASE_DIR / "test_30_clients.py"), "--config", config_path]
    try:
        result = subprocess.run(
            cmd, cwd=str(BASE_DIR), capture_output=True, text=True,
            timeout=7200,
            env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )
        elapsed = time.time() - start

        if result.returncode != 0:
            print(f"  FAILED (exit {result.returncode})")
            for line in result.stderr.strip().split('\n')[-5:]:
                print(f"  [stderr] {line.strip()[:140]}")
            return {"status": "failed", "label": label, "elapsed": elapsed}

        # 提取输出目录
        output_dir = None
        for line in result.stdout.strip().split('\n'):
            if "输出目录:" in line:
                output_dir = line.split("输出目录:")[-1].strip()
                break

        if output_dir is None:
            prefix = "de_asfl_"
            outputs_dir = BASE_DIR / "outputs"
            candidates = sorted(
                [d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)],
                key=lambda p: p.stat().st_mtime, reverse=True
            )
            if candidates:
                output_dir = str(candidates[0])

        print(f"  DONE ({elapsed:.0f}s) → {output_dir}")
        return {"status": "success", "label": label, "elapsed": elapsed,
                "output_dir": output_dir}

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"  TIMEOUT")
        return {"status": "timeout", "label": label, "elapsed": elapsed}
    except Exception as e:
        elapsed = time.time() - start
        print(f"  ERROR: {e}")
        return {"status": "error", "label": label, "elapsed": elapsed, "error": str(e)}


def parse_deasfl_comm_csv(csv_path: str) -> dict:
    """解析 DE-ASFL 的 communication_stats.csv"""
    df = pd.read_csv(csv_path)
    return {
        "total_msgs": int(df["total_msgs"].sum()),
        "total_bytes": int(df["total_bytes"].sum()),
        "avg_msgs_per_round": float(df["total_msgs"].mean()),
        "model_bytes": int(df["model_bytes"].iloc[-1]) if len(df) > 0 else 0,
        "intra_cluster_msgs": int(df["intra_cluster_msgs"].sum()),
        "inter_fwd_hops": int(df["inter_cluster_forward_hops"].sum()),
        "inter_ret_hops": int(df["inter_cluster_return_hops"].sum()),
        "intra_dist_msgs": int(df["intra_cluster_dist_msgs"].sum()),
        "flat_upload": int(df["flat_upload_msgs"].sum()),
        "flat_download": int(df["flat_download_msgs"].sum()),
    }


def parse_fedavg_gossip_comm_csv(csv_path: str) -> dict:
    """解析 FedAvg/Gossip 的 comm_events.csv"""
    df = pd.read_csv(csv_path)
    delivered = df[df["delivered"] == True]

    by_kind = delivered.groupby("kind")
    kind_counts = {}
    for kind, group in by_kind:
        kind_counts[kind] = len(group)

    total_msgs = sum(kind_counts.values())
    total_bytes = int(delivered["payload_bytes"].sum()) if len(delivered) > 0 else 0

    return {
        "total_msgs": total_msgs,
        "total_bytes": total_bytes,
        "avg_msgs_per_round": total_msgs / PARAMS["rounds"],
        "model_bytes": PARAMS["model_bytes"],
        "kind_breakdown": kind_counts,
    }


def find_comm_csv(output_dir: str, prefix: str = "de_asfl_") -> tuple:
    """在输出目录中查找通信统计 CSV"""
    if not output_dir or not os.path.isdir(output_dir):
        return None, None
    for f in os.listdir(output_dir):
        if f.endswith("_communication_stats.csv"):
            return os.path.join(output_dir, f), "deasfl"
        if f.endswith("_comm_events.csv"):
            return os.path.join(output_dir, f), "fedavg_gossip"
    return None, None


def generate_comparison(deasfl_results: dict, fedavg_result: dict, gossip_result: dict):
    """生成通信成本对比表和图表"""
    print("\n" + "=" * 70)
    print("通信成本对比")
    print("=" * 70)

    rows = []
    labels_map = {
        "deasfl_hier": "DE-ASFL (分层)",
        "deasfl_flat_election": "DE-ASFL (扁平+选举)",
        "deasfl_flat_noelection": "DE-ASFL (扁平, 无选举)",
        "fedavg": "FedAvg",
        "gossip": "Gossip",
    }

    for key, label in labels_map.items():
        if key.startswith("deasfl"):
            result = deasfl_results.get(key)
        elif key == "fedavg":
            result = fedavg_result
        elif key == "gossip":
            result = gossip_result
        else:
            continue
        if result is None:
            continue

        row = {
            "算法": label,
            "总消息数": result.get("total_msgs", 0),
            "总字节数 (MB)": result.get("total_bytes", 0) / (1024**2),
            "每轮平均消息数": result.get("avg_msgs_per_round", 0),
            "每轮平均 (MB)": result.get("avg_msgs_per_round", 0) * PARAMS["model_bytes"] / (1024**2),
        }

        if "intra_cluster_msgs" in result:
            row["簇内上传"] = result.get("intra_cluster_msgs", 0)
            row["簇间转发"] = result.get("inter_fwd_hops", 0)
            row["簇间返回"] = result.get("inter_ret_hops", 0)
            row["簇内分发"] = result.get("intra_dist_msgs", 0)

        if "kind_breakdown" in result:
            for kind, count in result["kind_breakdown"].items():
                row[f"类型:{kind}"] = count

        rows.append(row)

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))

    # 保存 CSV
    out_path = BASE_DIR / "outputs" / "communication_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"\n对比表已保存: {out_path}")

    # 生成对比图
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 图1: Total Messages
        ax = axes[0]
        algorithm_labels = df["算法"].tolist()
        # Use short ASCII-safe labels for chart
        short_labels = [l.replace("DE-ASFL", "DE").replace(" (分层)", "-Hier")
                        .replace(" (扁平+选举)", "-Flat+Elec").replace(" (扁平, 无选举)", "-Flat")
                        .replace(" (", "\n(") for l in algorithm_labels]
        total_msgs = df["总消息数"].tolist()
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0'][:len(short_labels)]
        bars = ax.bar(short_labels, total_msgs, color=colors, edgecolor='white')
        ax.set_ylabel("Total Messages (100 rounds)")
        ax.set_title("Communication Message Count Comparison")
        ax.tick_params(axis='x', rotation=0)
        for bar, val in zip(bars, total_msgs):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 100,
                    f'{val:,}', ha='center', va='bottom', fontsize=9)

        # 图2: Per-Round Messages
        ax = axes[1]
        per_round = [m / PARAMS["rounds"] for m in total_msgs]
        bars = ax.bar(short_labels, per_round, color=colors, edgecolor='white')
        ax.set_ylabel("Avg Messages per Round")
        ax.set_title(f"Per-Round Messages ({PARAMS['num_clients']} clients x {PARAMS['rounds']} rounds)")
        ax.tick_params(axis='x', rotation=0)
        for bar, val in zip(bars, per_round):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        chart_path = BASE_DIR / "outputs" / "communication_comparison.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"对比图已保存: {chart_path}")

    except Exception as e:
        print(f"  Warning: 生成图表失败: {e}")

    return df


def main():
    parser = argparse.ArgumentParser(description="通信成本对比实验")
    parser.add_argument("--list", action="store_true", help="仅列出计划")
    parser.add_argument("--from-existing", action="store_true",
                        help="从已有实验目录读取（不运行新实验）")
    parser.add_argument("--only", choices=["deasfl", "fedavg", "gossip"],
                        help="仅运行指定算法")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="跳过确认提示")
    args = parser.parse_args()

    if args.from_existing:
        print("=" * 70)
        print("从已有实验目录读取通信统计")
        print("=" * 70)

        deasfl_results = {}
        for key, dir_path in EXISTING_DIRS.items():
            full_path = BASE_DIR / dir_path
            csv_path, csv_type = find_comm_csv(str(full_path))

            if csv_path is None:
                print(f"  {key}: ⚠️  未找到通信统计文件（需要重新运行）")
                continue

            if csv_type == "deasfl":
                stats = parse_deasfl_comm_csv(csv_path)
                deasfl_results["deasfl_hier"] = stats
            elif csv_type == "fedavg_gossip":
                stats = parse_fedavg_gossip_comm_csv(csv_path)
                if key == "FedAvg":
                    fedavg_result = stats
                elif key == "Gossip":
                    gossip_result = stats

            print(f"  {key}: {stats['total_msgs']:,} 条消息, "
                  f"{stats['total_bytes']/(1024**2):.1f} MB")

        # 对于没有DE-ASFL通信统计的情况，使用理论计算
        if not deasfl_results:
            print("\n⚠️  DE-ASFL 的通信统计不存在（旧实验没有记录），使用理论估算：")
            print("  理论值基于聚类树结构和选举路由计算。")
            print("  建议运行: python run_communication_experiments.py 生成实际数据。")
            # 理论估算
            deasfl_results = estimate_deasfl_comm()

        generate_comparison(deasfl_results, fedavg_result, gossip_result)
        return

    # 运行模式
    experiments_to_run = []

    if not args.only or args.only == "deasfl":
        experiments_to_run.extend(DEASFL_EXPERIMENTS)

    print("=" * 70)
    print(f"通信成本对比实验（共 {len(experiments_to_run)} 组）")
    print("=" * 70)
    total_est = 0
    for i, exp in enumerate(experiments_to_run):
        est = 25  # ~25min per DE-ASFL run
        total_est += est
        print(f"  [{i+1}] {exp['label']} (~{est}min)")

    # FedAvg and Gossip as reference
    if not args.only or args.only in ("fedavg", "gossip"):
        print(f"\n  FedAvg / Gossip: 使用已有实验数据 (alpha=0.1)")

    print(f"\n预计总耗时: ~{total_est/60:.1f} 小时")

    if args.list:
        return

    if not args.yes:
        choice = input("\n是否开始运行？[y/N] ").strip().lower()
        if choice not in ('y', 'yes'):
            print("已取消")
            return

    # 运行实验
    deasfl_results = {}
    for i, exp in enumerate(experiments_to_run):
        print(f"\n{'#'*70}")
        print(f"实验 [{i+1}/{len(experiments_to_run)}]")
        print(f"{'#'*70}")

        result = run_one(exp["config"], exp["label"])

        if result["status"] == "success" and result.get("output_dir"):
            csv_path, csv_type = find_comm_csv(result["output_dir"])
            if csv_path and csv_type == "deasfl":
                stats = parse_deasfl_comm_csv(csv_path)
                key_map = {exp["config"]: exp["label"] for exp in DEASFL_EXPERIMENTS}
                label = key_map.get(exp["config"], exp["label"])
                if "扁平" in label and "选举" not in label:
                    deasfl_results["deasfl_flat_noelection"] = stats
                elif "扁平" in label:
                    deasfl_results["deasfl_flat_election"] = stats
                else:
                    deasfl_results["deasfl_hier"] = stats
                print(f"  📡 通信: {stats['total_msgs']:,} 条消息, "
                      f"{stats['total_bytes']/(1024**2):.1f} MB")

    # 读取 FedAvg/Gossip 通信数据
    fedavg_result = None
    gossip_result = None
    fedavg_dir = BASE_DIR / EXISTING_DIRS["FedAvg"]
    gossip_dir = BASE_DIR / EXISTING_DIRS["Gossip"]

    fedavg_csv, _ = find_comm_csv(str(fedavg_dir))
    gossip_csv, _ = find_comm_csv(str(gossip_dir))

    if fedavg_csv:
        fedavg_result = parse_fedavg_gossip_comm_csv(fedavg_csv)
        print(f"\n  FedAvg (已有): {fedavg_result['total_msgs']:,} 条消息")
    if gossip_csv:
        gossip_result = parse_fedavg_gossip_comm_csv(gossip_csv)
        print(f"  Gossip (已有): {gossip_result['total_msgs']:,} 条消息")

    if deasfl_results or fedavg_result or gossip_result:
        generate_comparison(deasfl_results, fedavg_result, gossip_result)


def estimate_deasfl_comm() -> dict:
    """理论估算 DE-ASFL 通信量（基于典型 Mesh 30c 聚类结果）"""
    # 典型值：30个节点，Mesh拓扑，~6个簇首，~4个节点/簇
    # 簇内聚合: 每节点1条上传消息（叶子→父）= 30 - 6 = 24
    # 簇间转发: 每个簇首~2跳 = 6 * 2 = 12
    # 簇间返回: 同转发 = 12
    # 簇内分发: 每节点1条（父→子）= 30 - 6 = 24
    # 总计: 24 + 12 + 12 + 24 = 72 条/轮
    # 100轮: 7200 条
    model_bytes = PARAMS["model_bytes"]
    per_round_msgs = 72
    total_msgs = per_round_msgs * PARAMS["rounds"]
    return {
        "deasfl_hier": {
            "total_msgs": total_msgs,
            "total_bytes": total_msgs * model_bytes,
            "avg_msgs_per_round": float(per_round_msgs),
            "model_bytes": model_bytes,
            "intra_cluster_msgs": 24 * PARAMS["rounds"],
            "inter_fwd_hops": 12 * PARAMS["rounds"],
            "inter_ret_hops": 12 * PARAMS["rounds"],
            "intra_dist_msgs": 24 * PARAMS["rounds"],
        }
    }


if __name__ == "__main__":
    main()
