#!/usr/bin/env python3
"""
消融实验 + JSD 阈值扫描 + 拓扑对比 — 批量运行脚本

用法:
    python run_ablation_experiments.py          # 全部运行（9组，~3小时）
    python run_ablation_experiments.py --list   # 仅列出计划
    python run_ablation_experiments.py --only ablation  # 仅消融实验（2组）
    python run_ablation_experiments.py --only jsd       # 仅JSD扫描（4组）
    python run_ablation_experiments.py --only topology  # 仅拓扑（2组）
    python run_ablation_experiments.py --dry-run        # 仅生成配置，不运行
"""

import os
import sys
import subprocess
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent.absolute()

# ============================================================
# 实验定义
# ============================================================

EXPERIMENTS = {
    # ---- 消融实验 ----
    "ablation": [
        {
            "config": "config_de_asfl_no_cluster.yaml",
            "label": "消融: 去聚类 (Best Client 扁平聚合)",
            "category": "ablation",
            "variant": "w/o Clustering",
        },
        {
            "config": "config_de_asfl_no_election_no_cluster.yaml",
            "label": "消融: 去选举+去聚类 (client 0 扁平聚合)",
            "category": "ablation",
            "variant": "w/o Election+Cluster",
        },
    ],

    # ---- JSD 阈值扫描 ----
    "jsd": [
        {
            "config": f"config_de_asfl_jsd_{v}.yaml",
            "label": f"JSD 阈值 = {v}",
            "category": "jsd_threshold",
            "jsd": v,
        }
        for v in [0.1, 0.2, 0.3, 0.5]
    ],

    # ---- 拓扑对比 ----
    "topology": [
        {
            "config": "config_de_asfl_ring.yaml",
            "label": "拓扑: Ring",
            "category": "topology",
            "topology": "ring",
        },
        {
            "config": "config_de_asfl_star.yaml",
            "label": "拓扑: Star",
            "category": "topology",
            "topology": "star",
        },
    ],
}


def run_one(config_path: str, label: str, dry_run: bool = False) -> dict:
    """运行单个实验，返回结果字典"""
    if dry_run:
        print(f"  [DRY RUN] {label}")
        return {"status": "skipped", "label": label}

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

        lines = result.stdout.strip().split('\n')
        for line in lines[-30:]:
            if line.strip():
                print(f"  | {line.strip()[:140]}")

        if result.returncode != 0:
            print(f"  ❌ 失败 (exit {result.returncode})")
            for line in result.stderr.strip().split('\n')[-8:]:
                print(f"  [stderr] {line.strip()[:140]}")
            return {"status": "failed", "label": label, "elapsed": elapsed}

        # 提取输出目录
        output_dir = None
        for line in lines:
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

        # 读取 summary
        summary = None
        if output_dir:
            sp = os.path.join(output_dir, "summary.json")
            if os.path.exists(sp):
                with open(sp) as f:
                    summary = json.load(f)

        acc = summary.get('final_val_acc', 0) if summary else 0
        print(f"  ✅ 完成 ({elapsed:.0f}s)  准确率: {acc:.2%}  → {output_dir}")

        return {
            "status": "success", "label": label, "elapsed": elapsed,
            "output_dir": output_dir, "summary": summary,
        }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"  ⏰ 超时")
        return {"status": "timeout", "label": label, "elapsed": elapsed}
    except Exception as e:
        elapsed = time.time() - start
        print(f"  ❌ 异常: {e}")
        return {"status": "error", "label": label, "elapsed": elapsed, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="消融/JSD/拓扑批量实验")
    parser.add_argument("--list", action="store_true", help="仅列出计划")
    parser.add_argument("--dry-run", action="store_true", help="仅生成配置，不运行")
    parser.add_argument("--only", choices=["ablation", "jsd", "topology"],
                        help="仅运行指定类别")
    parser.add_argument("--start-from", type=int, default=0, help="从第N个开始")
    args = parser.parse_args()

    # 选择实验类别
    if args.only:
        categories = [args.only]
    else:
        categories = ["ablation", "jsd", "topology"]

    all_exps = []
    for cat in categories:
        for exp in EXPERIMENTS[cat]:
            exp_copy = dict(exp)
            all_exps.append(exp_copy)

    # 打印计划
    print("=" * 70)
    print(f"实验计划（共 {len(all_exps)} 组）")
    print("=" * 70)
    total_est = 0
    for i, exp in enumerate(all_exps):
        est = 20  # DE-ASFL ~20min each
        total_est += est
        print(f"  [{i+1:2d}] {exp['label']}  (~{est}min)")

    print(f"\n预计总耗时: ~{total_est/60:.1f} 小时")
    if args.only:
        print(f"类别: {args.only}")
    print()

    if args.list:
        return

    # 确认
    if not args.dry_run:
        choice = input("是否开始运行？[y/N] ").strip().lower()
        if choice not in ('y', 'yes'):
            print("已取消")
            return

    # 运行
    results = []
    start_from = max(0, args.start_from)

    for i, exp in enumerate(all_exps):
        if i < start_from:
            print(f"\n跳过 [{i+1}/{len(all_exps)}] {exp['label']}")
            continue

        print(f"\n{'#'*70}")
        print(f"实验 [{i+1}/{len(all_exps)}]")
        print(f"{'#'*70}")

        result = run_one(exp["config"], exp["label"], dry_run=args.dry_run)
        result["category"] = exp.get("category", "")
        result["variant"] = exp.get("variant", "")
        result["jsd"] = exp.get("jsd")
        result["topology"] = exp.get("topology")
        results.append(result)

        # 打印中间汇总
        acc = (result.get('summary') or {}).get('final_val_acc', 0)
        if acc:
            print(f"  📊 准确率: {acc:.2%}")
        print(f"  进度: {len(results)}/{len(all_exps)}")

    # ---- 生成汇总 ----
    if args.dry_run:
        return

    print("\n" + "=" * 70)
    print("实验结果汇总")
    print("=" * 70)

    # 消融实验
    ablation_results = [r for r in results if r.get('category') == 'ablation']
    if ablation_results:
        print("\n消融实验:")
        print(f"  {'变体':30s} {'准确率':>10s}")
        print(f"  {'-'*30} {'-'*10}")
        # 加入基线结果
        print(f"  {'完整 DE-ASFL (已有)':30s} {'80.75%':>10s}")
        for r in ablation_results:
            acc = (r.get('summary') or {}).get('final_val_acc', 0)
            print(f"  {r.get('variant', '?'):30s} {acc:>9.2%}")

    # JSD 阈值
    jsd_results = [r for r in results if r.get('category') == 'jsd_threshold']
    if jsd_results:
        print(f"\nJSD 阈值扫描 (基线 JSD=0: 80.75%):")
        print(f"  {'阈值':>8s} {'准确率':>10s}")
        print(f"  {'-'*8} {'-'*10}")
        for r in sorted(jsd_results, key=lambda x: x.get('jsd', 0)):
            acc = (r.get('summary') or {}).get('final_val_acc', 0)
            print(f"  {r.get('jsd', 0):>8.1f} {acc:>9.2%}")

    # 拓扑
    topo_results = [r for r in results if r.get('category') == 'topology']
    if topo_results:
        print(f"\n拓扑对比 (基线 Mesh: 80.75%):")
        print(f"  {'拓扑':>8s} {'准确率':>10s}")
        print(f"  {'-'*8} {'-'*10}")
        for r in topo_results:
            acc = (r.get('summary') or {}).get('final_val_acc', 0)
            print(f"  {r.get('topology', '?'):>8s} {acc:>9.2%}")

    success = sum(1 for r in results if r['status'] == 'success')
    total_time = sum(r.get('elapsed', 0) for r in results)
    print(f"\n总计: {success}/{len(results)} 成功, 耗时 {total_time/3600:.1f}h")


if __name__ == "__main__":
    main()
