#!/usr/bin/env python3
"""
批量实验运行脚本 — 一键运行所有可行的 DE-ASFL / FedAvg / Gossip 对比实验

用法:
    python run_experiments.py              # 默认：核心数据异构实验（12组，~3小时）
    python run_experiments.py --quick      # 快速模式（5组，~1.5小时）
    python run_experiments.py --full       # 完整模式（所有实验，~6小时）
    python run_experiments.py --list       # 仅列出实验计划，不运行
    python run_experiments.py --skip de_asfl  # 跳过 DE-ASFL，只跑 FedAvg/Gossip

输出:
    outputs/                              # 每个实验的时间戳目录
    EXPERIMENT_RESULTS.md                 # 自动生成的结果汇总文档
"""

import os
import sys
import subprocess
import json
import time
import argparse
import copy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    print("请安装 PyYAML: pip install pyyaml")
    sys.exit(1)


# ============================================================
# 实验矩阵定义
# ============================================================

BASE_DIR = Path(__file__).parent.absolute()

# 数据异构性 — 核心实验
DATA_HETEROGENEITY_EXPS = [
    # (算法, 配置文件模板, alpha, 拓扑, 标签)
    ("de_asfl", "config_de_asfl.yaml", 0.05, "mesh", "alpha=0.05 极端异构"),
    ("de_asfl", "config_de_asfl.yaml", 0.1,  "mesh", "alpha=0.1 强异构"),
    ("de_asfl", "config_de_asfl.yaml", 0.5,  "mesh", "alpha=0.5 中等异构"),
    ("de_asfl", "config_de_asfl.yaml", 1.0,  "mesh", "alpha=1.0 轻度异构"),
    ("fedavg",  "config_fedavg.yaml",  0.05, "star", "alpha=0.05 极端异构"),
    ("fedavg",  "config_fedavg.yaml",  0.1,  "star", "alpha=0.1 强异构"),
    ("fedavg",  "config_fedavg.yaml",  0.5,  "star", "alpha=0.5 中等异构"),
    ("fedavg",  "config_fedavg.yaml",  1.0,  "star", "alpha=1.0 轻度异构"),
    ("gossip",  "config_gossip.yaml",  0.05, "mesh", "alpha=0.05 极端异构"),
    ("gossip",  "config_gossip.yaml",  0.1,  "mesh", "alpha=0.1 强异构"),
    ("gossip",  "config_gossip.yaml",  0.5,  "mesh", "alpha=0.5 中等异构"),
    ("gossip",  "config_gossip.yaml",  1.0,  "mesh", "alpha=1.0 轻度异构"),
]

# 拓扑实验
TOPOLOGY_EXPS = [
    ("de_asfl", "config_de_asfl.yaml", 0.1, "mesh", "Mesh 拓扑"),
    ("de_asfl", "config_de_asfl.yaml", 0.1, "ring", "Ring 拓扑"),
    ("de_asfl", "config_de_asfl.yaml", 0.1, "star", "Star 拓扑"),
]

# JSD 阈值扫描
JSD_THRESHOLD_EXPS = [
    ("de_asfl", "config_de_asfl.yaml", 0.1, "mesh", "JSD=0 (禁用分布检查)"),
    ("de_asfl", "config_de_asfl.yaml", 0.1, "mesh", "JSD=0.1"),
    ("de_asfl", "config_de_asfl.yaml", 0.1, "mesh", "JSD=0.2"),
    ("de_asfl", "config_de_asfl.yaml", 0.1, "mesh", "JSD=0.3"),
    ("de_asfl", "config_de_asfl.yaml", 0.1, "mesh", "JSD=0.5"),
]
JSD_VALUES = [0, 0.1, 0.2, 0.3, 0.5]

# 簇大小扫描
CLUSTER_SIZE_EXPS = [
    ("de_asfl", "config_de_asfl.yaml", 0.1, "mesh", "max_child=2"),
    ("de_asfl", "config_de_asfl.yaml", 0.1, "mesh", "max_child=3"),
    ("de_asfl", "config_de_asfl.yaml", 0.1, "mesh", "max_child=5"),
    ("de_asfl", "config_de_asfl.yaml", 0.1, "mesh", "max_child=10"),
]
CLUSTER_SIZES = [2, 3, 5, 10]


def load_yaml(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(config: dict, path: str):
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def build_de_asfl_config(base_config: dict, **overrides) -> dict:
    """构建 DE-ASFL 配置"""
    cfg = copy.deepcopy(base_config)
    cfg['alpha'] = overrides.get('alpha', cfg.get('alpha', 0.1))
    cfg['topology_type'] = overrides.get('topology_type', cfg.get('topology_type', 'mesh'))
    if 'distribution_similarity_threshold' in overrides:
        cfg['distribution_similarity_threshold'] = overrides['distribution_similarity_threshold']
    if 'cluster_max_child' in overrides:
        cfg['cluster_max_child'] = overrides['cluster_max_child']
    return cfg


def build_fedavg_config(base_config: dict, **overrides) -> dict:
    """构建 FedAvg 配置"""
    cfg = copy.deepcopy(base_config)
    cfg['dirichlet_alpha'] = overrides.get('alpha', cfg.get('dirichlet_alpha', 0.1))
    return cfg


def build_gossip_config(base_config: dict, **overrides) -> dict:
    """构建 Gossip 配置"""
    cfg = copy.deepcopy(base_config)
    cfg['dirichlet_alpha'] = overrides.get('alpha', cfg.get('dirichlet_alpha', 0.1))
    return cfg


def get_exp_name(algorithm: str, alpha: float, topology: str, extra: str = "") -> str:
    """生成实验名称"""
    name = f"{algorithm}_a{alpha}_{topology}"
    if extra:
        name += f"_{extra}"
    return name


def run_single_experiment(
    algorithm: str,
    temp_config_path: str,
    exp_name: str,
    dry_run: bool = False
) -> Tuple[bool, Optional[str], float]:
    """
    运行单个实验，返回 (成功, 输出目录, 耗时秒)
    """
    if dry_run:
        print(f"  [DRY RUN] {exp_name}")
        return True, None, 0

    start = time.time()

    if algorithm == "de_asfl":
        cmd = [
            sys.executable, str(BASE_DIR / "test_30_clients.py"),
            "--config", temp_config_path
        ]
    else:
        cmd = [
            sys.executable, str(BASE_DIR / "run_fl.py"),
            "--config", temp_config_path
        ]

    print(f"\n{'='*70}")
    print(f"运行: {exp_name}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*70}")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=7200,  # 2 小时超时
            env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )

        elapsed = time.time() - start

        # 输出最后 30 行以便监控
        lines = result.stdout.strip().split('\n')
        for line in lines[-40:]:
            if line.strip():
                print(f"  | {line.strip()[:140]}")

        if result.returncode != 0:
            print(f"  ❌ 失败 (exit code {result.returncode})")
            stderr_tail = result.stderr.strip().split('\n')[-10:]
            for line in stderr_tail:
                print(f"  [stderr] {line.strip()[:140]}")
            return False, None, elapsed

        # 从 stdout 提取输出目录
        output_dir = None
        for line in lines:
            # DE-ASFL 中文 "输出目录:" / FedAvg/Gossip 英文 "Output directory:"
            if "输出目录:" in line:
                output_dir = line.split("输出目录:")[-1].strip()
                break
            elif "Output directory:" in line:
                output_dir = line.split("Output directory:")[-1].strip()
                break

        # 没找到就找最新创建的目录
        if output_dir is None:
            outputs_dir = BASE_DIR / "outputs"
            prefix = "de_asfl_" if algorithm == "de_asfl" else (
                "fedavg_" if algorithm == "fedavg" else "gossip_")
            candidates = sorted(
                [d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)],
                key=lambda p: p.stat().st_mtime, reverse=True
            )
            if candidates:
                output_dir = str(candidates[0])

        if output_dir:
            print(f"  ✅ 完成 ({elapsed:.0f}s) → {output_dir}")
        else:
            print(f"  ✅ 完成 ({elapsed:.0f}s)")

        return True, output_dir, elapsed

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"  ⏰ 超时 (>2h)")
        return False, None, elapsed
    except Exception as e:
        elapsed = time.time() - start
        print(f"  ❌ 异常: {e}")
        return False, None, elapsed


def read_summary(output_dir: str) -> Optional[dict]:
    """读取实验的 summary.json"""
    if not output_dir:
        return None
    summary_path = os.path.join(output_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            return json.load(f)
    return None


def generate_results_markdown(results: List[dict], output_path: str):
    """生成实验结果汇总 Markdown 文档"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("# DE-ASFL 实验结果汇总")
    lines.append(f"\n生成时间: {now}\n")

    # 分类汇总
    # 1. 数据异构性
    lines.append("## 一、数据异构性对比\n")
    lines.append("| 算法 | Alpha | 最终准确率 | 最高准确率 | 最终 Loss | 耗时 | 输出目录 |")
    lines.append("|------|-------|-----------|-----------|----------|------|---------|")

    het_results = [r for r in results if r.get('category') == 'data_heterogeneity']
    for r in sorted(het_results, key=lambda x: (x['algorithm'], x.get('alpha', 0))):
        algo = r['algorithm'].upper()
        alpha = r.get('alpha', '?')
        s = r.get('summary', {})
        final_acc = f"{s.get('final_val_acc', 0):.2%}" if s else r.get('status', 'N/A')
        max_acc = f"{s.get('max_val_acc', 0):.2%}" if s else r.get('status', 'N/A')
        final_loss = f"{s.get('final_val_loss', 0):.4f}" if s else 'N/A'
        elapsed = f"{r.get('elapsed', 0)/60:.0f}min" if r.get('elapsed') else 'N/A'
        out = r.get('output_dir', 'N/A')
        lines.append(f"| {algo} | {alpha} | {final_acc} | {max_acc} | {final_loss} | {elapsed} | {out} |")

    lines.append("")

    # 2. 拓扑结构
    topo_results = [r for r in results if r.get('category') == 'topology']
    if topo_results:
        lines.append("## 二、拓扑结构对比\n")
        lines.append("| 拓扑 | 最终准确率 | 最高准确率 | 最终 Loss | 耗时 |")
        lines.append("|------|-----------|-----------|----------|------|")
        for r in sorted(topo_results, key=lambda x: x.get('topology', '')):
            topo = r.get('topology', '?')
            s = r.get('summary', {})
            final_acc = f"{s.get('final_val_acc', 0):.2%}" if s else r.get('status', 'N/A')
            max_acc = f"{s.get('max_val_acc', 0):.2%}" if s else r.get('status', 'N/A')
            final_loss = f"{s.get('final_val_loss', 0):.4f}" if s else 'N/A'
            elapsed = f"{r.get('elapsed', 0)/60:.0f}min" if r.get('elapsed') else 'N/A'
            lines.append(f"| {topo} | {final_acc} | {max_acc} | {final_loss} | {elapsed} |")
        lines.append("")

    # 3. JSD 阈值
    jsd_results = [r for r in results if r.get('category') == 'jsd_threshold']
    if jsd_results:
        lines.append("## 三、JSD 分布相似度阈值扫描\n")
        lines.append("| JSD 阈值 | 最终准确率 | 最高准确率 | 最终 Loss | 耗时 |")
        lines.append("|---------|-----------|-----------|----------|------|")
        for r in sorted(jsd_results, key=lambda x: x.get('jsd', 0)):
            jsd = r.get('jsd', '?')
            s = r.get('summary', {})
            final_acc = f"{s.get('final_val_acc', 0):.2%}" if s else r.get('status', 'N/A')
            max_acc = f"{s.get('max_val_acc', 0):.2%}" if s else r.get('status', 'N/A')
            final_loss = f"{s.get('final_val_loss', 0):.4f}" if s else 'N/A'
            elapsed = f"{r.get('elapsed', 0)/60:.0f}min" if r.get('elapsed') else 'N/A'
            lines.append(f"| {jsd} | {final_acc} | {max_acc} | {final_loss} | {elapsed} |")
        lines.append("")

    # 4. 簇大小
    cluster_results = [r for r in results if r.get('category') == 'cluster_size']
    if cluster_results:
        lines.append("## 四、簇大小限制扫描\n")
        lines.append("| max_child | 最终准确率 | 最高准确率 | 最终 Loss | 耗时 |")
        lines.append("|-----------|-----------|-----------|----------|------|")
        for r in sorted(cluster_results, key=lambda x: x.get('max_child', 0)):
            mc = r.get('max_child', '?')
            s = r.get('summary', {})
            final_acc = f"{s.get('final_val_acc', 0):.2%}" if s else r.get('status', 'N/A')
            max_acc = f"{s.get('max_val_acc', 0):.2%}" if s else r.get('status', 'N/A')
            final_loss = f"{s.get('final_val_loss', 0):.4f}" if s else 'N/A'
            elapsed = f"{r.get('elapsed', 0)/60:.0f}min" if r.get('elapsed') else 'N/A'
            lines.append(f"| {mc} | {final_acc} | {max_acc} | {final_loss} | {elapsed} |")
        lines.append("")

    # 汇总
    total = len(results)
    success = sum(1 for r in results if r.get('status') == 'success')
    failed = total - success
    total_time = sum(r.get('elapsed', 0) for r in results)
    lines.append("## 汇总统计\n")
    lines.append(f"- 总实验数: {total}")
    lines.append(f"- 成功: {success}")
    lines.append(f"- 失败: {failed}")
    lines.append(f"- 总耗时: {total_time/3600:.1f} 小时")
    lines.append("")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\n结果文档已生成: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="批量联邦学习实验")
    parser.add_argument("--quick", action="store_true", help="快速模式（仅数据异构核心 3 组）")
    parser.add_argument("--full", action="store_true", help="完整模式（所有实验）")
    parser.add_argument("--list", action="store_true", help="仅列出计划，不运行")
    parser.add_argument("--dry-run", action="store_true", help="不实际运行，仅生成配置文件")
    parser.add_argument("--skip", nargs="+", choices=["de_asfl", "fedavg", "gossip"],
                        help="跳过指定算法")
    parser.add_argument("--only", nargs="+", choices=["de_asfl", "fedavg", "gossip"],
                        help="仅运行指定算法")
    parser.add_argument("--start-from", type=int, default=0, help="从第 N 个实验开始（断点续跑）")
    args = parser.parse_args()

    skip_algorithms = set(args.skip or [])
    only_algorithms = set(args.only or [])

    # ---- 构建实验列表 ----
    experiments: List[dict] = []

    if args.quick:
        # 快速模式：仅核心对比
        quick_exps = [
            ("de_asfl", "config_de_asfl.yaml", 0.1, "mesh", "alpha=0.1 强异构", "data_heterogeneity"),
            ("fedavg",  "config_fedavg.yaml",  0.1, "star", "alpha=0.1 强异构", "data_heterogeneity"),
            ("gossip",  "config_gossip.yaml",  0.1, "mesh", "alpha=0.1 强异构", "data_heterogeneity"),
            ("de_asfl", "config_de_asfl.yaml", 0.5, "mesh", "alpha=0.5 中等异构", "data_heterogeneity"),
            ("de_asfl", "config_de_asfl.yaml", 1.0, "mesh", "alpha=1.0 轻度异构", "data_heterogeneity"),
        ]
        for alg, cfg_path, alpha, topo, label, cat in quick_exps:
            experiments.append({
                "algorithm": alg, "config_template": cfg_path, "alpha": alpha,
                "topology": topo, "label": label, "category": cat
            })

    elif args.full:
        # 完整模式
        for alg, cfg_path, alpha, topo, label in DATA_HETEROGENEITY_EXPS:
            experiments.append({
                "algorithm": alg, "config_template": cfg_path, "alpha": alpha,
                "topology": topo, "label": label, "category": "data_heterogeneity"
            })
        for alg, cfg_path, alpha, topo, label in TOPOLOGY_EXPS:
            experiments.append({
                "algorithm": alg, "config_template": cfg_path, "alpha": alpha,
                "topology": topo, "label": label, "category": "topology"
            })
        for (alg, cfg_path, alpha, topo, label), jsd in zip(JSD_THRESHOLD_EXPS, JSD_VALUES):
            experiments.append({
                "algorithm": alg, "config_template": cfg_path, "alpha": alpha,
                "topology": topo, "label": label, "category": "jsd_threshold", "jsd": jsd
            })
        for (alg, cfg_path, alpha, topo, label), mc in zip(CLUSTER_SIZE_EXPS, CLUSTER_SIZES):
            experiments.append({
                "algorithm": alg, "config_template": cfg_path, "alpha": alpha,
                "topology": topo, "label": label, "category": "cluster_size", "max_child": mc
            })

    else:
        # 默认：数据异构性全部
        for alg, cfg_path, alpha, topo, label in DATA_HETEROGENEITY_EXPS:
            experiments.append({
                "algorithm": alg, "config_template": cfg_path, "alpha": alpha,
                "topology": topo, "label": label, "category": "data_heterogeneity"
            })

    # 过滤
    if skip_algorithms:
        experiments = [e for e in experiments if e['algorithm'] not in skip_algorithms]
    if only_algorithms:
        experiments = [e for e in experiments if e['algorithm'] in only_algorithms]

    # 预加载基础配置
    config_cache = {}

    # ---- 列出实验计划 ----
    print("=" * 70)
    print(f"实验计划（共 {len(experiments)} 组）")
    print("=" * 70)
    total_est_minutes = 0
    for i, exp in enumerate(experiments):
        est = 20 if exp['algorithm'] == 'de_asfl' else 5  # 估计分钟数
        total_est_minutes += est
        print(f"  [{i+1:2d}/{len(experiments)}] {exp['algorithm']:8s} | {exp['label']} (~{est}min)")

    print(f"\n预计总耗时: ~{total_est_minutes/60:.1f} 小时")
    print(f"模式: {'快速' if args.quick else '完整' if args.full else '默认（数据异构）'}")
    print()

    if args.list:
        return

    # ---- 确认 ----
    print("-" * 70)
    choice = input("是否开始运行？[y/N] ").strip().lower()
    if choice not in ('y', 'yes'):
        print("已取消")
        return

    # ---- 运行实验 ----
    results = []
    start_from = max(0, args.start_from)

    for i, exp in enumerate(experiments):
        if i < start_from:
            print(f"\n跳过 [{i+1}/{len(experiments)}] {exp['label']}")
            continue

        print(f"\n{'#'*70}")
        print(f"实验 [{i+1}/{len(experiments)}] {exp['label']}")
        print(f"{'#'*70}")

        alg = exp['algorithm']
        cfg_template = exp['config_template']
        alpha = exp.get('alpha', 0.1)
        topology = exp.get('topology', 'mesh')

        # 加载/缓存配置
        if cfg_template not in config_cache:
            config_cache[cfg_template] = load_yaml(str(BASE_DIR / cfg_template))

        base_cfg = copy.deepcopy(config_cache[cfg_template])

        # 构建实验配置
        if alg == "de_asfl":
            cfg = build_de_asfl_config(base_cfg, alpha=alpha, topology_type=topology)
            if 'jsd' in exp:
                cfg['distribution_similarity_threshold'] = exp['jsd']
            if 'max_child' in exp:
                cfg['cluster_max_child'] = exp['max_child']
        elif alg == "fedavg":
            cfg = build_fedavg_config(base_cfg, alpha=alpha)
        elif alg == "gossip":
            cfg = build_gossip_config(base_cfg, alpha=alpha)
        else:
            print(f"  未知算法: {alg}")
            continue

        # 保存临时配置
        temp_dir = BASE_DIR / ".temp_configs"
        temp_dir.mkdir(exist_ok=True)
        ext = "yaml"
        temp_config_path = temp_dir / f"exp_{i:03d}_{alg}_a{alpha}_{topology}.{ext}"
        save_yaml(cfg, str(temp_config_path))

        # 运行
        success, output_dir, elapsed = run_single_experiment(
            alg, str(temp_config_path),
            f"{exp['label']}",
            dry_run=args.dry_run
        )

        # 收集结果
        result_entry = {
            "algorithm": alg,
            "alpha": alpha,
            "topology": topology,
            "label": exp['label'],
            "category": exp['category'],
            "status": "success" if success else "failed",
            "output_dir": output_dir,
            "elapsed": elapsed,
            "summary": read_summary(output_dir) if output_dir else None,
        }
        if 'jsd' in exp:
            result_entry['jsd'] = exp['jsd']
        if 'max_child' in exp:
            result_entry['max_child'] = exp['max_child']
        results.append(result_entry)

        # 清理临时配置
        if temp_config_path.exists():
            temp_config_path.unlink()

        # 打印中间汇总
        acc = result_entry.get('summary', {})
        if acc:
            print(f"  📊 准确率: {acc.get('final_val_acc', 0):.2%} (max: {acc.get('max_val_acc', 0):.2%})")
        print(f"  进度: {len(results)}/{len(experiments)}")

    # ---- 清理 ----
    temp_dir = BASE_DIR / ".temp_configs"
    if temp_dir.exists():
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    # ---- 生成结果文档 ----
    print("\n" + "=" * 70)
    print("实验全部完成，生成结果文档...")
    print("=" * 70)

    output_path = BASE_DIR / "EXPERIMENT_RESULTS.md"
    generate_results_markdown(results, str(output_path))

    # ---- 最终汇总 ----
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"\n总计: {success_count}/{len(results)} 成功")
    total_elapsed = sum(r.get('elapsed', 0) for r in results)
    print(f"实际总耗时: {total_elapsed/3600:.1f} 小时")


if __name__ == "__main__":
    main()
