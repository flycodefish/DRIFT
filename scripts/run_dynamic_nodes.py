"""
Dynamic Node Failure / Join 实验脚本

支持多种故障模式（随机/叶节点/簇首/Best Client）和恢复策略的对比实验。

用法:
    # 基本故障实验（5% 故障率）
    python scripts/run_dynamic_nodes.py --config configs/cifar10/config_de_asfl_dynamic_failure.yaml

    # 高故障率实验（10%）
    python scripts/run_dynamic_nodes.py --config configs/cifar10/config_de_asfl_dynamic_failure_high.yaml

    # 批量对比实验（不同故障率）
    python scripts/run_dynamic_nodes.py --batch --base-config configs/cifar10/config_de_asfl.yaml \
        --failure-rates 0.0,0.03,0.05,0.10 --patterns random,leaf_only,cluster_head
"""

import argparse
import json
import os
import sys
import time

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pandas as pd
from datetime import datetime

import glob

from deasfl import NewFLConfig, NewFLRuntime
from deasfl.runtime import (
    load_cifar10_data, load_cifar100_data, load_neu_data, load_har_data,
    resolve_dataset_split_policy,
)


def _is_experiment_complete(experiment_name: str, min_rounds: int = 90) -> bool:
    """检查指定实验是否已完成（存在输出目录且round_metrics有足够行数）"""
    pattern = f"outputs/*_{experiment_name}_*/"
    dirs = glob.glob(pattern)
    for d in dirs:
        csv_files = glob.glob(os.path.join(d, "*_round_metrics.csv"))
        for csv_f in csv_files:
            try:
                with open(csv_f) as f:
                    line_count = sum(1 for _ in f)
                if line_count >= min_rounds + 1:  # header + rounds
                    return True
            except Exception:
                pass
    return False


def run_single_experiment(config_path: str, override_params: dict = None) -> dict:
    """运行单个实验并返回结果摘要

    Args:
        config_path: YAML 配置文件路径
        override_params: 覆盖的参数字典（用于批量实验）

    Returns:
        实验结果摘要字典
    """
    cfg = NewFLConfig.from_yaml(config_path)

    # 应用参数覆盖
    if override_params:
        for key, value in override_params.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)

    # 验证动态节点配置
    if not cfg.enable_dynamic_nodes:
        print("⚠️  enable_dynamic_nodes=False，此实验不会产生动态节点行为")
        print("   请使用包含 enable_dynamic_nodes: true 的配置文件")

    print(f"\n{'='*80}")
    print(f"Dynamic Node Experiment: {cfg.experiment_name}")
    print(f"{'='*80}")
    print(f"  Failure rate: {cfg.failure_rate}, Start round: {cfg.failure_start_round}")
    print(f"  Max failures: {cfg.failure_max_nodes}, Pattern: {cfg.failure_pattern}")
    print(f"  Recovery strategy: {cfg.recovery_strategy}")
    print(f"  Join enabled: {cfg.node_join_enabled}")

    # 初始化并运行
    runtime = NewFLRuntime(cfg)
    runtime.start()

    try:
        # 加载数据
        if cfg.dataset == 'neu':
            train_data, val_data, indices, global_test_data, label_dists = load_neu_data(
                data_dir=cfg.data_dir, num_clients=cfg.num_clients,
                partition_mode=cfg.partition, alpha=cfg.alpha)
        elif cfg.dataset == 'har':
            train_data, val_data, indices, global_test_data, label_dists = load_har_data(
                data_dir=cfg.data_dir, num_clients=cfg.num_clients,
                partition_mode=cfg.partition, alpha=cfg.alpha,
                seed=cfg.seed,
                client_indices=(runtime.frozen_inputs["data_partition.json"]["client_indices"]
                                if runtime.frozen_inputs else None),
                split_policy=resolve_dataset_split_policy(cfg.dataset, cfg.dataset_split_policy),
                local_val_seed=cfg.har_local_val_seed)
        elif cfg.dataset == 'cifar100':
            train_data, val_data, indices, global_test_data, label_dists = load_cifar100_data(
                data_dir=cfg.data_dir, num_clients=cfg.num_clients,
                partition_mode=cfg.partition, alpha=cfg.alpha)
        else:
            train_data, val_data, indices, global_test_data, label_dists = load_cifar10_data(
                data_dir=cfg.data_dir, num_clients=cfg.num_clients,
                partition_mode=cfg.partition, alpha=cfg.alpha, seed=cfg.seed,
                client_indices=(runtime.frozen_inputs["data_partition.json"]["client_indices"]
                                if runtime.frozen_inputs else None))

        runtime.label_distributions = label_dists

        # 初始化
        runtime.init_clients(train_data, val_data, indices)
        runtime.build_topology()

        # 选举
        if cfg.enable_election:
            runtime.init_election(cfg.election_base_trigger_time, cfg.election_trigger_factor)
            election_result = runtime.run_election()
        else:
            election_result = {'best_client_id': 0, 'best_client_score': 0.0,
                              'convergence_hops': 0, 'all_nodes_info': []}

        # 聚类
        if cfg.enable_clustering:
            runtime.init_clustering(election_result, cfg.cluster_max_child, cfg.cluster_offset)
            clustering_result = runtime.run_clustering()
            runtime.set_clustering_results(clustering_result, election_result)
        else:
            runtime.clustering_results = None
            runtime.tree_mapping = None

        # Ray Actor + 训练
        runtime._init_ray_actors(train_data, val_data, global_test_data)
        training_results = runtime.run_training(num_rounds=cfg.rounds)

    finally:
        runtime.shutdown()

    # ---- 构建结果摘要 ----
    final_global_acc = (float(training_results['global_accuracies'][-1])
                        if training_results['global_accuracies'] else 0.0)
    failure_rounds = [f['round'] for f in training_results['failure_log']]

    summary = {
        'experiment_name': cfg.experiment_name,
        'failure_rate': cfg.failure_rate,
        'failure_pattern': cfg.failure_pattern,
        'failure_start_round': cfg.failure_start_round,
        'max_failures': cfg.failure_max_nodes,
        'recovery_strategy': cfg.recovery_strategy,
        'actual_failures': len(runtime.failed_nodes),
        'failed_nodes': sorted(runtime.failed_nodes),
        'failure_rounds': failure_rounds,
        'best_client_id': election_result.get('best_client_id', -1),
        'best_client_failed': election_result.get('best_client_id', -1) in runtime.failed_nodes,
        'final_avg_acc': final_global_acc,
        'final_max_acc': final_global_acc,
        'final_global_acc': final_global_acc,
        'training_time': training_results['training_time'],
        'num_rounds': cfg.rounds,
        'num_clients': cfg.num_clients,
        'dataset': cfg.dataset,
        'model': cfg.model,
    }

    return summary


def run_batch_experiments(base_config: str, failure_rates: list, patterns: list,
                          failure_start_round: int = 20, max_failures: int = 10,
                          recovery: str = "skip", resume: bool = False,
                          data_dir: str | None = None):
    """批量运行不同参数组合的实验

    Args:
        base_config: 基础配置文件路径
        failure_rates: 故障率列表，如 [0.0, 0.03, 0.05, 0.10]
        patterns: 故障模式列表，如 ['random', 'leaf_only', 'cluster_head']
        failure_start_round: 统一故障起始轮
        max_failures: 统一最大故障节点数
        recovery: 恢复策略
        resume: 是否跳过已完成的实验
    """
    all_summaries = []
    total_experiments = len(failure_rates) * len(patterns)

    # 始终包含一组无故障基线
    baseline_name = "baseline_no_failure"
    baseline_override = {
        'enable_dynamic_nodes': False,
        'failure_rate': 0.0,
        'recovery_strategy': 'skip',
        'experiment_name': baseline_name,
    }
    if data_dir:
        baseline_override['data_dir'] = data_dir
    exp_idx = 0

    if resume and _is_experiment_complete(baseline_name):
        print(f"\n⏭️  基线实验已完成，跳过")
    else:
        print(f"\n{'#'*80}")
        print(f"实验 {exp_idx}/{total_experiments + 1}: 基线（无故障）")
        print(f"{'#'*80}")
        baseline = run_single_experiment(base_config, baseline_override)
        baseline['experiment_name'] = baseline_name
        all_summaries.append(baseline)

    for rate in failure_rates:
        for pattern in patterns:
            exp_idx += 1
            exp_name = f"dyn_{pattern}_r{rate}"

            if resume and _is_experiment_complete(exp_name):
                print(f"\n⏭️  实验 {exp_idx}/{total_experiments + 1}: {exp_name} 已完成，跳过")
                continue

            print(f"\n{'#'*80}")
            print(f"实验 {exp_idx}/{total_experiments + 1}: rate={rate}, pattern={pattern}")
            print(f"{'#'*80}")

            override = {
                'enable_dynamic_nodes': True,
                'failure_rate': rate,
                'failure_pattern': pattern,
                'failure_start_round': failure_start_round,
                'failure_max_nodes': max_failures,
                'recovery_strategy': recovery,
                'node_join_enabled': False,
                'experiment_name': exp_name,
            }
            if data_dir:
                override['data_dir'] = data_dir
            summary = run_single_experiment(base_config, override)
            all_summaries.append(summary)

    if not all_summaries:
        print("\n所有实验均已完成，无需运行！")
        return pd.DataFrame()

    # 汇总结果
    print(f"\n{'='*80}")
    print(f"Batch Experiment Summary")
    print(f"{'='*80}")

    df = pd.DataFrame(all_summaries)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = f"outputs/dynamic_nodes_summary_{timestamp}.csv"
    os.makedirs("outputs", exist_ok=True)
    df.to_csv(summary_path, index=False)

    print(f"\n结果汇总:")
    print(df[['experiment_name', 'failure_rate', 'failure_pattern',
              'actual_failures', 'final_avg_acc', 'final_global_acc']].to_string(index=False))
    print(f"\n详细结果已保存到: {summary_path}")
    print(f"JSON 格式:")
    print(json.dumps(all_summaries, indent=2, ensure_ascii=False,
                     default=lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else str(x)))

    # 保存 JSON
    json_path = f"outputs/dynamic_nodes_summary_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False,
                  default=lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else str(x))
    print(f"JSON 详细结果已保存到: {json_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description='Dynamic Node Failure/Join Experiments')
    parser.add_argument('--config', type=str, default=None,
                       help='YAML 配置文件路径')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='数据集根目录，覆盖配置文件中的 data_dir')
    parser.add_argument('--batch', action='store_true',
                       help='批量运行多个参数组合')
    parser.add_argument('--base-config', type=str,
                       default='configs/cifar10/config_de_asfl_dynamic_failure.yaml',
                       help='批量模式的基础配置文件')
    parser.add_argument('--failure-rates', type=str, default='0.0,0.03,0.05,0.10',
                       help='批量模式: 逗号分隔的故障率列表')
    parser.add_argument('--patterns', type=str, default='random,leaf_only',
                       help='批量模式: 逗号分隔的故障模式')
    parser.add_argument('--failure-start-round', type=int, default=20)
    parser.add_argument('--max-failures', type=int, default=10)
    parser.add_argument('--recovery', type=str, default='skip',
                       choices=['skip', 'reassign', 'reelect_recluster'])
    parser.add_argument('--resume', action='store_true',
                       help='批量模式: 跳过已完成的实验（检测输出目录中的完整结果）')
    args = parser.parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"

    if args.batch:
        rates = [float(r) for r in args.failure_rates.split(',')]
        patterns = [p.strip() for p in args.patterns.split(',')]
        run_batch_experiments(
            base_config=args.base_config,
            failure_rates=rates,
            patterns=patterns,
            failure_start_round=args.failure_start_round,
            max_failures=args.max_failures,
            recovery=args.recovery,
            resume=args.resume,
            data_dir=args.data_dir,
        )
    elif args.config:
        overrides = {'data_dir': args.data_dir} if args.data_dir else None
        summary = run_single_experiment(args.config, overrides)
        print(f"\n{'='*80}")
        print(f"Experiment Complete!")
        print(f"{'='*80}")
        print(json.dumps(summary, indent=2, ensure_ascii=False,
                         default=lambda x: int(x) if isinstance(x, np.integer)
                         else float(x) if isinstance(x, np.floating) else str(x)))
    else:
        # 默认运行
        config_path = 'configs/cifar10/config_de_asfl_dynamic_failure.yaml'
        print(f"未指定配置文件，使用默认配置: {config_path}")
        summary = run_single_experiment(config_path)
        print(f"\n{'='*80}")
        print(f"Experiment Complete!")
        print(f"{'='*80}")
        print(json.dumps(summary, indent=2, ensure_ascii=False,
                         default=lambda x: int(x) if isinstance(x, np.integer)
                         else float(x) if isinstance(x, np.floating) else str(x)))


if __name__ == "__main__":
    main()
