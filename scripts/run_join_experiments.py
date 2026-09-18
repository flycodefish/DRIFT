"""
Node Join Adaptability 实验脚本（论文 Section VI-I）

场景: 20 初始客户端，第 50 轮加入 10 个新客户端，总 30。
对比三种策略: Static / Global Rebuild / Localized Join

用法:
    python scripts/run_join_experiments.py          # 跑全部三组
    python scripts/run_join_experiments.py --strategy localized  # 单组
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datetime import datetime

from deasfl import NewFLConfig, NewFLRuntime
from deasfl.runtime import load_cifar10_data


def run_join_experiment(strategy: str, config_path: str,
                        initial_active: int = 20, total_clients: int = 30,
                        join_round: int = 50, rounds: int = None) -> dict:
    """运行单组 join 实验

    Args:
        strategy: static | global_rebuild | localized
        config_path: 基础 YAML 配置文件
        initial_active: 初始活跃客户端数
        total_clients: 总潜在客户端数
        join_round: 加入轮次
    """
    cfg = NewFLConfig.from_yaml(config_path)

    exp_name = f"join_{strategy}_20to30_r{join_round}"
    override = {
        'num_clients': total_clients,
        'initial_active_clients': initial_active,
        'node_join_enabled': True,
        'join_start_round': join_round,
        'join_max_nodes': total_clients - initial_active,
        'join_strategy': strategy,
        'enable_dynamic_nodes': False,
        'experiment_name': exp_name,
    }
    if rounds is not None:
        override['rounds'] = rounds
    for key, value in override.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    print(f"\n{'='*80}")
    print(f"Node Join Experiment: {exp_name}")
    print(f"{'='*80}")
    print(f"  Initial active: {initial_active}, Total: {total_clients}")
    print(f"  Join round: {join_round}, Strategy: {strategy}")

    runtime = NewFLRuntime(cfg)
    runtime.start()

    try:
        train_data, val_data, indices, global_test_data, label_dists = load_cifar10_data(
            data_dir=cfg.data_dir, num_clients=cfg.num_clients,
            partition_mode=cfg.partition, alpha=cfg.alpha, seed=cfg.seed,
            client_indices=(runtime.frozen_inputs["data_partition.json"]["client_indices"]
                            if runtime.frozen_inputs else None))

        runtime.label_distributions = label_dists

        runtime.init_clients(train_data, val_data, indices)
        runtime.build_topology()

        if cfg.enable_election:
            runtime.init_election(cfg.election_base_trigger_time, cfg.election_trigger_factor)
            election_result = runtime.run_election()
        else:
            election_result = {'best_client_id': 0, 'best_client_score': 0.0,
                              'convergence_hops': 0, 'all_nodes_info': []}

        if cfg.enable_clustering:
            runtime.init_clustering(election_result, cfg.cluster_max_child, cfg.cluster_offset)
            clustering_result = runtime.run_clustering()
            runtime.set_clustering_results(clustering_result, election_result)
        else:
            runtime.clustering_results = None
            runtime.tree_mapping = None

        runtime._init_ray_actors(train_data, val_data, global_test_data)
        training_results = runtime.run_training(num_rounds=cfg.rounds)

    finally:
        runtime.shutdown()

    # 构建摘要
    final_global_acc = (float(training_results['global_accuracies'][-1])
                        if training_results['global_accuracies'] else 0.0)

    summary = {
        'experiment_name': cfg.experiment_name,
        'strategy': strategy,
        'initial_active': initial_active,
        'total_clients': total_clients,
        'join_round': join_round,
        'joined_nodes': sorted(runtime.joined_nodes),
        'join_log': training_results.get('join_log', []),
        'final_global_acc': final_global_acc,
        'final_avg_acc': final_global_acc,
        'final_max_acc': final_global_acc,
        'training_time': training_results['training_time'],
        'num_rounds': cfg.rounds,
        'dataset': cfg.dataset,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description='Node Join Experiments')
    parser.add_argument('--config', type=str,
                        default='configs/cifar10/config_de_asfl_dynamic_failure.yaml',
                        help='基础 YAML 配置文件')
    parser.add_argument('--strategy', type=str, default=None,
                        choices=['static', 'global_rebuild', 'localized'],
                        help='单跑一个策略（不指定则跑全部三组 + baseline）')
    parser.add_argument('--initial-active', type=int, default=20)
    parser.add_argument('--total-clients', type=int, default=30)
    parser.add_argument('--join-round', type=int, default=50)
    args = parser.parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"

    strategies = [args.strategy] if args.strategy else [
        'static', 'global_rebuild', 'localized']

    all_summaries = []

    # Baseline: 30 clients from start, no join
    print(f"\n{'#'*80}")
    print(f"Baseline: 30 clients from start (no join)")
    print(f"{'#'*80}")
    cfg = NewFLConfig.from_yaml(args.config)
    for k, v in {'num_clients': 30, 'initial_active_clients': 0,
                 'node_join_enabled': False,
                 'experiment_name': 'join_baseline_30c'}.items():
        setattr(cfg, k, v)
    runtime = NewFLRuntime(cfg)
    runtime.start()
    try:
        train_data, val_data, indices, global_test_data, label_dists = load_cifar10_data(
            data_dir=cfg.data_dir, num_clients=cfg.num_clients,
            partition_mode=cfg.partition, alpha=cfg.alpha, seed=cfg.seed,
            client_indices=(runtime.frozen_inputs["data_partition.json"]["client_indices"]
                            if runtime.frozen_inputs else None))
        runtime.label_distributions = label_dists
        runtime.init_clients(train_data, val_data, indices)
        runtime.build_topology()
        if cfg.enable_election:
            runtime.init_election(cfg.election_base_trigger_time, cfg.election_trigger_factor)
            er = runtime.run_election()
        else:
            er = {'best_client_id': 0, 'best_client_score': 0.0,
                  'convergence_hops': 0, 'all_nodes_info': []}
        if cfg.enable_clustering:
            runtime.init_clustering(er, cfg.cluster_max_child, cfg.cluster_offset)
            cr = runtime.run_clustering()
            runtime.set_clustering_results(cr, er)
        runtime._init_ray_actors(train_data, val_data, global_test_data)
        tr = runtime.run_training(num_rounds=cfg.rounds)
    finally:
        runtime.shutdown()
    baseline_acc = float(tr['global_accuracies'][-1]) if tr['global_accuracies'] else 0.0
    all_summaries.append({'experiment_name': 'baseline_30c', 'strategy': 'baseline',
                          'final_global_acc': baseline_acc})
    print(f"  Baseline (30c): {baseline_acc:.4f}")

    for strategy in strategies:
        summary = run_join_experiment(
            strategy=strategy, config_path=args.config,
            initial_active=args.initial_active,
            total_clients=args.total_clients,
            join_round=args.join_round)
        all_summaries.append(summary)

    # 汇总
    print(f"\n{'='*80}")
    print(f"Join Experiment Summary")
    print(f"{'='*80}")
    for s in all_summaries:
        print(f"  {s['strategy']:>20s}: global_acc={s.get('final_global_acc', 0):.4f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f"outputs/join_experiments_{timestamp}.json"
    os.makedirs("outputs", exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False,
                  default=lambda x: int(x) if isinstance(x, np.integer)
                  else float(x) if isinstance(x, np.floating) else str(x))
    print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    main()
