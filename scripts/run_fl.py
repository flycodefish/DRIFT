"""主运行文件：实现 Gossip 协议（mesh拓扑）和 FedAvg 算法（star拓扑）"""

import argparse
import time
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Prefer this checkout when launched as ``python scripts/run_fl.py``. This
# prevents another editable/install copy of ``fl_core`` from being imported.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if sys.path[0] != project_root_str:
    sys.path.insert(0, project_root_str)

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from fl_core.config import FLConfig
from fl_core.runtime import RayFLRuntime
import ray

try:
    from fl_core.plotter import generate_summary_report
    PLOTTER_AVAILABLE = True
except ImportError:
    PLOTTER_AVAILABLE = False


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """从YAML配置文件加载参数"""
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML is not installed. Please install it with 'pip install pyyaml'")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args() -> FLConfig:
    """解析命令行参数和配置文件"""
    parser = argparse.ArgumentParser(description="Run Federated Learning algorithms")
    
    # 配置文件参数
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file"
    )
    parser.add_argument("--frozen-inputs-dir", type=str, default=None,
                        help="Shared immutable input bundle directory")
    
    # 命令行参数（可覆盖配置文件）
    parser.add_argument(
        "--algorithm",
        choices=["fedavg", "gossip", "dtd"],
        default=None,
        help="FL algorithm to run"
    )
    parser.add_argument("--program", type=str, default=None)
    parser.add_argument(
        "--topology",
        choices=["star", "ring", "mesh"],
        default=None,
        help="Network topology"
    )
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--num-clients", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--client-sample-ratio", type=float, default=None)
    parser.add_argument("--client-drop-rate", type=float, default=None)
    
    # Gossip-specific parameters
    parser.add_argument(
        "--neighbor-strategy",
        choices=["fixed", "random_per_round"],
        default=None,
        help="Neighbor selection strategy for mesh topology"
    )
    parser.add_argument("--neighbor-k", type=int, default=None)

    # DTD-specific parameters
    parser.add_argument("--jsd-temperature", type=float, default=None,
                        help="JSD temperature for DTD mixing weights")
    
    # Data parameters
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "cifar10", "cifar100", "har"],
        default=None
    )
    parser.add_argument(
        "--data-partition-mode",
        choices=["iid", "label_skew", "dirichlet"],
        default=None
    )
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--input-dim", type=int, default=None)
    parser.add_argument("--dirichlet-alpha", type=float, default=None)
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Dataset root directory (overrides YAML data_dir)")
    
    # Model parameters
    parser.add_argument("--model", type=str, default=None)
    
    # Output parameters
    parser.add_argument("--output-dir", type=str, default=None)
    
    args = parser.parse_args()
    
    # 初始化配置字典
    config_dict = {}
    
    # 如果提供了配置文件，加载配置
    if args.config:
        config_dict = load_config_from_yaml(args.config)
    
    # 用命令行参数覆盖配置文件（仅覆盖非None的值）
    def set_param(key: str, arg_value, config_key: str = None):
        if arg_value is not None:
            if config_key is None:
                # 去掉开头的 "--" 然后替换剩余的 "-" 为 "_"
                config_key = key.lstrip('-').replace('-', '_')
            config_dict[config_key] = arg_value
    
    set_param("--algorithm", args.algorithm)
    set_param("--program", args.program, "program_name")
    set_param("--topology", args.topology)
    set_param("--rounds", args.rounds)
    set_param("--num-clients", args.num_clients, "num_clients")
    set_param("--local-epochs", args.local_epochs, "local_epochs")
    set_param("--batch-size", args.batch_size, "batch_size")
    set_param("--learning-rate", args.learning_rate, "learning_rate")
    set_param("--client-sample-ratio", args.client_sample_ratio, "client_sample_ratio")
    set_param("--client-drop-rate", args.client_drop_rate, "client_drop_rate")
    set_param("--neighbor-strategy", args.neighbor_strategy, "neighbor_strategy")
    set_param("--neighbor-k", args.neighbor_k, "neighbor_k")
    set_param("--jsd-temperature", args.jsd_temperature, "jsd_temperature")
    set_param("--dataset", args.dataset)
    set_param("--data-partition-mode", args.data_partition_mode, "data_partition_mode")
    set_param("--num-classes", args.num_classes, "num_classes")
    set_param("--input-dim", args.input_dim, "input_dim")
    set_param("--dirichlet-alpha", args.dirichlet_alpha, "dirichlet_alpha")
    set_param("--data-dir", args.data_dir, "data_dir")
    set_param("--model", args.model)
    set_param("--output-dir", args.output_dir, "output_dir")
    set_param("--frozen-inputs-dir", args.frozen_inputs_dir, "frozen_inputs_dir")
    
    # 根据算法自动设置拓扑
    algorithm = config_dict.get("algorithm", "fedavg")
    if algorithm in ("fedavg", "fedprox"):
        config_dict["topology"] = "star"
        config_dict["program_name"] = config_dict.get("program_name", algorithm)
    elif algorithm == "dtd":
        config_dict["topology"] = "mesh"
        config_dict["program_name"] = config_dict.get("program_name", "dtd")
    else:
        config_dict["topology"] = "mesh"
        config_dict["program_name"] = config_dict.get("program_name", "gossip")
    
    # 设置默认值（如果配置文件和命令行都没有提供）
    defaults = {
        "seed": 42,
        "rounds": 100,
        "num_clients": 30,
        "local_epochs": 1,
        "batch_size": 64,
        "learning_rate": 0.05,
        "client_sample_ratio": 1.0,
        "client_drop_rate": 0.0,
        "neighbor_strategy": "random_per_round",
        "neighbor_k": 3,
        "dataset": "synthetic",
        "model": "cnn",
        "data_partition_mode": "iid",
        "num_classes": 3,
        "input_dim": 20,
        "dirichlet_alpha": 0.5,
        "output_dir": "outputs",
        "algorithm": "fedavg",
        "jsd_temperature": 1.0,
    }
    
    # 合并默认值
    for key, default_value in defaults.items():
        if key not in config_dict:
            config_dict[key] = default_value
    
    # 创建FLConfig对象
    cfg = FLConfig(**config_dict)
    
    return cfg


def aggregate_models_fedavg(payloads: list[dict]) -> dict:
    """
    FedAvg 聚合策略：基于样本数量的加权平均
    
    Args:
        payloads: 包含 {"client_id", "params", "num_examples"} 的列表
    
    Returns:
        聚合后的模型参数
    """
    if not payloads:
        raise ValueError("No payloads to aggregate")

    from deasfl.aggregation import sample_weighted_average

    counts = [item["num_examples"] for item in payloads]
    if sum(counts) == 0:
        counts = [1] * len(payloads)
    return sample_weighted_average(
        [item["params"] for item in payloads], counts
    )


def run_fedavg(runtime: RayFLRuntime, cfg: FLConfig, run_timestamp: str) -> None:
    """运行 FedAvg 算法（星形拓扑）"""
    params = runtime.init_model_params()
    runtime.save_experiment_inputs(params)
    start_time = time.perf_counter()
    
    print("=" * 70)
    print(f"Running FedAvg with {cfg.num_clients} clients, {cfg.rounds} rounds")
    print(f"Dataset: {cfg.dataset}, Model: {cfg.model}")
    print(f"Data partition: {cfg.data_partition_mode} (alpha={cfg.dirichlet_alpha})")
    if cfg.proximal_mu > 0:
        print(f"FedProx mode: proximal_mu = {cfg.proximal_mu}")
    print("=" * 70)

    # FedProx: 设置proximal term强度
    if cfg.proximal_mu > 0:
        for client_id in range(cfg.num_clients):
            runtime.clients[client_id].set_proximal_mu.remote(cfg.proximal_mu)
    
    for round_id in range(cfg.rounds):
        round_start_time = time.perf_counter()
        print(f"\n[Round {round_id + 1}/{cfg.rounds}]")
        
        runtime.broadcast_model(params, round_id)

        # FedProx: 设置全局参数作为proximal term的参考点
        if cfg.proximal_mu > 0:
            for client_id in range(cfg.num_clients):
                runtime.clients[client_id].set_global_params.remote(params)

        selected = runtime.select_clients()
        active, dropped = runtime.inject_dropouts(selected, round_id)
        
        print(f"  Selected: {len(selected)}, Active: {len(active)}, Dropped: {len(dropped)}")
        
        if not active:
            print("  No active clients, skipping this round")
            continue
        
        train_results = runtime.train_selected_clients(active, round_id)
        payloads = runtime.collect_client_models(active, round_id)
        
        if not payloads:
            print("  No payloads collected, skipping aggregation")
            continue
        
        params = aggregate_models_fedavg(payloads)
        val_loss, val_acc = runtime.evaluate_global(params)
        avg_train_loss = sum(r["train_loss"] for r in train_results) / max(1, len(train_results))
        comm_summary = runtime.summarize_round_comm(round_id)
        
        runtime.monitor.record_round(
            round_id=round_id,
            avg_train_loss=avg_train_loss,
            val_loss=val_loss,
            val_acc=val_acc,
            selected_count=len(selected),
            dropped_count=len(dropped),
            comm_success_count=comm_summary["comm_success_count"],
            comm_failure_count=comm_summary["comm_failure_count"],
            comm_timeout_count=comm_summary["comm_timeout_count"],
            broadcast_success_count=comm_summary["broadcast_success_count"],
            broadcast_failure_count=comm_summary["broadcast_failure_count"],
            upload_success_count=comm_summary["upload_success_count"],
            upload_failure_count=comm_summary["upload_failure_count"],
            upload_storage_blocked_count=comm_summary["upload_storage_blocked_count"],
            gossip_success_count=0,
            gossip_failure_count=0,
            round_wall_clock_seconds=time.perf_counter() - round_start_time,
        )
        
        print(f"  Train loss: {avg_train_loss:.4f}")
        print(f"  Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")
        print(f"  Comm: success={comm_summary['comm_success_count']}, "
              f"failure={comm_summary['comm_failure_count']}, "
              f"timeout={comm_summary['comm_timeout_count']}")
    
    end_time = time.perf_counter()
    print("\n" + "=" * 70)
    print(f"FedAvg completed in {end_time - start_time:.2f}s")
    print("=" * 70)
    
    model_path = runtime.save_final_model(params)
    print(f"\nFinal model saved to: {model_path}")
    
    # 生成图表报告
    generate_report(runtime, cfg, run_timestamp)


def run_gossip(runtime: RayFLRuntime, cfg: FLConfig, run_timestamp: str) -> None:
    """运行 Gossip 协议（网状拓扑）"""
    params = runtime.init_model_params()
    runtime.save_experiment_inputs(params)
    start_time = time.perf_counter()
    
    print("=" * 70)
    print(f"Running Gossip with {cfg.num_clients} clients, {cfg.rounds} rounds")
    print(f"Topology: {cfg.topology}, Neighbor strategy: {cfg.neighbor_strategy}, k={cfg.neighbor_k}")
    print(f"Dataset: {cfg.dataset}, Model: {cfg.model}")
    print(f"Data partition: {cfg.data_partition_mode} (alpha={cfg.dirichlet_alpha})")
    print("=" * 70)
    
    print("\nInitializing models on all clients...")
    runtime.broadcast_model(params, round_id=0)
    
    for round_id in range(cfg.rounds):
        round_start_time = time.perf_counter()
        print(f"\n[Round {round_id + 1}/{cfg.rounds}]")
        
        selected = runtime.select_clients()
        active, dropped = runtime.inject_dropouts(selected, round_id)
        
        print(f"  Selected: {len(selected)}, Active: {len(active)}, Dropped: {len(dropped)}")
        
        if not active:
            print("  No active clients, skipping this round")
            continue
        
        train_results = runtime.train_selected_clients(active, round_id)
        gossip_stats = runtime.push_gossip_async(round_id, active)
        
        # 等待Gossip完成后再评估
        if gossip_stats.get('pending_refs'):
            # 使用 num_returns 参数等待所有任务完成
            ray.wait(gossip_stats['pending_refs'], num_returns=len(gossip_stats['pending_refs']))
            # 清空列表释放 Ray 引用计数
            gossip_stats['pending_refs'] = []
        
        # 强制进行 Python 垃圾回收，释放 Ray ObjectRef
        import gc
        gc.collect()
        
        # 清理所有客户端的GPU显存缓存，释放临时张量
        runtime.clear_all_client_caches()
        
        include_local = (
            cfg.decentralized_diagnostic_interval > 0
            and (round_id + 1) % cfg.decentralized_diagnostic_interval == 0
        )
        params, decentralized_metrics = runtime.evaluate_decentralized(
            active, include_local=include_local
        )
        val_loss = decentralized_metrics["consensus_loss"]
        val_acc = decentralized_metrics["consensus_accuracy"]
        avg_train_loss = sum(r["train_loss"] for r in train_results) / max(1, len(train_results))
        comm_summary = runtime.summarize_round_comm(round_id)
        
        runtime.monitor.record_round(
            round_id=round_id,
            avg_train_loss=avg_train_loss,
            val_loss=val_loss,
            val_acc=val_acc,
            selected_count=len(selected),
            dropped_count=len(dropped),
            comm_success_count=comm_summary["comm_success_count"],
            comm_failure_count=comm_summary["comm_failure_count"],
            comm_timeout_count=comm_summary["comm_timeout_count"],
            broadcast_success_count=comm_summary["broadcast_success_count"],
            broadcast_failure_count=comm_summary["broadcast_failure_count"],
            upload_success_count=comm_summary["upload_success_count"],
            upload_failure_count=comm_summary["upload_failure_count"],
            upload_storage_blocked_count=comm_summary["upload_storage_blocked_count"],
            gossip_success_count=gossip_stats.get("gossip_push_success_count", 0),
            gossip_failure_count=gossip_stats.get("gossip_push_failure_count", 0),
            round_wall_clock_seconds=time.perf_counter() - round_start_time,
        )
        runtime.monitor.record_decentralized_metrics(
            round_id, **decentralized_metrics
        )
        
        print(f"  Train loss: {avg_train_loss:.4f}")
        print(f"  Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")
        # print(f"  Gossip ops: {gossip_stats.get('pending_refs', 0)}")
        print(f"  Comm: success={comm_summary['comm_success_count']}, "
              f"failure={comm_summary['comm_failure_count']}, "
              f"timeout={comm_summary['comm_timeout_count']}")
    
    end_time = time.perf_counter()
    print("\n" + "=" * 70)
    print(f"Gossip completed in {end_time - start_time:.2f}s")
    print("=" * 70)
    
    model_path = runtime.save_final_model(params)
    print(f"\nFinal model saved to: {model_path}")
    
    # 生成图表报告
    generate_report(runtime, cfg, run_timestamp)


def run_dtd(runtime: RayFLRuntime, cfg: FLConfig, run_timestamp: str) -> None:
    """DTD: Distribution-aware Decentralized Training.

    Each client pushes its model to neighbors. Receivers average using
    JSD-based weights — neighbors with similar label distributions get
    higher mixing weights.
    """
    params = runtime.init_model_params()
    runtime.save_experiment_inputs(params)
    start_time = time.perf_counter()

    # Compute and distribute label distributions
    label_dists = runtime._compute_label_distributions()
    runtime._distribute_label_dists(label_dists)

    # Set JSD temperature on all clients
    temperature = getattr(cfg, 'jsd_temperature', 1.0)
    temperature_refs = [runtime.clients[cid].set_jsd_temperature.remote(temperature)
                        for cid in range(cfg.num_clients)]
    ray.get(temperature_refs)

    print("=" * 70)
    print(f"Running DTD with {cfg.num_clients} clients, {cfg.rounds} rounds")
    print(f"Topology: {cfg.topology}, Neighbor strategy: {cfg.neighbor_strategy}, k={cfg.neighbor_k}")
    print(f"JSD temperature: {temperature}")
    print(f"Dataset: {cfg.dataset}, Model: {cfg.model}")
    print(f"Data partition: {cfg.data_partition_mode} (alpha={cfg.dirichlet_alpha})")
    print("=" * 70)

    print("\nInitializing models on all clients...")
    runtime.broadcast_model(params, round_id=0)

    for round_id in range(cfg.rounds):
        round_start_time = time.perf_counter()
        print(f"\n[Round {round_id + 1}/{cfg.rounds}]")

        selected = runtime.select_clients()
        active, dropped = runtime.inject_dropouts(selected, round_id)

        print(f"  Selected: {len(selected)}, Active: {len(active)}, Dropped: {len(dropped)}")

        if not active:
            print("  No active clients, skipping this round")
            continue

        train_results = runtime.train_selected_clients(active, round_id)
        dtd_stats = runtime.push_dtd_async(round_id, active, label_dists)

        if dtd_stats.get('pending_refs'):
            ray.wait(dtd_stats['pending_refs'],
                     num_returns=len(dtd_stats['pending_refs']))
            dtd_stats['pending_refs'] = []

        # Force GC to release Ray ObjectRef
        import gc
        gc.collect()

        runtime.clear_all_client_caches()

        include_local = (
            cfg.decentralized_diagnostic_interval > 0
            and (round_id + 1) % cfg.decentralized_diagnostic_interval == 0
        )
        params, decentralized_metrics = runtime.evaluate_decentralized(
            active, include_local=include_local
        )
        val_loss = decentralized_metrics["consensus_loss"]
        val_acc = decentralized_metrics["consensus_accuracy"]
        avg_train_loss = sum(r["train_loss"] for r in train_results) / max(1, len(train_results))
        comm_summary = runtime.summarize_round_comm(round_id)

        runtime.monitor.record_round(
            round_id=round_id,
            avg_train_loss=avg_train_loss,
            val_loss=val_loss,
            val_acc=val_acc,
            selected_count=len(selected),
            dropped_count=len(dropped),
            comm_success_count=comm_summary["comm_success_count"],
            comm_failure_count=comm_summary["comm_failure_count"],
            comm_timeout_count=comm_summary["comm_timeout_count"],
            broadcast_success_count=comm_summary["broadcast_success_count"],
            broadcast_failure_count=comm_summary["broadcast_failure_count"],
            upload_success_count=comm_summary["upload_success_count"],
            upload_failure_count=comm_summary["upload_failure_count"],
            upload_storage_blocked_count=comm_summary["upload_storage_blocked_count"],
            gossip_success_count=dtd_stats.get("gossip_push_success_count", 0),
            gossip_failure_count=dtd_stats.get("gossip_push_failure_count", 0),
            round_wall_clock_seconds=time.perf_counter() - round_start_time,
        )
        runtime.monitor.record_decentralized_metrics(
            round_id, **decentralized_metrics
        )

        print(f"  Train loss: {avg_train_loss:.4f}")
        print(f"  Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")
        print(f"  Comm: success={comm_summary['comm_success_count']}, "
              f"failure={comm_summary['comm_failure_count']}, "
              f"timeout={comm_summary['comm_timeout_count']}")

    end_time = time.perf_counter()
    print("\n" + "=" * 70)
    print(f"DTD completed in {end_time - start_time:.2f}s")
    print("=" * 70)

    model_path = runtime.save_final_model(params)
    print(f"\nFinal model saved to: {model_path}")

    generate_report(runtime, cfg, run_timestamp)


def generate_report(runtime: RayFLRuntime, cfg: FLConfig, run_timestamp: str) -> None:
    """生成训练报告和图表"""
    if not PLOTTER_AVAILABLE:
        print("\nWarning: Plotter not available (matplotlib/seaborn not installed)")
        return
    
    # 当前输出目录已经是时间命名的文件夹
    output_dir = cfg.output_dir
    
    # 使用 runtime.run_name 来构建正确的文件名
    # run_name 的格式: {program_name}_{topology}_{num_clients}c_{rounds}r
    run_name = runtime.run_name
    
    # 先确保 metrics 文件已经导出（因为 shutdown() 还没调用）
    runtime.monitor.dump(output_dir, run_name)
    
    # 加载轮次指标
    import pandas as pd
    metrics_path = os.path.join(output_dir, f"{run_name}_round_metrics.csv")
    if os.path.exists(metrics_path):
        round_metrics = pd.read_csv(metrics_path)
        
        # 生成报告和图表
        config_dict = {
            'algorithm': cfg.algorithm,
            'topology': cfg.topology,
            'num_clients': cfg.num_clients,
            'rounds': cfg.rounds,
            'dataset': cfg.dataset,
            'model': cfg.model,
            'data_partition_mode': cfg.data_partition_mode,
            'dirichlet_alpha': cfg.dirichlet_alpha,
        }
        generate_summary_report(round_metrics, config_dict, output_dir)
    else:
        print(f"Warning: Metrics file not found: {metrics_path}")
        print(f"Available files in output directory:")
        for f in os.listdir(output_dir):
            print(f"  - {f}")


def main():
    """主函数"""
    # 获取当前时间戳（用于输出文件命名）
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 解析参数
    cfg = parse_args()
    
    # 创建时间命名的输出文件夹
    timestamp_output_dir = os.path.join(cfg.output_dir, f"{cfg.program_name}_{run_timestamp}")
    os.makedirs(timestamp_output_dir, exist_ok=True)
    
    # 更新输出目录为时间命名的文件夹
    cfg.output_dir = timestamp_output_dir
    
    # 更新程序名称（不再需要时间戳后缀，因为目录已经有时间戳了）
    original_program_name = cfg.program_name
    cfg.program_name = f"{cfg.program_name}"
    
    print(f"\nRun timestamp: {run_timestamp}")
    print(f"Output directory: {cfg.output_dir}")
    print(f"Run name: {cfg.program_name}")
    
    # 初始化运行时
    runtime = RayFLRuntime(cfg)
    runtime.start()
    
    try:
        if cfg.algorithm == "dtd":
            run_dtd(runtime, cfg, run_timestamp)
        elif cfg.algorithm in ("fedavg", "fedprox") or cfg.topology == "star":
            run_fedavg(runtime, cfg, run_timestamp)
        else:
            run_gossip(runtime, cfg, run_timestamp)
        runtime.save_completion_status("COMPLETE")
    except BaseException as exc:
        runtime.save_completion_status("FAILED", error=repr(exc))
        raise
    finally:
        runtime.shutdown()


if __name__ == "__main__":
    # Keep CUDA allocator/module settings outside this launcher.  In the
    # current fl_env setting them after importing torch can hide CUDA.
    
    main()
