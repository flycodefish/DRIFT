from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FLConfig:
    seed: int = 42
    deterministic_training: bool = True
    frozen_inputs_dir: str = ""
    strict_frozen_inputs: bool = False
    decentralized_diagnostic_interval: int = 1
    num_clients: int = 4
    rounds: int = 5
    local_epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 0.001  # Adam 默认使用较小的学习率
    learning_rate_min: float = 1e-5  # 最小学习率
    warmup_rounds: int = 5  # Warmup 轮数
    cosine_annealing: bool = True  # 是否使用 Cosine Annealing
    learning_rate_schedule_rounds: int = 0  # 0 = follow rounds
    client_sample_ratio: float = 1.0
    client_drop_rate: float = 0.0
    link_bandwidth_mbps: float = 100.0
    link_latency_ms: float = 5.0
    link_drop_rate: float = 0.0
    link_profile_mode: str = "homogeneous"  # homogeneous | role_aware
    server_link_bandwidth_mbps: float | None = None
    server_link_latency_ms: float | None = None
    server_link_drop_rate: float | None = None
    peer_link_bandwidth_mbps: float | None = None
    peer_link_latency_ms: float | None = None
    peer_link_drop_rate: float | None = None
    max_comm_retries: int = 0
    comm_timeout_ms: float = 0.0
    backoff_base_ms: float = 0.0
    backoff_multiplier: float = 2.0
    enforce_comm_delay: bool = False

    num_samples: int = 4000
    input_dim: int = 20
    num_classes: int = 3
    data_partition_mode: str = "iid"  # iid | label_skew | dirichlet
    classes_per_client: int = 2
    dirichlet_alpha: float = 0.5
    data_dir: str = "./data"
    dataset_split_policy: str = "default"
    har_local_val_seed: int = 42
    weight_decay: float = 1e-4
    use_gpu: bool = True
    num_gpus: float = 0.066
    max_concurrent_gpu_clients: int = 8
    client_speed_mode: str = "uniform"  # uniform | two_group
    slow_client_ratio: float = 0.5
    slow_client_delay_ms: float = 0.0
    client_network_mode: str = "uniform"  # uniform | two_group
    slow_network_client_ratio: float = 0.5
    slow_network_latency_multiplier: float = 2.0
    slow_network_bandwidth_multiplier: float = 0.5
    slow_network_drop_addition: float = 0.1
    client_storage_mode: str = "uniform"  # uniform | two_group
    low_storage_client_ratio: float = 0.5
    low_storage_max_upload_bytes: int = 0
    edge_profile_json: str = ""
    dynamic_failure_trace_json: str = ""
    dynamic_initial_inactive_nodes: str = ""
    # Epidemic Learning: number of random overlay peers sampled per round.
    # Zero follows the paper-inspired ceil(log2(num_clients)) default.
    epidemic_sample_size: int = 0

    topology: str = "star"  # star | ring | mesh
    neighbor_strategy: str = "fixed"  # fixed | random_per_round
    neighbor_k: int = 2
    fixed_hierarchical_root_id: int = 0
    fixed_hierarchical_num_clusters: int = 2
    program_name: str = "fedavg"
    algorithm: str = "fedavg"  # fedavg | gossip | fedprox | dtd
    proximal_mu: float = 0.0  # FedProx proximal term strength (0 = disabled)
    jsd_temperature: float = 1.0  # DTD JSD temperature for mixing weights
    output_dir: str = "outputs"
    dataset: str = "synthetic"  # synthetic | cifar10
    data_dir: str = "./data"
    model: str = "cnn"  # cnn | resnet18
    use_gpu: bool = True
    num_gpus: float = 0.066
