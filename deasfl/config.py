"""
配置类模块：定义联邦学习实验的所有配置参数

该模块使用 Python dataclass 来管理配置，提供以下功能：
1. 定义所有训练参数的默认值
2. 支持从 YAML 文件加载配置
3. 提供输出路径生成方法

配置参数分类：
- 实验设置：实验名称、输出目录
- 数据集设置：数据集类型、数据路径
- 模型设置：模型类型、类别数
- 训练设置：客户端数、轮数、批次大小、学习率等
- 学习率调度：warmup轮数、cosine退火等
- 数据划分：划分策略、异构程度参数
- 算法设置：算法类型、聚合策略
- 通信设置：带宽、延迟、丢包率
- 设备设置：是否使用GPU、GPU资源分配
- 日志设置：日志间隔、是否保存模型和图表
"""

import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class NewFLConfig:
    """
    联邦学习配置类，存储所有实验参数
    
    使用示例：
    >>> cfg = NewFLConfig(num_clients=30, rounds=100)
    >>> cfg = NewFLConfig.from_yaml("config.yaml")
    """
    
    # ==================== 实验设置 ====================
    experiment_name: str = "deasfl"  # 实验名称，用于输出目录命名
    output_dir: str = "outputs"       # 输出目录
    run_name: str = ""                # 运行名称（可选）
    seed: int = 42                    # 随机种子，确保实验可重复
    
    # ==================== 数据集设置 ====================
    dataset: str = "cifar10"   # 数据集类型：cifar10, mnist 等
    data_dir: str = "./data"   # 数据集存储路径
    
    # ==================== 模型设置 ====================
    model: str = "resnet18"     # 模型类型：resnet18, resnet34 等
    num_classes: int = 10       # 分类类别数（CIFAR-10 为 10）
    
    # ==================== 训练设置 ====================
    num_clients: int = 30           # 客户端数量
    rounds: int = 100               # 联邦学习轮数
    local_epochs: int = 5           # 每轮每个客户端的本地训练轮数
    batch_size: int = 32            # 批次大小
    learning_rate: float = 0.001    # 初始学习率
    weight_decay: float = 1e-4      # 权重衰减（L2正则化）
    
    # ==================== 学习率调度设置 ====================
    learning_rate_min: float = 1e-5  # 最小学习率（cosine退火结束值）
    warmup_rounds: int = 5           # warmup轮数（学习率线性增长阶段）
    cosine_annealing: bool = True    # 是否启用cosine退火
    
    # ==================== 数据划分设置 ====================
    partition: str = "dirichlet"  # 数据划分策略：dirichlet, random 等
    alpha: float = 0.1            # Dirichlet分布参数，越小异构性越强
    
    # ==================== 算法设置 ====================
    algorithm: str = "custom"              # 算法类型
    aggregation_strategy: str = "hierarchical_weighted"  # 原始分层样本加权平均
    
    # ==================== 拓扑设置（去中心化） ====================
    topology_type: str = "mesh"            # 拓扑类型：mesh, ring, star, fully_connected
    neighbor_selection: str = "random"      # 邻居选择策略：static, performance_based, random
    min_neighbors: int = 2                 # 每个节点的最小邻居数量
    max_neighbors: int = 4                 # 每个节点的最大邻居数量
    static_neighbors: Optional[list] = None  # 静态邻居配置，如 [[0,1,2], [0,2,3]] 表示client0连接0,1,2

    # ==================== 选举设置 ====================
    enable_election: bool = True           # 是否启用分布式选举
    election_base_trigger_time: float = 10.0  # 选举基础触发时间（秒）
    election_trigger_factor: float = 0.1   # 选举触发时间因子（得分乘数）
    cpu_cores_range: tuple = (2, 16)       # CPU核心数随机范围
    memory_gb_range: tuple = (4, 64)       # 内存大小随机范围（GB）
    bandwidth_mbps_range: tuple = (50, 1000)  # 带宽随机范围（Mbps）
    gpu_memory_gb_range: tuple = (0, 32)   # GPU内存随机范围（GB）

    # ==================== 聚类设置 ====================
    enable_clustering: bool = True         # 是否启用分布式聚类
    cluster_max_child: int = 3             # 每个簇首最大子节点数
    cluster_max_retry: int = 3             # 聚合请求最大重试次数
    cluster_request_timeout: float = 2.0    # 请求超时时间（秒）
    cluster_offset: float = 30.0           # 聚类触发时间偏移量（秒）
    cluster_trigger_factor: float = 0.05    # 聚类触发时间因子
    distribution_similarity_threshold: float = 0.0  # 分布感知聚类JSD阈值（0=禁用，0.3=中等相似度）

    # ==================== 动态节点设置（Node Failure / Join） ====================
    enable_dynamic_nodes: bool = False     # 是否启用动态节点故障/加入
    failure_rate: float = 0.0             # 每轮节点故障概率（0-1）
    failure_start_round: int = 10         # 开始故障的轮次
    failure_max_nodes: int = 5            # 最多故障节点数
    failure_pattern: str = "random"       # 故障模式: random | leaf_only | cluster_head | best_client
    node_join_enabled: bool = False       # 是否启用新节点加入
    join_rate: float = 0.0               # 每轮新节点加入概率（0-1）
    join_start_round: int = 20           # 开始加入的轮次
    join_max_nodes: int = 5              # 最多加入节点数
    initial_active_clients: int = 0      # 初始活跃客户端数（0=全部激活，用于增量加入实验）
    join_strategy: str = "localized"     # 加入策略: static | global_rebuild | localized
    recovery_strategy: str = "skip"      # 恢复策略: skip | reassign | reelect_recluster

    # ==================== 通信设置 ====================
    bandwidth: Optional[float] = None  # 带宽限制（MB/s），None表示无限制
    latency: Optional[float] = None     # 延迟（ms），None表示无延迟
    drop_rate: float = 0.0              # 丢包率（0-1）
    async_communication: bool = True    # 是否使用异步通信
    
    # ==================== 设备设置 ====================
    use_gpu: bool = True   # 是否使用GPU
    num_gpus: float = 0.05 # 每个客户端分配的GPU资源比例
    use_ray_actors: bool = True  # False 时使用单进程顺序模拟
    
    # ==================== 日志设置 ====================
    log_interval: int = 1      # 日志打印间隔（轮数）
    save_model: bool = True    # 是否保存最终模型
    save_plots: bool = True    # 是否生成可视化图表
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'NewFLConfig':
        """
        从YAML配置文件加载配置
        
        参数：
            yaml_path: YAML配置文件路径
            
        返回：
            NewFLConfig实例
        """
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        # 将配置字典转换为NewFLConfig实例
        return cls(**config_dict)
    
    def get_output_path(self, run_timestamp: str) -> str:
        """
        生成带时间戳的输出目录路径
        
        参数：
            run_timestamp: 运行时间戳，格式如 "20260429_103000"
            
        返回：
            完整的输出目录路径
        """
        # 目录名格式：experiment_name_timestamp
        name = f"{self.experiment_name}_{run_timestamp}"
        path = os.path.join(self.output_dir, name)
        # 创建目录（如果不存在）
        os.makedirs(path, exist_ok=True)
        return path
