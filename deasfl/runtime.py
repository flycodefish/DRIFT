"""New FL Runtime - Core orchestration for the custom FL framework"""

import numpy as np
import os
import time
import random
import math
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from fl_core.monitor import Monitor
from fl_core.data_loader import (
    SimpleModel, CNNModel, HARMLP,
    load_cifar10_data, load_cifar100_data, load_neu_data, load_har_data,
)

try:
    import networkx as nx
    import matplotlib.pyplot as plt
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import ray

from .config import NewFLConfig


def build_client_tree_mapping(clustering_result: Dict, num_clients: int) -> Dict[int, Dict]:
    """
    根据聚类结果构建每个节点的父子关系和子节点树映射

    参数：
        clustering_result: 聚类结果字典
        num_clients: 客户端数量

    返回：
        节点树映射字典 {node_id: {
            'parent': int or None,      # 父节点ID，簇首为None
            'children': List[int],       # 子节点ID列表
            'cluster_id': int,          # 所属簇ID（簇首ID）
            'total_clients': int,       # 该节点子树下的总客户端数
            'is_leaf': bool             # 是否为叶节点
        }}
    """
    tree_mapping = {}

    for client_id in range(num_clients):
        tree_mapping[client_id] = {
            'parent': None,
            'children': [],
            'cluster_id': None,
            'total_clients': 1,
            'sample_count': 0,
            'total_samples': 0,
            'is_leaf': True
        }

    if 'all_nodes_info' not in clustering_result:
        return tree_mapping

    all_info = clustering_result['all_nodes_info']

    for info in all_info:
        node_id = info['client_id']

        if info['parent'] is not None:
            tree_mapping[node_id]['parent'] = info['parent']
            if node_id not in tree_mapping[info['parent']]['children']:
                tree_mapping[info['parent']]['children'].append(node_id)
                tree_mapping[info['parent']]['is_leaf'] = False  # 父节点有子节点，不再是叶

        if info['state'] == 'cluster_head' or info['parent'] is None:
            tree_mapping[node_id]['cluster_id'] = node_id
        else:
            tree_mapping[node_id]['cluster_id'] = info.get('cluster_id', info['parent'])

    for cluster_head in clustering_result.get('cluster_heads', []):
        _calc_subtree_clients(tree_mapping, cluster_head)

    return tree_mapping


def _calc_subtree_clients(tree_mapping: Dict[int, Dict], node_id: int) -> int:
    """递归计算节点子树下的总客户端数"""
    total = 1
    for child_id in tree_mapping[node_id]['children']:
        total += _calc_subtree_clients(tree_mapping, child_id)
    tree_mapping[node_id]['total_clients'] = total
    return total


def _calc_subtree_samples(tree_mapping: Dict[int, Dict], node_id: int) -> int:
    """递归计算节点子树下的总样本数"""
    total = tree_mapping[node_id].get('sample_count', 0)
    for child_id in tree_mapping[node_id]['children']:
        total += _calc_subtree_samples(tree_mapping, child_id)
    tree_mapping[node_id]['total_samples'] = total
    return total


def set_tree_sample_counts(tree_mapping: Dict[int, Dict],
                           sample_counts: List[int],
                           cluster_heads: List[int]) -> None:
    """将每个客户端的样本数写入树映射，并递归计算子树样本数"""
    for node_id, count in enumerate(sample_counts):
        if node_id in tree_mapping:
            tree_mapping[node_id]['sample_count'] = count
            tree_mapping[node_id]['total_samples'] = count
    for head in cluster_heads:
        _calc_subtree_samples(tree_mapping, head)


# ==============================================================================
#                          通信成本统计
# ==============================================================================

class CommStats:
    """通信成本统计器

    跟踪分层聚合过程中所有模型传输的消息数和字节数。
    """
    def __init__(self):
        self.intra_cluster_msgs: int = 0       # 簇内树边上传（子→父）
        self.inter_cluster_forward_hops: int = 0  # 簇间转发跳数（簇首→最优节点）
        self.inter_cluster_return_hops: int = 0   # 簇间返回跳数（最优节点→簇首）
        self.intra_cluster_dist_msgs: int = 0    # 簇内树边分发（父→子）
        self.model_bytes: int = 0               # 单个模型大小（字节）
        self.flat_upload_msgs: int = 0           # 扁平聚合：上传消息数
        self.flat_download_msgs: int = 0         # 扁平聚合：下载消息数

    @property
    def total_msgs(self) -> int:
        return (self.intra_cluster_msgs + self.inter_cluster_forward_hops +
                self.inter_cluster_return_hops + self.intra_cluster_dist_msgs +
                self.flat_upload_msgs + self.flat_download_msgs)

    @property
    def total_bytes(self) -> int:
        return self.total_msgs * self.model_bytes

    def to_dict(self) -> dict:
        return {
            'intra_cluster_msgs': self.intra_cluster_msgs,
            'inter_cluster_forward_hops': self.inter_cluster_forward_hops,
            'inter_cluster_return_hops': self.inter_cluster_return_hops,
            'intra_cluster_dist_msgs': self.intra_cluster_dist_msgs,
            'flat_upload_msgs': self.flat_upload_msgs,
            'flat_download_msgs': self.flat_download_msgs,
            'total_msgs': self.total_msgs,
            'model_bytes': self.model_bytes,
            'total_bytes': self.total_bytes,
        }

    def __repr__(self):
        return (f"CommStats(total_msgs={self.total_msgs}, total_bytes={self.total_bytes:,}"
                f" | intra={self.intra_cluster_msgs}, fwd={self.inter_cluster_forward_hops},"
                f" ret={self.inter_cluster_return_hops}, dist={self.intra_cluster_dist_msgs})")


def compute_model_bytes(params: Dict[str, torch.Tensor]) -> int:
    """计算一个模型的 state_dict 的字节大小"""
    total = 0
    for tensor in params.values():
        total += tensor.numel() * tensor.element_size()
    return total


def weighted_validation_accuracy(eval_results: List[Dict]) -> float:
    """按各节点验证样本数汇总准确率，等价于在合并验证集上评估。"""
    total_samples = sum(result.get('num_samples', 0) for result in eval_results)
    if total_samples == 0:
        return float(np.mean([result['val_acc'] for result in eval_results]))
    return sum(result['val_acc'] * result['num_samples']
               for result in eval_results) / total_samples


def hierarchical_aggregate(models: List[nn.Module], tree_mapping: Dict[int, Dict],
                           cluster_heads: List[int], device: torch.device,
                           comm: Optional['CommStats'] = None,
                           skip_nodes: Optional[set] = None) -> Dict[int, Dict]:
    """
    执行分层聚合（支持跳过故障节点）

    流程：
    1. 叶节点直接聚合（只有自己）
    2. 有子节点的节点等待所有子节点聚合完成后，与自己的模型一起加权聚合
    3. 层层向上直到簇首

    参数：
        models: 所有客户端的模型列表，索引为client_id（故障节点位置为 None）
        tree_mapping: 节点树映射
        cluster_heads: 簇首ID列表
        device: 计算设备
        skip_nodes: 需要跳过的节点 ID 集合

    返回：
        聚合后的模型字典 {client_id: aggregated_model}
    """
    if skip_nodes is None:
        skip_nodes = set()

    aggregated_models = {}

    leaf_nodes = [node_id for node_id, info in tree_mapping.items()
                  if info['is_leaf'] and node_id not in skip_nodes]

    for leaf_id in leaf_nodes:
        if models[leaf_id] is not None:
            aggregated_models[leaf_id] = models[leaf_id].state_dict()

    def collect_and_aggregate(node_id: int):
        if node_id in aggregated_models:
            return aggregated_models[node_id]

        children = tree_mapping[node_id].get('children', [])
        # 过滤掉已故障的子节点
        live_children = [c for c in children if c not in skip_nodes]

        # 故障节点自身无模型，只聚合子节点
        node_in_skip = node_id in skip_nodes

        child_models = []
        child_weights = []

        for child_id in live_children:
            child_result = collect_and_aggregate(child_id)
            if child_result is not None:
                child_models.append(child_result)
                child_weights.append(
                    tree_mapping[child_id].get('total_samples') or
                    tree_mapping[child_id].get('total_clients', 1)
                )
                if comm is not None:
                    comm.intra_cluster_msgs += 1

        # 加入节点自身模型（故障节点跳过）
        if not node_in_skip and models[node_id] is not None:
            own_model_dict = models[node_id].state_dict()
            child_models.append(own_model_dict)
            child_weights.append(tree_mapping[node_id].get('sample_count') or 1)
        elif not child_models:
            # 故障节点且无存活子节点 → 返回 None
            aggregated_models[node_id] = None
            return None

        if not child_models:
            aggregated_models[node_id] = None
            return None

        total_weight = sum(child_weights)
        if total_weight == 0:
            aggregated_models[node_id] = None
            return None

        first_model = child_models[0]
        aggregated_dict = {}
        for key in first_model.keys():
            weighted_sum = torch.zeros_like(first_model[key], dtype=torch.float32).to(device)
            for i, model_dict in enumerate(child_models):
                weight = child_weights[i] / total_weight
                weighted_sum += model_dict[key].float().to(device) * weight
            aggregated_dict[key] = weighted_sum

        aggregated_models[node_id] = aggregated_dict
        return aggregated_dict

    for cluster_head in cluster_heads:
        if cluster_head not in skip_nodes:
            collect_and_aggregate(cluster_head)

    for node_id in tree_mapping.keys():
        if node_id not in aggregated_models:
            if node_id in skip_nodes or models[node_id] is None:
                aggregated_models[node_id] = None
            else:
                aggregated_models[node_id] = models[node_id].state_dict()

    return aggregated_models


def flat_aggregate(params_list: List[Dict[str, torch.Tensor]],
                   weights: Optional[List[float]] = None,
                   device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    """
    扁平加权平均聚合（消融实验：无聚类模式）

    所有客户端模型直接加权平均，不经过树状分层。

    参数：
        params_list: 各客户端的 state_dict 列表
        weights: 权重列表，默认等权
        device: 计算设备，默认 CPU

    返回：
        聚合后的 state_dict
    """
    if not params_list:
        raise ValueError("params_list 为空")
    if device is None:
        device = torch.device('cpu')

    n = len(params_list)
    if weights is None:
        weights = [1.0 / n] * n
    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    first = params_list[0]
    aggregated = {}
    for k, v in first.items():
        aggregated[k] = torch.zeros_like(v, device=device, dtype=torch.float32)

    for params, w in zip(params_list, weights):
        for k in aggregated:
            aggregated[k] += params[k].to(device).float() * w

    return aggregated


def distribute_models_to_cluster(aggregated_models: Dict[int, Dict],
                                tree_mapping: Dict[int, Dict],
                                models: List[nn.Module],
                                cluster_heads: List[int],
                                device: torch.device,
                                comm: Optional['CommStats'] = None):
    """
    将聚合后的模型按父子顺序分发到簇中的每个节点

    分发顺序：簇首 -> 子节点 -> 叶节点（深度优先遍历）

    参数：
        aggregated_models: 聚合后的模型参数字典
        tree_mapping: 节点树映射
        models: 模型列表（用于加载参数）
        cluster_heads: 簇首ID列表
        device: 计算设备
    """
    def distribute_recursive(node_id: int, model_state_dict: Dict):
        models[node_id].load_state_dict(model_state_dict)

        for child_id in sorted(tree_mapping[node_id]['children']):
            child_cluster_id = tree_mapping[child_id]['cluster_id']
            if child_cluster_id == tree_mapping[node_id]['cluster_id']:
                # 最终全局模型沿树广播给每个节点。原实现错误地重新加载
                # 子节点的旧子树模型，使普通节点没有收到全局模型。
                if comm is not None:
                    comm.intra_cluster_dist_msgs += 1
                distribute_recursive(child_id, model_state_dict)

    for cluster_head in cluster_heads:
        if cluster_head in aggregated_models:
            distribute_recursive(cluster_head, aggregated_models[cluster_head])


# ==============================================================================
#                          簇间聚合（最佳客户端）
# ==============================================================================

from typing import Dict, List, Tuple
from copy import deepcopy


@dataclass
class ModelForwardMessage:
    """模型转发消息"""
    source_id: int                  # 源节点（簇首）
    model_dict: Dict                # 模型参数
    client_count: int               # 该模型代表的客户端数量
    path_stack: List[int]           # 转发路径栈（用于原路返回）


def build_cluster_info(tree_mapping: Dict[int, Dict], 
                      clustering_result: Dict,
                      election_result: Dict) -> Tuple[Dict[int, int], Dict[int, Dict]]:
    """
    构建簇信息和路由表
    
    Args:
        tree_mapping: 节点树映射
        clustering_result: 聚类结果
        election_result: 选举结果
    
    Returns:
        cluster_sizes: {cluster_head_id: client_count}
        routing_tables: {node_id: {'next_hop': int, 'is_best_client': bool}}
    """
    cluster_sizes = {}
    routing_tables = {}
    
    cluster_heads = clustering_result.get('cluster_heads', [])
    all_nodes_info_election = election_result.get('all_nodes_info', [])
    
    for head_id in cluster_heads:
        cluster_sizes[head_id] = tree_mapping[head_id].get('total_samples') or tree_mapping[head_id]['total_clients']
    
    for node_id in range(len(all_nodes_info_election)):
        info = all_nodes_info_election[node_id]
        routing_tables[node_id] = {
            'next_hop': info.get('next_hop', -1),
            'is_best_client': info.get('is_best_client', False)
        }
    
    return cluster_sizes, routing_tables


def forward_model_to_best_client(source_cluster_head: int,
                                 model_dict: Dict,
                                 client_count: int,
                                 routing_tables: Dict[int, Dict],
                                 best_client_id: int,
                                 message_queue: Dict[int, List],
                                 comm: Optional['CommStats'] = None) -> None:
    """
    簇首将模型沿路由转发到最佳客户端

    Args:
        source_cluster_head: 源簇首ID
        model_dict: 模型参数
        client_count: 该模型代表的客户端数量
        routing_tables: 路由表
        best_client_id: 最佳客户端ID
        message_queue: 消息队列，用于模拟转发过程
    """
    current_node = source_cluster_head
    path_stack = [source_cluster_head]
    visited = {source_cluster_head}
    max_hops = len(routing_tables) + 1  # 安全上限：节点数+1

    while current_node != best_client_id:
        if current_node not in routing_tables:
            print(f"  ⚠ 路由表缺少节点 {current_node}，停止转发（源簇首={source_cluster_head}）")
            break

        next_hop = routing_tables[current_node].get('next_hop', -1)

        if next_hop == -1 or next_hop == current_node:
            print(f"  ⚠ 节点 {current_node} 的 next_hop={next_hop}，停止转发（源簇首={source_cluster_head}）")
            break

        if next_hop in visited:
            print(f"  ⚠ 检测到路由环路: {next_hop} 已访问，停止转发（源簇首={source_cluster_head}，路径={path_stack}）")
            break

        if len(path_stack) > max_hops:
            print(f"  ⚠ 路径超过最大跳数 {max_hops}，停止转发（源簇首={source_cluster_head}）")
            break

        path_stack.append(next_hop)
        visited.add(next_hop)
        current_node = next_hop

    if path_stack[-1] != best_client_id:
        print(f"  ⚠ 簇首 {source_cluster_head} 未能到达最佳客户端 {best_client_id}，"
              f"路径终止于 {path_stack[-1]}（hops={len(path_stack)-1}）")

    # 统计：路径上除源节点外的每一跳都是一次模型转发
    hops = len(path_stack) - 1
    if comm is not None and hops > 0:
        comm.inter_cluster_forward_hops += hops
        if comm.model_bytes == 0:
            comm.model_bytes = compute_model_bytes(model_dict)

    msg = ModelForwardMessage(
        source_id=source_cluster_head,
        model_dict=deepcopy(model_dict),
        client_count=client_count,
        path_stack=path_stack
    )

    if best_client_id not in message_queue:
        message_queue[best_client_id] = []
    message_queue[best_client_id].append(msg)


def aggregate_models_at_best_client(best_client_id: int,
                                   models: List,
                                   message_queue: Dict[int, List],
                                   device: torch.device) -> Tuple[Dict, Dict[int, List[int]]]:
    """
    在最佳客户端处聚合所有簇首上传的模型
    
    Args:
        best_client_id: 最佳客户端ID
        models: 所有模型列表
        message_queue: 消息队列（包含各簇首的模型）
        device: 计算设备
    
    Returns:
        final_aggregated_model: 最终聚合后的模型
        path_stacks: {source_cluster_head: path_stack}，用于原路返回
    """
    messages = message_queue.get(best_client_id, [])
    
    if not messages:
        return models[best_client_id].state_dict(), {}
    
    client_counts = []
    model_dicts = []
    path_stacks = {}
    
    for msg in messages:
        client_counts.append(msg.client_count)
        model_dicts.append(msg.model_dict)
        path_stacks[msg.source_id] = msg.path_stack
    
    total_clients = sum(client_counts)
    
    final_model = {}
    for key in model_dicts[0].keys():
        weighted_sum = torch.zeros_like(model_dicts[0][key], dtype=torch.float32).to(device)
        for i, model_dict in enumerate(model_dicts):
            weight = client_counts[i] / total_clients
            weighted_sum += model_dict[key].float() * weight
        final_model[key] = weighted_sum
    
    return final_model, path_stacks


def return_model_to_cluster_heads(final_model: Dict,
                                  path_stacks: Dict[int, List[int]],
                                  return_queue: Dict[int, List],
                                  comm: Optional['CommStats'] = None) -> None:
    """
    将最终聚合后的模型沿原路返回到各簇首
    
    Args:
        final_model: 最终聚合后的模型
        path_stacks: {source_cluster_head: path_stack}
        return_queue: 返回消息队列
    """
    for source_head, path_stack in path_stacks.items():
        reverse_path = path_stack[::-1]

        # 统计：返回路径上的每一跳（不含目标节点自己）
        hops = len(reverse_path) - 1
        if comm is not None and hops > 0:
            comm.inter_cluster_return_hops += hops

        return_msg = {
            'target_cluster_head': source_head,
            'model_dict': deepcopy(final_model),
            'path_stack': reverse_path
        }
        
        if source_head not in return_queue:
            return_queue[source_head] = []
        return_queue[source_head].append(return_msg)


class ModelWrapper:
    """轻量包装，让 param dict 兼容 .state_dict() 和 .load_state_dict() 调用"""
    def __init__(self, state_dict):
        self._state_dict = state_dict
    def state_dict(self):
        return self._state_dict
    def load_state_dict(self, state_dict, strict=True):
        self._state_dict = state_dict


class NewFLRuntime:
    """
    NewFL运行时类，负责协调联邦学习的各个阶段
    """
    
    def __init__(self, config: NewFLConfig):
        """
        初始化运行时
        
        参数：
            config: 配置对象
        """
        self.config = config
        self.topology = None
        self.clients = None
        self.election_results = None
        self.clustering_results = None
        self.tree_mapping = None

        # 训练相关属性
        self.models = None
        self.optimizers = None
        self.criterions = None
        self.train_loaders = None
        self.val_loaders = None
        self.device = None

        # 动态节点管理（Node Failure / Join）
        self.original_num_clients = self.config.num_clients
        self.initial_active = (self.config.initial_active_clients
                               if self.config.initial_active_clients > 0
                               else self.config.num_clients)
        self.total_nodes = self.config.num_clients
        self.active_nodes: set = set(range(self.initial_active))
        self.failed_nodes: set = set()
        self.joined_nodes: set = set()
        self.failure_log: list = []    # 每轮的故障记录
        self.join_log: list = []       # 每轮的加入记录
        
    def start(self):
        """启动运行时"""
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if getattr(self.config, 'use_ray_actors', True):
            ray.init(ignore_reinit_error=True, logging_level="ERROR")
        print(f"NewFL Runtime started with seed: {self.config.seed}")

    def shutdown(self):
        """关闭运行时"""
        if hasattr(self, 'client_actors') and self.client_actors:
            for actor in self.client_actors:
                if actor is not None:
                    try:
                        ray.kill(actor)
                    except Exception:
                        pass
            self.client_actors = []
        ray.shutdown()
        print("NewFL Runtime shutdown complete")
    
    def init_clients(self, train_data, val_data, indices):
        """初始化客户端"""
        self.clients = []
        for i in range(self.config.num_clients):
            client_info = {
                'id': i,
                'train_data': train_data[i],
                'val_data': val_data[i],
                'indices': indices[i]
            }
            self.clients.append(client_info)
        self.client_sample_counts = [len(train_data[i]['labels']) for i in range(self.config.num_clients)]
        print(f"Initialized {len(self.clients)} clients")
    
    def build_topology(self):
        """构建网络拓扑"""
        self.topology = self._build_topology(
            self.config.num_clients,
            self.config.topology_type,
            self.config.neighbor_selection,
            self.config.min_neighbors,
            self.config.max_neighbors
        )
        print(f"Topology built: {self.config.topology_type}")
        return self.topology
    
    def _build_topology(self, n_clients: int, topology_type: str, 
                       selection_strategy: str, min_neighbors: int, max_neighbors: int):
        """
        根据类型构建拓扑
        
        参数：
            n_clients: 客户端数量
            topology_type: 拓扑类型 (mesh, ring, star, fully_connected)
            selection_strategy: 邻居选择策略
            min_neighbors: 最小邻居数
            max_neighbors: 最大邻居数
        
        返回：
            拓扑结构列表
        """
        if topology_type == 'ring':
            return self._build_ring_topology(n_clients)
        elif topology_type == 'star':
            return self._build_star_topology(n_clients)
        elif topology_type == 'fully_connected':
            return self._build_fully_connected_topology(n_clients)
        elif topology_type == 'mesh':
            return self._build_mesh_topology(n_clients, min_neighbors, max_neighbors, selection_strategy)
        else:
            raise ValueError(f"Unsupported topology type: {topology_type}")
    
    def _build_mesh_topology(self, n_clients: int, min_neighbors: int, max_neighbors: int, selection_strategy: str):
        """
        构建Mesh拓扑，每个节点随机连接2-4个邻居（可配置范围）

        参数：
            n_clients: 客户端数量
            min_neighbors: 每个节点的最小邻居数量
            max_neighbors: 每个节点的最大邻居数量
            selection_strategy: 邻居选择策略

        返回：
            邻居列表
        """
        topology = [[] for _ in range(n_clients)]
        
        if selection_strategy == "static":
            # 静态策略：按顺序选择邻居
            for i in range(n_clients):
                num_nbrs = min(np.random.randint(min_neighbors, max_neighbors + 1), n_clients - 1)
                neighbors = []
                offset = 1
                while len(neighbors) < num_nbrs:
                    candidate = (i + offset) % n_clients
                    if candidate != i and candidate not in neighbors:
                        neighbors.append(candidate)
                    offset += 1
                topology[i] = neighbors
        else:
            # 随机策略：随机选择邻居，确保每个节点邻居数在[min, max]范围内
            for i in range(n_clients):
                # 计算该节点还需要多少邻居（不超过max）
                current_count = len(topology[i])
                if current_count >= max_neighbors:
                    continue
                
                # 每个节点随机选择2-4个邻居（考虑已有邻居数）
                num_nbrs = min(np.random.randint(min_neighbors, max_neighbors + 1), n_clients - 1)
                num_nbrs = max(num_nbrs - current_count, 0)  # 不超过需要的数量
                
                if num_nbrs <= 0:
                    continue
                
                # 排除自己、已有的邻居，以及已满的节点
                available = [j for j in range(n_clients) if 
                            j != i and 
                            j not in topology[i] and 
                            len(topology[j]) < max_neighbors]
                
                # 随机选择邻居
                if len(available) >= num_nbrs:
                    new_neighbors = list(np.random.choice(available, num_nbrs, replace=False))
                else:
                    new_neighbors = available
                
                topology[i].extend(new_neighbors)
                
                # 确保双向连接（无向图），但只在邻居节点还有容量时
                for neighbor in new_neighbors:
                    if i not in topology[neighbor] and len(topology[neighbor]) < max_neighbors:
                        topology[neighbor].append(i)
        
        # 最后确保每个节点至少有min_neighbors个邻居（同时不超过max）
        for i in range(n_clients):
            while len(topology[i]) < min_neighbors:
                # 只选择还有容量的节点作为候选
                available = [j for j in range(n_clients) if 
                            j != i and 
                            j not in topology[i] and 
                            len(topology[j]) < max_neighbors]
                if available:
                    neighbor = np.random.choice(available)
                    topology[i].append(neighbor)
                    topology[neighbor].append(i)
                else:
                    # 没有可用的邻居了，尝试找已满但还没连接的节点（放宽限制）
                    available_full = [j for j in range(n_clients) if 
                                    j != i and 
                                    j not in topology[i]]
                    if available_full:
                        neighbor = np.random.choice(available_full)
                        topology[i].append(neighbor)
                        if i not in topology[neighbor]:
                            topology[neighbor].append(i)
                    break
        
        # 最后检查并确保所有节点的邻居数不超过max_neighbors
        for i in range(n_clients):
            if len(topology[i]) > max_neighbors:
                # 如果超过，随机截断到max_neighbors个
                topology[i] = list(np.random.choice(topology[i], max_neighbors, replace=False))
        
        return topology
    
    def _build_ring_topology(self, n_clients: int):
        """
        构建环形拓扑，每个节点只连接左右邻居
        
        返回：
            邻居列表
        """
        topology = []
        for i in range(n_clients):
            left = (i - 1) % n_clients
            right = (i + 1) % n_clients
            topology.append([left, right])
        return topology
    
    def _build_star_topology(self, n_clients: int):
        """
        构建星形拓扑，所有节点连接到中心节点（节点0）
        
        返回：
            邻居列表
        """
        topology = []
        for i in range(n_clients):
            if i == 0:
                # 中心节点连接所有其他节点
                topology.append(list(range(1, n_clients)))
            else:
                # 其他节点只连接中心节点
                topology.append([0])
        return topology
    
    def _build_fully_connected_topology(self, n_clients: int):
        """
        构建全连接拓扑，每个节点连接所有其他节点
        
        返回：
            邻居列表
        """
        topology = []
        for i in range(n_clients):
            neighbors = [j for j in range(n_clients) if j != i]
            topology.append(neighbors)
        return topology
    
    def init_election(self, base_trigger_time: float, trigger_factor: float):
        """初始化选举"""
        print(f"Initializing election with base_trigger_time={base_trigger_time}, trigger_factor={trigger_factor}")
    
    def run_election(self):
        """运行选举算法"""
        from .election import ElectionManager
        
        # 创建虚拟客户端列表（仅包含活跃节点）
        clients = [{'id': i} for i in sorted(self.active_nodes)]

        # 过滤拓扑：只保留活跃节点之间的连接
        active_set = self.active_nodes
        filtered_topology = []
        for i in range(self.config.num_clients):
            if i in active_set:
                filtered_topology.append([n for n in self.topology[i] if n in active_set])
            else:
                filtered_topology.append([])

        # 初始化选举管理器
        election_manager = ElectionManager(
            clients=clients,
            topology=filtered_topology,
            base_trigger_time=self.config.election_base_trigger_time,
            trigger_factor=self.config.election_trigger_factor
        )
        
        # 运行选举
        self.election_results = election_manager.start_election()
        return self.election_results
    
    def init_clustering(self, election_results: Dict, max_child: int, cluster_offset: float):
        """初始化聚类"""
        print(f"Initializing clustering with max_child={max_child}, cluster_offset={cluster_offset}")
    
    def run_clustering(self):
        """运行聚类算法"""
        from .election import ClusteringManager

        # 创建虚拟客户端列表（仅包含活跃节点）
        clients = [{'id': i} for i in sorted(self.active_nodes)]

        # 过滤拓扑：只保留活跃节点之间的连接
        active_set = self.active_nodes
        filtered_topology = []
        for i in range(self.config.num_clients):
            if i in active_set:
                filtered_topology.append([n for n in self.topology[i] if n in active_set])
            else:
                filtered_topology.append([])

        # 使用存储的标签分布（如果有的话）
        label_dists = getattr(self, 'label_distributions', None)
        dist_threshold = getattr(self.config, 'distribution_similarity_threshold', 0.0)

        # 初始化聚类管理器
        clustering_manager = ClusteringManager(
            clients=clients,
            topology=filtered_topology,
            election_results=self.election_results,
            max_child=self.config.cluster_max_child,
            max_retry=self.config.cluster_max_retry,
            request_timeout=self.config.cluster_request_timeout,
            cluster_offset=self.config.cluster_offset,
            label_distributions=label_dists,
            distribution_similarity_threshold=dist_threshold
        )
        
        # 运行聚类
        self.clustering_results = clustering_manager.run_clustering()
        self.tree_mapping = build_client_tree_mapping(self.clustering_results, self.config.num_clients)
        return self.clustering_results

    def set_clustering_results(self, clustering_results: Dict, election_results: Dict = None):
        """
        设置聚类结果并构建树映射

        参数：
            clustering_results: 聚类结果字典
            election_results: 选举结果字典（包含最佳客户端信息）
        """
        self.clustering_results = clustering_results
        self.tree_mapping = build_client_tree_mapping(clustering_results, self.config.num_clients)

        # 分层权重必须代表真实样本数；只按节点数量会破坏与 FedAvg 的等价性。
        if hasattr(self, 'client_sample_counts'):
            set_tree_sample_counts(
                self.tree_mapping,
                self.client_sample_counts,
                clustering_results.get('cluster_heads', []),
            )

        if election_results is not None:
            self.election_results = election_results

        print(f"\n树结构信息:")
        print(f"  簇首数量: {len(self.clustering_results['cluster_heads'])}")
        for head in self.clustering_results['cluster_heads']:
            print(f"  簇 {head}: {self.tree_mapping[head]['total_clients']} 个客户端")

    def init_training(self, train_data, val_data, use_cnn=False, input_dim=10, num_classes=10,
                     global_test_data=None):
        """
        初始化训练环境

        参数：
            train_data: 训练数据列表，每个客户端一个字典 {'images': tensor, 'labels': tensor}
            val_data: 验证数据列表
            use_cnn: 是否使用CNN模型（用于CIFAR-10图像数据）
            input_dim: 输入特征维度（仅用于SimpleModel）
            num_classes: 类别数
            global_test_data: 全局测试数据字典 {'images': tensor, 'labels': tensor}，用于评估聚合后的准确率
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Cannot initialize training.")

        use_cuda = self.config.use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        print(f"Training device: {self.device}")

        self.models = []
        self.optimizers = []
        self.criterions = []
        self.train_loaders = []
        self.val_loaders = []

        base_model = (CNNModel(num_classes=num_classes) if use_cnn
                      else SimpleModel(input_dim=input_dim, num_classes=num_classes))
        base_state = {key: value.detach().clone()
                      for key, value in base_model.state_dict().items()}

        for client_id in range(self.config.num_clients):
            if use_cnn:
                model = CNNModel(num_classes=num_classes).to(self.device)
            else:
                model = SimpleModel(input_dim=input_dim, num_classes=num_classes).to(self.device)
            model.load_state_dict(base_state)

            self.models.append(model)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            self.optimizers.append(optimizer)

            criterion = nn.CrossEntropyLoss()
            self.criterions.append(criterion)

            train_dataset = TensorDataset(train_data[client_id]['images'], train_data[client_id]['labels'])
            val_dataset = TensorDataset(val_data[client_id]['images'], val_data[client_id]['labels'])

            generator = torch.Generator().manual_seed(self.config.seed + client_id)
            self.train_loaders.append(DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                generator=generator,
            ))
            self.val_loaders.append(DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False))

        if global_test_data is not None:
            global_test_dataset = TensorDataset(global_test_data['images'], global_test_data['labels'])
            self.global_test_loader = DataLoader(global_test_dataset, batch_size=self.config.batch_size, shuffle=False)
            print(f"Global test dataset loaded with {len(global_test_dataset)} samples")
        else:
            self.global_test_loader = None

        model_type = "CNN" if use_cnn else "MLP"
        print(f"Initialized {model_type} training for {self.config.num_clients} clients")

    def _create_model(self):
        """根据 config.model 创建模型"""
        model_name = getattr(self.config, 'model', 'cnn')
        if model_name == 'resnet18':
            from fl_core.actors import ResNet18 as RN18
            return RN18(self.config.num_classes)
        elif model_name in ('cnn', 'cnnmodel'):
            return CNNModel(num_classes=self.config.num_classes)
        elif model_name == 'harmlp':
            return HARMLP(num_classes=self.config.num_classes)
        else:
            return SimpleModel(input_dim=10, num_classes=self.config.num_classes)

    def _calc_learning_rate(self, round_id: int) -> float:
        """Cosine Annealing + Warmup 学习率计算"""
        initial_lr = float(self.config.learning_rate)
        min_lr = float(self.config.learning_rate_min)
        warmup = int(self.config.warmup_rounds)
        max_rounds = int(self.config.rounds)

        if self.config.cosine_annealing:
            if round_id < warmup:
                return initial_lr * ((round_id + 1) / max(1, warmup))
            else:
                progress = round_id - warmup
                total_steps = max_rounds - warmup
                return min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * progress / total_steps))
        else:
            return max(initial_lr * (0.95 ** round_id), min_lr)

    def _init_ray_actors(self, train_data, val_data):
        """创建 NewClientActor 的 Ray 实例，平分到多 GPU 上"""
        from .actors import NewClientActor

        num_gpus = torch.cuda.device_count()
        self.device = torch.device("cuda:0" if num_gpus > 0 else "cpu")

        # 所有客户端共享同一个初始模型（对齐 FedAvg）
        base_model = self._create_model()
        base_state = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}

        self.client_actors = [None] * self.config.num_clients
        actor_gpu = (self.config.num_gpus
                     if self.config.use_gpu and torch.cuda.is_available()
                     else 0.0)
        for client_id in range(self.config.num_clients):
            if client_id not in self.active_nodes:
                continue
            model = self._create_model()
            model.load_state_dict(base_state)
            train_ds = TensorDataset(
                train_data[client_id]['images'],
                train_data[client_id]['labels']
            )
            val_ds = TensorDataset(
                val_data[client_id]['images'],
                val_data[client_id]['labels']
            )
            neighbors = list(self.topology[client_id]) if self.topology else []

            use_img_xf = self.config.dataset not in ('har',)
            actor = NewClientActor.options(num_gpus=actor_gpu).remote(
                client_id=client_id,
                model=model,
                train_dataset=train_ds,
                val_dataset=val_ds,
                batch_size=self.config.batch_size,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                device=None,
                neighbors=neighbors,
                use_image_transform=use_img_xf,
                seed=self.config.seed + client_id,
            )
            actor.set_local_epochs.remote(self.config.local_epochs)
            self.client_actors[client_id] = actor

        # 存储每个客户端的样本数（用于样本加权聚合）
        self.client_sample_counts = [len(train_data[i]['labels'])
                                     for i in range(self.config.num_clients)]

        # 保存训练数据引用（供后续 join 时创建新 actor 使用）
        self._reserve_train_data = train_data
        self._reserve_val_data = val_data

        active_count = sum(1 for a in self.client_actors if a is not None)
        print(f"Initialized {active_count} Ray client actors "
              f"across {num_gpus} GPU(s) (explicit round-robin assignment)")

    def _train_single_client(self, client_id: int, round_id: int) -> float:
        """
        训练单个客户端
        
        参数：
            client_id: 客户端ID
            round_id: 当前轮次
            
        返回：
            训练损失
        """
        model = self.models[client_id]
        # 与 Ray/FedAvg 路径一致：每个联邦轮次重建本地 Adam 状态。
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self._calc_learning_rate(round_id),
            weight_decay=self.config.weight_decay,
        )
        criterion = self.criterions[client_id]
        train_loader = self.train_loaders[client_id]
        
        model.train()
        total_loss = 0.0
        total_samples = 0
        
        for epoch in range(self.config.local_epochs):
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                
                optimizer.zero_grad()
                output = model(xb)
                loss = criterion(output, yb)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * xb.size(0)
                total_samples += xb.size(0)
        
        return total_loss / total_samples
    
    def _evaluate_single_client(self, client_id: int) -> tuple:
        """
        评估单个客户端
        
        参数：
            client_id: 客户端ID
            
        返回：
            (验证损失, 准确率)
        """
        model = self.models[client_id]
        criterion = self.criterions[client_id]
        val_loader = self.val_loaders[client_id]
        
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                output = model(xb)
                loss = criterion(output, yb)
                
                total_loss += loss.item() * xb.size(0)
                _, predicted = torch.max(output.data, 1)
                total += yb.size(0)
                correct += (predicted == yb).sum().item()
        
        return total_loss / total, correct / total

    def _evaluate_on_global_data(self, model: nn.Module) -> tuple:
        """
        在全局测试数据上评估模型

        参数：
            model: 要评估的模型

        返回：
            (验证损失, 准确率)
        """
        if self.global_test_loader is None:
            return 0.0, 0.0

        criterion = nn.CrossEntropyLoss()
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for xb, yb in self.global_test_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                output = model(xb)
                loss = criterion(output, yb)

                total_loss += loss.item() * xb.size(0)
                _, predicted = torch.max(output.data, 1)
                total += yb.size(0)
                correct += (predicted == yb).sum().item()

        return total_loss / total if total > 0 else 0.0, correct / total if total > 0 else 0.0

    # ==========================================================================
    #                    动态节点管理（Node Failure / Join）
    # ==========================================================================

    def _simulate_failures(self, round_id: int) -> list:
        """模拟节点故障，返回本轮新故障的节点 ID 列表"""
        cfg = self.config
        newly_failed = []

        if not cfg.enable_dynamic_nodes:
            return newly_failed
        if round_id < cfg.failure_start_round:
            return newly_failed

        remaining_active = self.active_nodes - self.failed_nodes
        if len(remaining_active) <= 1:
            return newly_failed

        max_additional = max(0, cfg.failure_max_nodes - len(self.failed_nodes))
        if max_additional <= 0:
            return newly_failed

        rng = random.Random(cfg.seed * 1000 + round_id)

        if cfg.failure_pattern == "leaf_only" and self.tree_mapping:
            candidates = [n for n in remaining_active
                          if self.tree_mapping.get(n, {}).get('is_leaf', True)]
        elif cfg.failure_pattern == "cluster_head" and self.tree_mapping:
            candidates = [n for n in remaining_active
                          if self.tree_mapping.get(n, {}).get('parent') is None]
        elif cfg.failure_pattern == "best_client":
            best_id = self.election_results.get('best_client_id', -1) if self.election_results else -1
            candidates = [best_id] if best_id in remaining_active else []
        else:  # "random"
            candidates = list(remaining_active)

        if not candidates:
            return newly_failed

        for node_id in candidates:
            if len(newly_failed) >= max_additional:
                break
            if rng.random() < cfg.failure_rate:
                newly_failed.append(node_id)

        for nid in newly_failed:
            self.failed_nodes.add(nid)

        if newly_failed:
            self.failure_log.append({'round': round_id, 'failed_nodes': newly_failed.copy()})
            self._repair_tree_after_failures(newly_failed)
            self._kill_failed_actors(newly_failed)

        return newly_failed

    def _activate_reserve_nodes(self, round_id: int) -> list:
        """激活预留节点（增量加入实验）

        在 join_start_round 触发，根据 join_strategy 执行不同策略：
        - static: 不做任何事
        - global_rebuild: 创建新 Ray actor + 全局重新选举聚类
        - localized: 创建新 Ray actor + 局部 JSD 匹配接入
        """
        cfg = self.config
        reserve = sorted(set(range(cfg.num_clients)) - self.active_nodes)
        if not reserve:
            return []

        # 限制加入数量
        max_join = cfg.join_max_nodes if cfg.join_max_nodes > 0 else len(reserve)
        joining = reserve[:max_join]

        print(f"\n{'='*60}")
        print(f"[加入] 第 {round_id} 轮: 激活 {len(joining)} 个新节点 {joining}")
        print(f"  加入策略: {cfg.join_strategy}")
        print(f"{'='*60}")

        if cfg.join_strategy == 'static':
            # 不做任何事，新节点不参与训练
            self.join_log.append({'round': round_id, 'joined_nodes': [],
                                  'strategy': 'static'})
            return []

        # 获取当前最佳模型权重（用于初始化新节点）
        base_state = None
        if hasattr(self, 'client_actors') and self.client_actors:
            live_actors, live_ids = self._get_live_ray_actors()
            if live_actors:
                try:
                    base_state = ray.get(live_actors[0].get_model_params.remote())
                except Exception:
                    pass

        # 为新节点创建 Ray actor
        from .actors import NewClientActor
        train_data = getattr(self, '_reserve_train_data', None)
        val_data = getattr(self, '_reserve_val_data', None)

        for nid in joining:
            self.active_nodes.add(nid)
            self.joined_nodes.add(nid)

            model = self._create_model()
            if base_state:
                model.load_state_dict({k: v.cpu().clone() for k, v in base_state.items()})

            if train_data and nid < len(train_data):
                train_ds = TensorDataset(train_data[nid]['images'], train_data[nid]['labels'])
                val_ds = TensorDataset(val_data[nid]['images'], val_data[nid]['labels'])
            else:
                print(f"  WARNING: 节点 {nid} 缺少训练数据，跳过 actor 创建")
                continue

            neighbors = list(self.topology[nid]) if self.topology else []
            use_img_xf = cfg.dataset not in ('har',)
            actor = NewClientActor.remote(
                client_id=nid, model=model,
                train_dataset=train_ds, val_dataset=val_ds,
                batch_size=cfg.batch_size, lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay, device=None,
                neighbors=neighbors, use_image_transform=use_img_xf)
            actor.set_local_epochs.remote(cfg.local_epochs)
            self.client_actors[nid] = actor

        # 根据策略执行拓扑集成
        if cfg.join_strategy == 'global_rebuild':
            print(f"\n  [Global Rebuild] 执行全局重新选举+聚类...")
            self._reelect_and_recluster(reason='join')

        elif cfg.join_strategy == 'localized':
            print(f"\n  [Localized Join] 执行局部 JSD 匹配接入...")
            self._localized_join(joining)

        self.join_log.append({'round': round_id, 'joined_nodes': joining,
                              'strategy': cfg.join_strategy,
                              'total_active': len(self.active_nodes)})
        print(f"  加入完成，当前活跃节点: {len(self.active_nodes)}/{self.total_nodes}")
        return joining

    def _localized_join(self, new_node_ids: list):
        """局部 JSD 匹配接入（论文 IV-C-3 的 Node Join 机制）

        新节点计算与活跃邻居的 JSD，选择满足阈值且资源最高的邻居作为父节点，
        以叶子身份挂载。仅局部邻域参与，无需全局重构。
        """
        label_dists = getattr(self, 'label_distributions', None)
        tau = getattr(self.config, 'distribution_similarity_threshold', 0.0)

        for nid in new_node_ids:
            print(f"\n  [JOIN] 节点 {nid} 正在接入...")
            neighbors = [n for n in self.topology[nid]
                        if n in self.active_nodes and n not in self.failed_nodes]

            if not neighbors:
                print(f"    无活跃邻居，自成一簇")
                self._make_cluster_head(nid)
                continue

            # 计算与每个邻居（或其簇首）的 JSD
            if label_dists and nid < len(label_dists):
                p_new = label_dists[nid]
                candidates = []
                for nb in neighbors:
                    # 向上找簇首
                    head = self._find_cluster_head(nb)
                    if head is not None and head < len(label_dists):
                        jsd_val = self._jsd(p_new, label_dists[head])
                        if jsd_val <= tau or tau == 0.0:
                            # 资源分 = 邻居自己的 resource score（得分越高越好）
                            score = self._get_node_resource_score(head)
                            candidates.append((head, jsd_val, score))

                if candidates:
                    # 选 JSD 最小且资源得分最高的
                    candidates.sort(key=lambda x: (x[1], -x[2]))
                    best_head, best_jsd, best_score = candidates[0]
                    self._attach_as_leaf(nid, best_head)
                    print(f"    接入簇首 {best_head} (JSD={best_jsd:.4f}, score={best_score:.1f})")
                else:
                    print(f"    无满足 JSD≤{tau} 的邻居，自成一簇")
                    self._make_cluster_head(nid)
            else:
                # 无标签分布，挂载到资源最高的邻居
                best_nb = max(neighbors, key=lambda n: self._get_node_resource_score(n))
                head = self._find_cluster_head(best_nb)
                if head is not None:
                    self._attach_as_leaf(nid, head)
                    print(f"    接入簇首 {head} (无 JSD 信息，按资源选择)")
                else:
                    self._make_cluster_head(nid)

        # 更新 cluster_heads 列表
        self._update_cluster_heads_from_tree()
        print(f"\n  局部接入完成，当前簇首: {self.clustering_results.get('cluster_heads', [])}")

    def _find_cluster_head(self, node_id: int):
        """向上遍历树找到 node_id 所属的簇首"""
        if not self.tree_mapping or node_id not in self.tree_mapping:
            return node_id
        info = self.tree_mapping[node_id]
        if info.get('parent') is None:
            return node_id
        # 简单实现：tree_mapping 里 cluster_id 就是根
        cid = info.get('cluster_id')
        if cid is not None and cid != node_id:
            return cid
        return node_id

    def _attach_as_leaf(self, node_id: int, parent_id: int):
        """将 node_id 作为叶子挂载到 parent_id 下"""
        if not self.tree_mapping:
            self._make_cluster_head(node_id)
            return

        # 确保父节点不是叶子
        if parent_id in self.tree_mapping:
            self.tree_mapping[parent_id]['children'].append(node_id)
            self.tree_mapping[parent_id]['is_leaf'] = False

        # 设置新节点信息
        self.tree_mapping[node_id] = {
            'children': [],
            'parent': parent_id,
            'is_leaf': True,
            'cluster_id': self.tree_mapping[parent_id].get('cluster_id', parent_id),
            'total_clients': 1,
            'total_samples': (self.client_sample_counts[node_id]
                              if node_id < len(self.client_sample_counts) else 0),
        }

    def _make_cluster_head(self, node_id: int):
        """将 node_id 设置为独立簇首"""
        if not self.tree_mapping:
            return
        self.tree_mapping[node_id] = {
            'children': [],
            'parent': None,
            'is_leaf': True,
            'cluster_id': node_id,
            'total_clients': 1,
            'total_samples': (self.client_sample_counts[node_id]
                              if node_id < len(self.client_sample_counts) else 0),
        }
        if self.clustering_results:
            if 'cluster_heads' not in self.clustering_results:
                self.clustering_results['cluster_heads'] = []
            if node_id not in self.clustering_results['cluster_heads']:
                self.clustering_results['cluster_heads'].append(node_id)

    def _get_node_resource_score(self, node_id: int) -> float:
        """获取节点的资源得分"""
        if hasattr(self, 'election_results') and self.election_results:
            all_info = self.election_results.get('all_nodes_info', [])
            if node_id < len(all_info):
                return all_info[node_id].get('my_score', 0.0)
        return 0.0

    @staticmethod
    def _jsd(p, q):
        """Jensen-Shannon Divergence"""
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        p = p / (p.sum() + 1e-12)
        q = q / (q.sum() + 1e-12)
        m = (p + q) / 2
        kl_pm = np.sum(p * np.log((p + 1e-12) / (m + 1e-12)))
        kl_qm = np.sum(q * np.log((q + 1e-12) / (m + 1e-12)))
        return float((kl_pm + kl_qm) / 2)

    def _kill_failed_actors(self, failed_ids: list):
        """Kill 故障节点的 Ray Actor 以释放 GPU 显存和系统内存"""
        if not hasattr(self, 'client_actors') or not self.client_actors:
            return
        for nid in failed_ids:
            if nid < len(self.client_actors) and self.client_actors[nid] is not None:
                try:
                    ray.kill(self.client_actors[nid])
                except Exception:
                    pass
                self.client_actors[nid] = None
        # 强制回收 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    def _repair_tree_after_failures(self, failed_ids: list):
        """修复树结构：将故障节点的子节点重新挂载到故障节点的父节点上"""
        if not self.tree_mapping:
            return

        t0 = time.time()
        for failed_id in failed_ids:
            if failed_id not in self.tree_mapping:
                continue
            node_info = self.tree_mapping[failed_id]
            parent = node_info.get('parent')

            # 将子节点提升到父节点层级
            for child_id in list(node_info.get('children', [])):
                if child_id in self.failed_nodes:
                    continue
                self.tree_mapping[child_id]['parent'] = parent
                if parent is not None and parent in self.tree_mapping:
                    if child_id not in self.tree_mapping[parent]['children']:
                        self.tree_mapping[parent]['children'].append(child_id)
                        self.tree_mapping[parent]['is_leaf'] = False
                # 如果原父节点是簇首（无父节点），子节点晋升为新簇首
                if parent is None:
                    self.tree_mapping[child_id]['cluster_id'] = child_id

            # 从父节点的 children 列表中移除故障节点
            if parent is not None and parent in self.tree_mapping:
                if failed_id in self.tree_mapping[parent]['children']:
                    self.tree_mapping[parent]['children'].remove(failed_id)
                    if not self.tree_mapping[parent]['children']:
                        self.tree_mapping[parent]['is_leaf'] = True

            # 标记故障节点为孤立
            self.tree_mapping[failed_id]['children'] = []
            self.tree_mapping[failed_id]['parent'] = None
            self.tree_mapping[failed_id]['is_leaf'] = True
            self.tree_mapping[failed_id]['total_clients'] = 0
            self.tree_mapping[failed_id]['total_samples'] = 0
            self.tree_mapping[failed_id]['cluster_id'] = None

        # 更新 cluster_heads 列表
        if hasattr(self, 'clustering_results') and self.clustering_results:
            self.clustering_results['cluster_heads'] = [
                nid for nid in self.clustering_results.get('cluster_heads', [])
                if nid not in self.failed_nodes
            ]
            # 添加晋升为簇首的子节点
            for failed_id in failed_ids:
                for nid, info in self.tree_mapping.items():
                    if nid in self.failed_nodes:
                        continue
                    if info.get('parent') is None and info.get('cluster_id') == nid:
                        if nid not in self.clustering_results['cluster_heads']:
                            self.clustering_results['cluster_heads'].append(nid)

        # 更新 sample_counts（将故障节点样本数置零）
        if hasattr(self, 'client_sample_counts'):
            for failed_id in failed_ids:
                if failed_id < len(self.client_sample_counts):
                    self.client_sample_counts[failed_id] = 0

        elapsed = time.time() - t0
        pattern = self.config.failure_pattern if self.config.enable_dynamic_nodes else 'unknown'
        self.recovery_events.append({
            'event': 'tree_repair',
            'failure_pattern': pattern,
            'failed_nodes': list(failed_ids),
            'time_seconds': round(elapsed, 4),
            'method': 'skip',
        })

    def _update_cluster_heads_from_tree(self):
        """从 tree_mapping 重新推导 cluster_heads 并重算子树统计（修复后调用）"""
        if not self.tree_mapping or not self.clustering_results:
            return
        heads = []
        for nid, info in self.tree_mapping.items():
            if nid in self.failed_nodes:
                continue
            if info.get('parent') is None and info.get('cluster_id') is not None:
                if nid not in heads:
                    heads.append(nid)
        self.clustering_results['cluster_heads'] = heads
        # 重新计算所有簇首的子树统计（加入节点后 total_clients/total_samples 需更新）
        for head in heads:
            _calc_subtree_clients(self.tree_mapping, head)
            if hasattr(self, 'client_sample_counts'):
                _calc_subtree_samples(self.tree_mapping, head)

    def _get_live_cluster_heads(self) -> list:
        """获取当前存活的簇首列表"""
        if not self.clustering_results:
            return []
        return [h for h in self.clustering_results.get('cluster_heads', [])
                if h not in self.failed_nodes]

    def _get_live_ray_actors(self):
        """返回 (存活节点actor列表, 存活节点ID列表)"""
        if not hasattr(self, 'client_actors') or not self.client_actors:
            return [], []
        live_actors = []
        live_ids = []
        for i, actor in enumerate(self.client_actors):
            if i not in self.failed_nodes and actor is not None:
                live_actors.append(actor)
                live_ids.append(i)
        return live_actors, live_ids

    def _reelect_and_recluster(self, reason: str = 'unknown') -> tuple:
        """重新选举和聚类（Best Client 或簇首故障时调用）

        构建隔离故障节点的拓扑，重新运行选举和聚类，更新所有内部状态。
        故障节点没有邻居，无法参与选举和聚类。

        Args:
            reason: 'best_client' 或 'cluster_head'，用于计时日志

        Returns:
            (election_results, clustering_results)
        """
        from .election import ElectionManager, ClusteringManager

        t0 = time.time()

        print(f"\n{'='*60}")
        print(f"[恢复] 重新选举 + 重新聚类（排除故障节点: {sorted(self.failed_nodes)}，原因: {reason}）")
        print(f"{'='*60}")

        # 构建隔离故障节点和非活跃节点的拓扑
        excluded = self.failed_nodes | (set(range(self.config.num_clients)) - self.active_nodes)
        modified_topology = []
        for i in range(self.config.num_clients):
            if i in excluded:
                modified_topology.append([])
            else:
                modified_topology.append([
                    n for n in self.topology[i] if n not in excluded
                ])

        # 重新选举（仅活跃节点参与）
        clients = [{'id': i} for i in sorted(self.active_nodes - self.failed_nodes)]
        election_manager = ElectionManager(
            clients=clients,
            topology=modified_topology,
            base_trigger_time=self.config.election_base_trigger_time,
            trigger_factor=self.config.election_trigger_factor
        )
        self.election_results = election_manager.start_election()

        # 确保 Best Client 是活跃且非故障节点
        if (self.election_results['best_client_id'] in self.failed_nodes
                or self.election_results['best_client_id'] not in self.active_nodes):
            all_info = self.election_results.get('all_nodes_info', [])
            live_scores = [(i, info['my_score']) for i, info in enumerate(all_info)
                           if i in self.active_nodes and i not in self.failed_nodes]
            if live_scores:
                new_best = max(live_scores, key=lambda x: x[1])
                self.election_results['best_client_id'] = new_best[0]
                self.election_results['best_client_score'] = new_best[1]
                print(f"  修正 Best Client: Node {new_best[0]} (score={new_best[1]:.2f})")

        new_best = self.election_results['best_client_id']
        print(f"  新 Best Client: Node {new_best} (score={self.election_results['best_client_score']:.2f})")

        # 重新聚类
        label_dists = getattr(self, 'label_distributions', None)
        dist_threshold = getattr(self.config, 'distribution_similarity_threshold', 0.0)
        clustering_manager = ClusteringManager(
            clients=clients,
            topology=modified_topology,
            election_results=self.election_results,
            max_child=self.config.cluster_max_child,
            max_retry=self.config.cluster_max_retry,
            request_timeout=self.config.cluster_request_timeout,
            cluster_offset=self.config.cluster_offset,
            label_distributions=label_dists,
            distribution_similarity_threshold=dist_threshold
        )
        self.clustering_results = clustering_manager.run_clustering()
        self.tree_mapping = build_client_tree_mapping(
            self.clustering_results, self.config.num_clients)

        print(f"  新簇首: {self.clustering_results['cluster_heads']}")
        print(f"  新簇数量: {self.clustering_results['num_clusters']}")

        # 确保故障节点不在树结构中
        for nid in self.failed_nodes:
            if nid in self.tree_mapping:
                self.tree_mapping[nid]['children'] = []
                self.tree_mapping[nid]['parent'] = None
                self.tree_mapping[nid]['is_leaf'] = True
                self.tree_mapping[nid]['total_clients'] = 0
                self.tree_mapping[nid]['total_samples'] = 0
                self.tree_mapping[nid]['cluster_id'] = None

        self._update_cluster_heads_from_tree()

        elapsed = time.time() - t0
        self.recovery_events.append({
            'event': 'reelect_recluster',
            'trigger': reason,
            'time_seconds': round(elapsed, 4),
            'election_hops': self.election_results.get('convergence_hops', -1),
            'num_clusters': self.clustering_results.get('num_clusters', -1),
            'method': 'reelect',
        })
        print(f"  恢复耗时: {elapsed:.2f}s")

        return self.election_results, self.clustering_results

    def run_training(self, num_rounds: int = None) -> Dict:
        if num_rounds is None:
            num_rounds = self.config.rounds

        use_ray = hasattr(self, 'client_actors') and self.client_actors
        use_hierarchical = (hasattr(self, 'clustering_results') and
                           self.clustering_results is not None and
                           hasattr(self, 'tree_mapping') and
                           self.tree_mapping is not None)

        if use_ray:
            if use_hierarchical:
                print(f"\n{'='*80}")
                print(f"分层联邦学习训练模式（Ray并行 + 簇内聚合 + 簇间聚合）")
                print(f"{'='*80}")
            else:
                print(f"\n{'='*80}")
                print(f"开始 {num_rounds} 轮训练（Ray并行模式）")
                print(f"{'='*80}")
        else:
            if not self.models:
                raise RuntimeError("Training not initialized. Call init_training() first.")
            if use_hierarchical:
                print(f"\n{'='*80}")
                print(f"分层联邦学习训练模式（串行 + 簇内聚合 + 簇间聚合）")
                print(f"{'='*80}")
            else:
                print(f"\n{'='*80}")
                print(f"开始 {num_rounds} 轮训练（串行模式）")
                print(f"{'='*80}")

        results = {
            'round_accuracies': [],
            'round_losses': [],
            'global_accuracies': [],
            'final_client_accuracies': [],
            'training_time': 0.0,
            'failure_log': [],      # 动态节点故障日志
            'join_log': [],         # 动态节点加入日志
        }
        self.recovery_events = []   # 恢复事件计时日志
        self._join_triggered = False  # 防止重复触发 join

        # 创建带时间戳的输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = self.config.experiment_name
        exp_tag = f"_{exp_name}" if exp_name and exp_name != "deasfl" else ""
        run_name = f"de_asfl_{self.config.topology_type}_{self.config.num_clients}c_{num_rounds}r{exp_tag}"
        output_dir = os.path.join(self.config.output_dir, f"{run_name}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        print(f"\n输出目录: {output_dir}")

        # 动态节点模式：打印配置信息
        if self.config.enable_dynamic_nodes:
            print(f"\n[动态节点模式]")
            print(f"  故障率: {self.config.failure_rate}, 起始轮: {self.config.failure_start_round}")
            print(f"  最大故障节点: {self.config.failure_max_nodes}, 模式: {self.config.failure_pattern}")
            print(f"  恢复策略: {self.config.recovery_strategy}")

        monitor = Monitor()

        total_start_time = time.time()

        for round_id in range(num_rounds):
            round_start_time = time.time()
            round_losses = []
            round_accs = []

            print(f"\n{'='*60}")
            print(f"第 {round_id + 1}/{num_rounds} 轮训练")
            print(f"{'='*60}")

            # ---- 学习率调度 ----
            if use_ray:
                live_actors, live_ids = self._get_live_ray_actors()
                new_lr = self._calc_learning_rate(round_id)
                ray.get([a.set_learning_rate.remote(new_lr) for a in live_actors])

            # ---- 动态节点管理 ----
            newly_failed = self._simulate_failures(round_id)
            if newly_failed:
                print(f"\n⚠️  节点故障: {newly_failed}")
                print(f"  存活节点: {len(self.active_nodes - self.failed_nodes)}/{self.total_nodes}")
                # 刷新存活 actor 列表（故障节点不再参与后续操作）
                if use_ray:
                    live_actors, live_ids = self._get_live_ray_actors()
                # 更新 cluster_heads
                self._update_cluster_heads_from_tree()

                # reelect_recluster: 簇首故障时触发完整重新选举+聚类
                if self.config.recovery_strategy == "reelect_recluster":
                    old_heads = set(self.clustering_results.get('cluster_heads', [])
                                    if self.clustering_results else [])
                    failed_heads = [n for n in newly_failed if n in old_heads]
                    if failed_heads:
                        print(f"\n⚠️  簇首故障: {failed_heads}，触发重新选举+重新聚类...")
                        self._reelect_and_recluster(reason='cluster_head')
                        if use_ray:
                            live_actors, live_ids = self._get_live_ray_actors()

            # ---- 动态节点加入 ----
            if (self.config.node_join_enabled
                    and round_id == self.config.join_start_round
                    and len(self.active_nodes) < self.config.num_clients
                    and not self._join_triggered):
                self._activate_reserve_nodes(round_id)
                self._join_triggered = True
                if use_ray:
                    live_actors, live_ids = self._get_live_ray_actors()

            # ---- 阶段1: 本地训练（只训练存活节点） ----
            print("\n[阶段1] 各客户端本地训练")
            if use_ray:
                live_actors, live_ids = self._get_live_ray_actors()
                train_results = ray.get([a.local_train.remote(round_id) for a in live_actors])
                round_losses = [r['train_loss'] for r in train_results]

                eval_results = ray.get([a.evaluate.remote() for a in live_actors])
                round_accs = [r['val_acc'] for r in eval_results]

                live_n = len(live_ids)
                for i in range(0, live_n, 10):
                    end = min(i + 10, live_n)
                    cid_start = live_ids[i] if i < live_n else 0
                    cid_end = live_ids[end - 1] if end <= live_n else live_ids[-1]
                    print(f"  存活客户端 {i + 1}-{end} (ID {cid_start}-{cid_end}) 训练完成 - "
                          f"平均准确率: {np.mean(round_accs[i:end]):.4f}")
                if self.failed_nodes:
                    print(f"  已故障节点（跳过）: {sorted(self.failed_nodes)}")
            else:
                for client_id in range(self.config.num_clients):
                    if client_id in self.failed_nodes:
                        continue
                    loss = self._train_single_client(client_id, round_id)
                    val_loss, val_acc = self._evaluate_single_client(client_id)
                    round_losses.append(loss)
                    round_accs.append(val_acc)

                    if (client_id + 1) % 10 == 0:
                        print(f"  客户端 {client_id - 9}-{client_id + 1} 训练完成 - 平均准确率: {np.mean(round_accs[-10:]):.4f}")

            # ---- 阶段2-7: 分层聚合 ----
            comm = CommStats()
            if use_hierarchical:
                cluster_heads = self._get_live_cluster_heads()
                best_client_id = self.election_results.get('best_client_id', -1)

                # 如果 Best Client 故障，根据策略处理
                if best_client_id in self.failed_nodes:
                    if self.config.recovery_strategy == "reelect_recluster":
                        print(f"\n⚠️  Best Client (Node {best_client_id}) 故障，触发重新选举+重新聚类...")
                        self._reelect_and_recluster(reason='best_client')
                        cluster_heads = self._get_live_cluster_heads()
                        best_client_id = self.election_results['best_client_id']
                        print(f"  恢复完成，新 Best Client: Node {best_client_id}，"
                              f"簇首: {cluster_heads}")
                    else:
                        print(f"\n⚠️  Best Client (Node {best_client_id}) 已故障，"
                              f"使用第一个存活节点作为聚合节点")
                        best_client_id = next(iter(self.active_nodes - self.failed_nodes), 0)

                if use_ray:
                    live_actors, live_ids = self._get_live_ray_actors()
                    live_params = ray.get([a.get_model_params.remote() for a in live_actors])
                    # 构建完整参数列表（client_id 索引，故障节点为 None）
                    all_params = [None] * self.config.num_clients
                    for lid, params in zip(live_ids, live_params):
                        all_params[lid] = params
                    wrapped_models = [ModelWrapper(p) if p is not None else None for p in all_params]
                else:
                    wrapped_models = self.models

                # 聚合在 CPU 上做（避免 GPU OOM，30个ResNet18 + 中间结果 ~3GB）
                agg_device = torch.device('cpu')

                print("\n[阶段2] 簇内分层聚合")
                aggregated_models = hierarchical_aggregate(
                    wrapped_models,
                    self.tree_mapping,
                    cluster_heads,
                    agg_device,
                    comm=comm,
                    skip_nodes=self.failed_nodes,
                )
                print(f"  簇内分层聚合完成，共 {len(cluster_heads)} 个簇")
                print(f"  📡 通信: 簇内上传 {comm.intra_cluster_msgs} 条消息")

                print("\n[阶段3] 簇间聚合（上传到最佳客户端）")
                cluster_sizes, routing_tables = build_cluster_info(
                    self.tree_mapping,
                    self.clustering_results,
                    self.election_results
                )
                print(f"  最佳客户端: Node {best_client_id}")
                print(f"  簇大小信息: {cluster_sizes}")

                message_queue = {}
                for head_id in cluster_heads:
                    forward_model_to_best_client(
                        source_cluster_head=head_id,
                        model_dict=aggregated_models[head_id],
                        client_count=cluster_sizes[head_id],
                        routing_tables=routing_tables,
                        best_client_id=best_client_id,
                        message_queue=message_queue,
                        comm=comm
                    )
                    print(f"  簇首 {head_id} 的模型已发送到最佳客户端（路径: {message_queue[best_client_id][-1].path_stack}）")
                print(f"  📡 通信: 簇间转发 {comm.inter_cluster_forward_hops} 跳")

                print("\n[阶段4] 最佳客户端聚合所有簇首模型")
                final_model, path_stacks = aggregate_models_at_best_client(
                    best_client_id=best_client_id,
                    models=wrapped_models,
                    message_queue=message_queue,
                    device=agg_device
                )
                print(f"  聚合了 {len(path_stacks)} 个簇首的模型")

                print("\n[阶段5] 原路返回模型到各簇首")
                return_queue = {}
                return_model_to_cluster_heads(
                    final_model=final_model,
                    path_stacks=path_stacks,
                    return_queue=return_queue,
                    comm=comm
                )
                print(f"  📡 通信: 簇间返回 {comm.inter_cluster_return_hops} 跳")

                print("\n[阶段6] 将最终模型分发到所有存活节点")
                if use_ray:
                    final_model_cpu = {k: v.cpu() for k, v in final_model.items()}
                    live_actors, live_ids = self._get_live_ray_actors()
                    ray.get([a.set_model_params.remote(final_model_cpu) for a in live_actors])
                    comm.flat_download_msgs = len(live_ids)
                else:
                    for head_id in cluster_heads:
                        distribute_models_to_cluster(
                            aggregated_models={head_id: final_model},
                            tree_mapping=self.tree_mapping,
                            models=self.models,
                            cluster_heads=[head_id],
                            device=self.device,
                            comm=comm
                        )
                print(f"  📡 通信: 簇内分发 {comm.intra_cluster_dist_msgs} 条消息")

                print("\n[阶段7] 计算所有存活节点在本地验证集上的平均准确率")
                if use_ray:
                    live_actors, live_ids = self._get_live_ray_actors()
                    eval_results = ray.get([a.evaluate.remote() for a in live_actors])
                    all_node_accs = [r['val_acc'] for r in eval_results]
                else:
                    all_node_accs = []
                    eval_results = []
                    for client_id in range(self.config.num_clients):
                        _, val_acc = self._evaluate_single_client(client_id)
                        all_node_accs.append(val_acc)
                        eval_results.append({
                            'val_acc': val_acc,
                            'num_samples': len(self.val_loaders[client_id].dataset),
                        })

                avg_all_node_acc = weighted_validation_accuracy(eval_results)
                results['global_accuracies'].append(avg_all_node_acc)
                print(f"  所有节点平均准确率: {avg_all_node_acc:.4f}")

            elif use_ray:
                # ---- 阶段2-4: 扁平聚合（消融实验：无聚类模式） ----
                print("\n[阶段2] 扁平聚合（存活客户端 → 聚合节点）")
                live_actors, live_ids = self._get_live_ray_actors()
                all_params = ray.get([a.get_model_params.remote() for a in live_actors])
                agg_device = torch.device('cpu')

                # 有选举则用 Best Client 作为聚合节点，否则用 client 0
                best_id = (self.election_results.get('best_client_id', 0)
                           if (hasattr(self, 'election_results') and self.election_results)
                           else 0)
                # 如果 Best Client 故障，选择新的聚合节点
                if best_id in self.failed_nodes:
                    best_id = live_ids[0] if live_ids else 0

                final_params = flat_aggregate(all_params, device=agg_device)
                comm.flat_upload_msgs = len(all_params)
                if all_params:
                    comm.model_bytes = compute_model_bytes(all_params[0])
                print(f"  聚合了 {len(all_params)} 个客户端的模型 (聚合节点: Node {best_id})")
                print(f"  📡 通信: 上传 {comm.flat_upload_msgs} 条")

                print("\n[阶段3] 分发聚合模型到所有存活节点")
                final_params_cpu = {k: v.cpu() for k, v in final_params.items()}
                ray.get([a.set_model_params.remote(final_params_cpu) for a in live_actors])
                comm.flat_download_msgs = len(live_ids)
                print(f"  📡 通信: 下载 {comm.flat_download_msgs} 条")

                print(f"  📡 本轮总通信: {comm.total_msgs} 条消息, {comm.total_bytes:,} 字节")

                print("\n[阶段4] 计算所有存活节点在本地验证集上的平均准确率")
                eval_results = ray.get([a.evaluate.remote() for a in live_actors])
                avg_all_node_acc = weighted_validation_accuracy(eval_results)
                results['global_accuracies'].append(avg_all_node_acc)
                print(f"  存活节点平均准确率: {avg_all_node_acc:.4f}")

            avg_loss = np.mean(round_losses)
            avg_acc = np.mean(round_accs)
            results['round_losses'].append(avg_loss)
            results['round_accuracies'].append(avg_acc)

            round_time = time.time() - round_start_time
            print(f"\n  本轮平均训练损失: {avg_loss:.4f}")
            print(f"  本轮平均验证准确率: {avg_acc:.4f}")
            if results['global_accuracies']:
                print(f"  本轮全局准确率: {results['global_accuracies'][-1]:.4f}")
            print(f"  本轮耗时: {round_time:.2f}s")

            # 记录本轮指标（使用分层聚合后的全局准确率作为 val_acc）
            global_acc = results['global_accuracies'][-1] if results['global_accuracies'] else avg_acc
            live_count = len(self.active_nodes - self.failed_nodes)
            failed_count = len(self.failed_nodes)
            monitor.record_round(
                round_id=round_id,
                avg_train_loss=avg_loss,
                val_loss=avg_loss,
                val_acc=global_acc,
                selected_count=live_count,
                dropped_count=failed_count,
                comm_success_count=comm.total_msgs if use_ray else 0,
                comm_failure_count=0,
                comm_timeout_count=0,
                broadcast_success_count=comm.flat_download_msgs if use_ray else 0,
                broadcast_failure_count=0,
                upload_success_count=comm.flat_upload_msgs if use_ray else 0,
                upload_failure_count=0,
                upload_storage_blocked_count=0,
                gossip_success_count=0,
                gossip_failure_count=0,
            )
            # 保存每轮通信统计
            if 'round_comm_stats' not in results:
                results['round_comm_stats'] = []
            results['round_comm_stats'].append(comm.to_dict())

        results['training_time'] = time.time() - total_start_time

        # 保存动态节点日志
        results['failure_log'] = self.failure_log
        results['join_log'] = self.join_log

        print(f"\n{'='*60}")
        print(f"训练完成")
        print(f"{'='*60}")

        if self.failed_nodes:
            print(f"\n动态节点统计:")
            print(f"  故障节点总数: {len(self.failed_nodes)}")
            print(f"  故障节点列表: {sorted(self.failed_nodes)}")
            print(f"  存活节点数: {len(self.active_nodes - self.failed_nodes)}")

        print("\n最终每个存活客户端的准确率:")
        if use_ray:
            live_actors, live_ids = self._get_live_ray_actors()
            eval_results = ray.get([a.evaluate.remote() for a in live_actors])
            acc_by_id = dict(zip(live_ids, [r['val_acc'] for r in eval_results]))
            final_accs = [
                acc_by_id.get(i, 0.0) if i not in self.failed_nodes else 0.0
                for i in range(self.config.num_clients)
            ]
        else:
            final_accs = []
            for client_id in range(self.config.num_clients):
                if client_id in self.failed_nodes:
                    final_accs.append(0.0)  # 故障节点准确率记为 0
                else:
                    _, val_acc = self._evaluate_single_client(client_id)
                    final_accs.append(val_acc)

        results['final_client_accuracies'] = final_accs

        for i in range(0, self.config.num_clients, 10):
            end = min(i + 10, self.config.num_clients)
            acc_str = ", ".join([
                f"{final_accs[j]:.4f}" if j not in self.failed_nodes else f"FAILED"
                for j in range(i, end)
            ])
            print(f"  客户端 {i + 1}-{end}: {acc_str}")

        live_accs = [a for i, a in enumerate(final_accs) if i not in self.failed_nodes]
        if live_accs:
            print(f"\n训练统计（存活节点）:")
            print(f"  平均准确率: {np.mean(live_accs):.4f}")
            print(f"  最高准确率: {np.max(live_accs):.4f}")
            print(f"  最低准确率: {np.min(live_accs):.4f}")
        print(f"  总训练耗时: {results['training_time']:.2f}s")

        if results['global_accuracies']:
            print(f"\n全局平均准确率统计（所有节点本地验证集平均）:")
            print(f"  最终准确率: {results['global_accuracies'][-1]:.4f}")
            print(f"  最高准确率: {np.max(results['global_accuracies']):.4f}")
            print(f"  最低准确率: {np.min(results['global_accuracies']):.4f}")

        # ---- 保存指标并生成图表 ----
        print(f"\n{'='*60}")
        print("保存训练指标和图表")
        print(f"{'='*60}")

        monitor.dump(output_dir, run_name)
        print(f"  指标文件已保存到: {output_dir}")

        # ---- 保存动态节点日志 ----
        if self.failure_log or self.join_log:
            dyn_log = {
                'failure_log': self.failure_log,
                'join_log': self.join_log,
                'total_nodes': self.total_nodes,
                'original_num_clients': self.original_num_clients,
                'failed_nodes': sorted(self.failed_nodes),
                'joined_nodes': sorted(self.joined_nodes),
                'config': {
                    'enable_dynamic_nodes': self.config.enable_dynamic_nodes,
                    'failure_rate': self.config.failure_rate,
                    'failure_pattern': self.config.failure_pattern,
                    'failure_start_round': self.config.failure_start_round,
                    'failure_max_nodes': self.config.failure_max_nodes,
                    'recovery_strategy': self.config.recovery_strategy,
                },
                'recovery_events': getattr(self, 'recovery_events', []),
            }
            import json as _json
            dyn_log_path = os.path.join(output_dir, f"{run_name}_dynamic_nodes.json")
            with open(dyn_log_path, 'w') as f:
                _json.dump(dyn_log, f, indent=2, ensure_ascii=False)
            print(f"  动态节点日志已保存: {dyn_log_path}")

        # ---- 保存通信统计 ----
        if 'round_comm_stats' in results and results['round_comm_stats']:
            import json as _json
            import pandas as pd
            comm_csv_path = os.path.join(output_dir, f"{run_name}_communication_stats.csv")
            comm_df = pd.DataFrame(results['round_comm_stats'])
            comm_df.to_csv(comm_csv_path, index=False)
            print(f"  通信统计已保存: {comm_csv_path}")

            # 汇总
            total = comm_df['total_msgs'].sum()
            total_bytes = comm_df['total_bytes'].sum()
            avg_per_round = comm_df['total_msgs'].mean()
            model_mb = comm_df['model_bytes'].iloc[-1] / (1024 * 1024) if len(comm_df) > 0 else 0
            print(f"  通信汇总: 总计 {total} 条消息, {total_bytes / (1024**2):.1f} MB")
            print(f"    每轮平均 {avg_per_round:.1f} 条消息 ({avg_per_round * model_mb:.1f} MB)")
            print(f"    模型大小: {model_mb:.1f} MB")

        config_dict = {
            'algorithm': 'de_asfl',
            'topology': self.config.topology_type,
            'num_clients': self.config.num_clients,
            'rounds': num_rounds,
            'dataset': self.config.dataset,
            'model': self.config.model,
            'data_partition_mode': self.config.partition,
            'dirichlet_alpha': self.config.alpha,
            'enable_election': self.config.enable_election,
            'enable_clustering': self.config.enable_clustering,
        }
        # 追加通信汇总
        if 'round_comm_stats' in results and results['round_comm_stats']:
            import pandas as pd
            comm_df = pd.DataFrame(results['round_comm_stats'])
            config_dict['comm_total_msgs'] = int(comm_df['total_msgs'].sum())
            config_dict['comm_total_bytes'] = int(comm_df['total_bytes'].sum())
            config_dict['comm_avg_msgs_per_round'] = float(comm_df['total_msgs'].mean())
            config_dict['comm_model_bytes'] = int(comm_df['model_bytes'].iloc[-1]) if len(comm_df) > 0 else 0
            config_dict['comm_intra_cluster_msgs'] = int(comm_df['intra_cluster_msgs'].sum())
            config_dict['comm_inter_fwd_hops'] = int(comm_df['inter_cluster_forward_hops'].sum())
            config_dict['comm_inter_ret_hops'] = int(comm_df['inter_cluster_return_hops'].sum())

        try:
            from fl_core.plotter import generate_summary_report
            import pandas as pd
            metrics_path = os.path.join(output_dir, f"{run_name}_round_metrics.csv")
            if os.path.exists(metrics_path):
                round_metrics_df = pd.read_csv(metrics_path)
                generate_summary_report(round_metrics_df, config_dict, output_dir)
            else:
                print(f"  Warning: 指标CSV未找到: {metrics_path}")
        except ImportError as e:
            print(f"  Warning: 无法导入绘图模块: {e}")
        except Exception as e:
            print(f"  Warning: 生成图表失败: {e}")

        return results
