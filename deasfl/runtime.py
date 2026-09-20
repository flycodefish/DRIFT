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
from fl_core.artifacts import (
    failure_state_at_round, frozen_failure_events, generate_node_resources, load_frozen_inputs,
    normalize_physical_graph, validate_frozen_inputs, write_completion_status,
    write_experiment_inputs,
)
from fl_core.system_metrics import dump_system_metrics
from fl_core.data_loader import (
    SimpleModel, CNNModel, HARMLP,
    load_cifar10_data, load_cifar100_data, load_neu_data, load_har_data,
    resolve_dataset_split_policy,
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
    from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import ray

from .config import NewFLConfig
from .aggregation import (
    WeightedState,
    finalize_weighted_state,
    leaf_weighted_state,
    merge_weighted_states,
)


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
    def __init__(self, round_id: int = 0, bandwidth_mbps=None,
                 latency_ms=None, drop_rate: float = 0.0):
        self.round_id = round_id
        self.bandwidth_mbps = bandwidth_mbps
        self.latency_ms = latency_ms
        self.drop_rate = drop_rate
        self.intra_cluster_msgs: int = 0       # 簇内树边上传（子→父）
        self.inter_cluster_forward_hops: int = 0  # 簇间转发跳数（簇首→最优节点）
        self.inter_cluster_return_hops: int = 0   # 簇间返回跳数（最优节点→簇首）
        self.intra_cluster_dist_msgs: int = 0    # 簇内树边分发（父→子）
        self.model_bytes: int = 0               # 单个模型大小（字节）
        self.flat_upload_msgs: int = 0           # 扁平聚合：上传消息数
        self.flat_download_msgs: int = 0         # 扁平聚合：下载消息数
        self.global_sync_performed: bool = True  # 本训练周期是否执行簇间全局同步
        self.events: list[dict] = []

    def record_transfer(self, src: int, dst: int, message_type: str,
                        payload_bytes: int, hop_index: int = 1,
                        hop_count: int = 1, phase: str | None = None,
                        flow_id: str | None = None,
                        logical_parent_message_id: str | None = None) -> None:
        """Record one logical hop without changing legacy counters."""
        self.events.append({
            "round": self.round_id, "src": src, "dst": dst,
            "message_type": message_type, "payload_bytes": payload_bytes,
            "bandwidth_mbps": self.bandwidth_mbps,
            "latency_ms": self.latency_ms, "drop_rate": self.drop_rate,
            "phase": phase or {
                "AGGREGATED_MODEL": "INTRA_CLUSTER_UPLOAD",
                "ROUTED_MODEL": "INTER_CLUSTER_UPLOAD",
                "GLOBAL_MODEL": "MODEL_DISTRIBUTION",
            }.get(message_type, "CONTROL"),
            "flow_id": flow_id,
            "logical_parent_message_id": logical_parent_message_id,
            "link_type": "LOGICAL_SIMULATED", "success": True,
            "physical_hop_index": hop_index,
            "physical_hop_count": hop_count,
            "attempt_count": 1, "backoff_ms": 0.0,
        })

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
            'round_id': self.round_id,
            'global_sync_performed': self.global_sync_performed,
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


def global_sync_due(round_id: int, num_rounds: int, interval: int) -> bool:
    """Return whether a completed local/cluster cycle should globally sync."""
    interval = max(1, int(interval))
    return (round_id + 1) % interval == 0 or round_id == num_rounds - 1


def compute_model_bytes(params: Dict[str, torch.Tensor]) -> int:
    """计算一个模型的 state_dict 的字节大小"""
    total = 0
    for tensor in params.values():
        total += tensor.numel() * tensor.element_size()
    return total


def weighted_validation_accuracy(eval_results: List[Dict]) -> float:
    """Aggregate client validation accuracy by validation sample count."""
    total_samples = sum(result.get("num_samples", 0) for result in eval_results)
    if total_samples == 0:
        return float(np.mean([result["val_acc"] for result in eval_results]))
    return sum(
        result["val_acc"] * result["num_samples"] for result in eval_results
    ) / total_samples


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
                    comm.record_transfer(
                        child_id, node_id, "AGGREGATED_MODEL", comm.model_bytes,
                        phase="INTRA_CLUSTER_UPLOAD",
                        flow_id=f"cluster:{tree_mapping[node_id].get('cluster_id', node_id)}:up",
                    )

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


def hierarchical_weighted_reduce(models: List, tree_mapping: Dict[int, Dict],
                                 cluster_heads: List[int],
                                 skip_nodes: Optional[set] = None,
                                 comm: Optional['CommStats'] = None,
                                 ) -> Dict[int, Optional[WeightedState]]:
    """Accumulate float64 weighted sums without intermediate normalization."""
    if skip_nodes is None:
        skip_nodes = set()
    reduced = {}

    def collect(node_id):
        if node_id in reduced:
            return reduced[node_id]
        parts = []
        for child_id in tree_mapping[node_id].get('children', []):
            if child_id in skip_nodes:
                continue
            child = collect(child_id)
            if child is not None:
                parts.append(child)
                if comm is not None:
                    comm.intra_cluster_msgs += 1
                    comm.record_transfer(
                        child_id, node_id, "AGGREGATED_MODEL", comm.model_bytes,
                        phase="INTRA_CLUSTER_UPLOAD",
                        flow_id=f"cluster:{tree_mapping[node_id].get('cluster_id', node_id)}:up",
                    )
        if node_id not in skip_nodes and models[node_id] is not None:
            count = tree_mapping[node_id].get('sample_count') or 1
            parts.append(leaf_weighted_state(
                models[node_id].state_dict(), count, torch.device('cpu')
            ))
        if not parts:
            reduced[node_id] = None
            return None
        reduced[node_id] = merge_weighted_states(parts)
        return reduced[node_id]

    for head in cluster_heads:
        if head not in skip_nodes:
            collect(head)
    return reduced


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
                # Every descendant receives the same finalized cluster/global
                # model. Loading a child's stale subtree aggregate here breaks
                # FedAvg equivalence and leaves ordinary members unsynchronized.
                if comm is not None:
                    comm.intra_cluster_dist_msgs += 1
                    comm.record_transfer(
                        node_id, child_id, "GLOBAL_MODEL", comm.model_bytes,
                        phase="INTRA_CLUSTER_DOWNLOAD",
                        flow_id=f"cluster:{tree_mapping[node_id].get('cluster_id', node_id)}:down",
                    )
                distribute_recursive(child_id, model_state_dict)

    for cluster_head in cluster_heads:
        if cluster_head in aggregated_models:
            distribute_recursive(cluster_head, aggregated_models[cluster_head])


def distribute_returned_models_to_clusters(returned_models: Dict[int, Dict],
                                           tree_mapping: Dict[int, Dict],
                                           models: List,
                                           cluster_heads: List[int],
                                           skip_nodes: Optional[set] = None,
                                           comm: Optional['CommStats'] = None,
                                           allow_unreachable: bool = False) -> None:
    """Consume the GEA return result and broadcast it down each original cluster tree."""
    skip_nodes = skip_nodes or set()

    def distribute_recursive(node_id: int, model_state: Dict) -> None:
        if node_id in skip_nodes or models[node_id] is None:
            return
        models[node_id].load_state_dict(model_state)
        for child_id in sorted(tree_mapping[node_id]['children']):
            if child_id in skip_nodes:
                continue
            if tree_mapping[child_id]['cluster_id'] != tree_mapping[node_id]['cluster_id']:
                continue
            if comm is not None:
                comm.intra_cluster_dist_msgs += 1
                comm.record_transfer(
                    node_id, child_id, "GLOBAL_MODEL", comm.model_bytes,
                    phase="INTRA_CLUSTER_DOWNLOAD",
                    flow_id=f"cluster:{tree_mapping[node_id].get('cluster_id', node_id)}:down",
                )
            distribute_recursive(child_id, model_state)

    for head_id in cluster_heads:
        if head_id not in returned_models:
            if allow_unreachable:
                continue
            raise RuntimeError(f"Cluster head {head_id} did not receive the global model")
        distribute_recursive(head_id, returned_models[head_id])


# ==============================================================================
#                          簇间聚合（最佳客户端）
# ==============================================================================

from typing import Dict, List, Tuple
from copy import deepcopy


@dataclass
class ModelForwardMessage:
    """模型转发消息"""
    source_id: int                  # 源节点（簇首）
    weighted_state: WeightedState   # float64 加权和 + 真实样本数
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
        cluster_sizes: {cluster_head_id: total_sample_count}
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
                                 weighted_state: WeightedState,
                                 routing_tables: Dict[int, Dict],
                                 best_client_id: int,
                                 message_queue: Dict[int, List],
                                 comm: Optional['CommStats'] = None) -> None:
    """
    簇首将模型沿路由转发到最佳客户端

    Args:
        source_cluster_head: 源簇首ID
        weighted_state: 该簇未归一化的加权和与样本数
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
        # A disconnected cluster must not be included in the GEA aggregate.
        # Logging a warning and still enqueueing it would make link failures
        # cosmetic rather than changing reachability.
        return

    # 统计：路径上除源节点外的每一跳都是一次模型转发
    hops = len(path_stack) - 1
    if comm is not None and hops > 0:
        comm.inter_cluster_forward_hops += hops
        if comm.model_bytes == 0:
            # WeightedState is a high-precision numerical accumulator. The
            # protocol still transmits one model payload per hop, so preserve
            # float32 model-size accounting used by all communication results.
            comm.model_bytes = compute_model_bytes(
                finalize_weighted_state(weighted_state)
            )
        for hop_index, (src, dst) in enumerate(
            zip(path_stack[:-1], path_stack[1:]), start=1
        ):
            comm.record_transfer(
                src, dst,
                "AGGREGATED_MODEL" if dst == best_client_id else "ROUTED_MODEL",
                comm.model_bytes,
                hop_index=hop_index, hop_count=hops,
                phase="INTER_CLUSTER_UPLOAD",
                flow_id=f"cluster:{source_cluster_head}:inter-up",
            )

    msg = ModelForwardMessage(
        source_id=source_cluster_head,
        weighted_state=deepcopy(weighted_state),
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
    
    weighted_states = []
    path_stacks = {}
    
    for msg in messages:
        weighted_states.append(msg.weighted_state)
        path_stacks[msg.source_id] = msg.path_stack
    
    final_model = finalize_weighted_state(merge_weighted_states(weighted_states))
    
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
            for hop_index, (src, dst) in enumerate(
                zip(reverse_path[:-1], reverse_path[1:]), start=1
            ):
                comm.record_transfer(
                    src, dst, "GLOBAL_MODEL", comm.model_bytes,
                    hop_index=hop_index, hop_count=hops,
                    phase="INTER_CLUSTER_DOWNLOAD",
                    flow_id=f"cluster:{source_head}:inter-down",
                )

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
        self.global_test_data = None
        self._initial_model_state = None
        self.frozen_inputs = None
        self.node_resources_input = None

        # 动态节点管理（Node Failure / Join）
        self.original_num_clients = self.config.num_clients
        self.initial_active = (self.config.initial_active_clients
                               if self.config.initial_active_clients > 0
                               else self.config.num_clients)
        self.total_nodes = self.config.num_clients
        self.active_nodes: set = set(range(self.initial_active))
        self.failed_nodes: set = set()
        self.failed_edges: set[tuple[int, int]] = set()
        self._last_recovered_nodes: list[int] = []
        self._last_link_state_changed = False
        self.joined_nodes: set = set()
        self.failure_log: list = []    # 每轮的故障记录
        self.join_log: list = []       # 每轮的加入记录
        
    def start(self):
        """启动运行时"""
        if self.config.use_gpu and not torch.cuda.is_available():
            raise RuntimeError("GPU execution requested but CUDA is unavailable")
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if self.config.frozen_inputs_dir:
            self.frozen_inputs = load_frozen_inputs(self.config.frozen_inputs_dir)
            expected_fingerprint = None
            if self.config.dataset == "har":
                from fl_core.data_loader import har_dataset_fingerprint
                expected_fingerprint = har_dataset_fingerprint(self.config.data_dir)
            expected_split_policy = resolve_dataset_split_policy(
                self.config.dataset, getattr(self.config, "dataset_split_policy", None)
            )
            validate_frozen_inputs(
                self.frozen_inputs,
                self.config.num_clients,
                expected_dataset=self.config.dataset,
                expected_split_policy=expected_split_policy,
                expected_local_val_seed=(getattr(self.config, "har_local_val_seed", None)
                                         if self.config.dataset == "har" else None),
                expected_dataset_fingerprint=expected_fingerprint,
                strict=getattr(self.config, "strict_frozen_inputs", False),
            )
        self.node_resources_input = (
            self.frozen_inputs["node_resources.json"] if self.frozen_inputs
            else generate_node_resources(self.config.num_clients, self.config.seed)
        )
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

    def set_global_test_data(self, global_test_data) -> None:
        """Register the shared test set used for the sole formal evaluation."""
        if (self.frozen_inputs and getattr(self.config, "strict_frozen_inputs", False)
                and self.config.dataset == "har"):
            from fl_core.data_loader import tensor_fingerprint
            expected = self.frozen_inputs["data_partition.json"].get("global_test_fingerprint")
            actual = tensor_fingerprint(global_test_data["images"], global_test_data["labels"])
            if expected != actual:
                raise ValueError("Frozen HAR global test fingerprint does not match loaded data")
        self.global_test_data = global_test_data
    
    def build_topology(self):
        """构建网络拓扑"""
        if self.frozen_inputs:
            graph = self.frozen_inputs["physical_graph.json"]
            if isinstance(graph, dict):
                self.topology = [list(graph[str(i)] if str(i) in graph else graph[i])
                                 for i in range(self.config.num_clients)]
            else:
                self.topology = [list(neighbors) for neighbors in graph]
        else:
            self.topology = self._build_topology(
            self.config.num_clients,
            self.config.topology_type,
            self.config.neighbor_selection,
            self.config.min_neighbors,
            self.config.max_neighbors
        )
        normalized = normalize_physical_graph(
            self.topology, self.config.num_clients
        )
        self.topology = [normalized[node] for node in range(self.config.num_clients)]
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
            trigger_factor=self.config.election_trigger_factor,
            seed=self.config.seed,
            resource_params=self._frozen_election_resources(),
        )
        
        # 运行选举
        self.election_results = election_manager.start_election()
        return self.election_results

    def _frozen_election_resources(self):
        if not self.node_resources_input:
            return None
        resources = self.node_resources_input
        if not isinstance(resources, list) or not resources:
            return None
        normalized = []
        for item in resources:
            params = item.get("resource_params", item)
            required = {"cpu_cores", "memory_gb", "bandwidth_mbps", "gpu_memory_gb"}
            if not required.issubset(params):
                return None
            value = {key: params[key] for key in required}
            if not self.config.resource_awareness:
                value["resource_score_override"] = 1.0
            normalized.append(value)
        return normalized
    
    def init_clustering(self, election_results: Dict, max_child: int, cluster_offset: float):
        """初始化聚类"""
        print(f"Initializing clustering with max_child={max_child}, cluster_offset={cluster_offset}")
    
    def run_clustering(self):
        """运行聚类算法"""
        from .election import ClusteringManager
        from .cluster_assignments import (
            EXPERIMENTAL_MODES, build_matched_clustering_result,
        )

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
        reference_result = clustering_manager.run_clustering()
        assignment_mode = getattr(
            self.config, "clustering_assignment_mode", "distributed"
        )
        if assignment_mode == "distributed":
            self.clustering_results = reference_result
        elif assignment_mode in EXPERIMENTAL_MODES:
            if label_dists is None or len(label_dists) != self.config.num_clients:
                raise RuntimeError(
                    f"{assignment_mode} requires one training-label distribution per client"
                )
            print("\n" + "!" * 72)
            print(f"EXPERIMENT ONLY: centralized cluster assignment mode={assignment_mode}")
            print("This mode is not part of the deployable distributed DRIFT protocol.")
            print("!" * 72)
            self.clustering_results = build_matched_clustering_result(
                reference_result=reference_result,
                label_distributions=label_dists,
                election_results=self.election_results,
                max_child=self.config.cluster_max_child,
                mode=assignment_mode,
                seed=self.config.seed,
            )
        else:
            raise ValueError(f"Unknown clustering_assignment_mode: {assignment_mode}")
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training device: {self.device}")

        self.models = []
        self.optimizers = []
        self.criterions = []
        self.train_loaders = []
        self.val_loaders = []

        for client_id in range(self.config.num_clients):
            if use_cnn:
                model = CNNModel(num_classes=num_classes).to(self.device)
            else:
                model = SimpleModel(input_dim=input_dim, num_classes=num_classes).to(self.device)

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

            self.train_loaders.append(DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True))
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
            return HARMLP(
                input_dim=getattr(self.config, 'input_dim', 561),
                num_classes=self.config.num_classes,
            )
        else:
            return SimpleModel(input_dim=10, num_classes=self.config.num_classes)

    def _calc_learning_rate(self, round_id: int) -> float:
        """Cosine Annealing + Warmup 学习率计算"""
        initial_lr = float(self.config.learning_rate)
        min_lr = float(self.config.learning_rate_min)
        warmup = int(self.config.warmup_rounds)
        schedule_rounds = int(self.config.learning_rate_schedule_rounds)
        max_rounds = schedule_rounds if schedule_rounds > 0 else int(self.config.rounds)

        if self.config.cosine_annealing:
            if round_id < warmup:
                return initial_lr * ((round_id + 1) / max(1, warmup))
            else:
                progress = round_id - warmup
                total_steps = max_rounds - warmup
                return min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * progress / total_steps))
        else:
            return max(initial_lr * (0.95 ** round_id), min_lr)

    def _init_ray_actors(self, train_data, val_data, global_test_data=None):
        """创建 NewClientActor 的 Ray 实例，平分到多 GPU 上"""
        from .actors import NewClientActor

        num_gpus = torch.cuda.device_count()
        self.device = torch.device(
            "cuda:0" if self.config.use_gpu and num_gpus > 0 else "cpu"
        )

        # 所有客户端共享同一个初始模型（对齐 FedAvg）
        torch.manual_seed(self.config.seed)
        base_model = self._create_model()
        base_state = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}
        if self.frozen_inputs:
            frozen = self.frozen_inputs["initial_model.pt"]
            if base_state.keys() != frozen.keys():
                raise ValueError("Frozen initial model is incompatible with configured model")
            for key in base_state:
                if base_state[key].shape != frozen[key].shape:
                    raise ValueError(f"Frozen initial model shape mismatch: {key}")
            base_state = {key: value.detach().cpu().clone() for key, value in frozen.items()}
        self._initial_model_state = base_state
        if global_test_data is not None:
            self.set_global_test_data(global_test_data)

        self.client_actors = [None] * self.config.num_clients
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
            actor_gpu = self.config.num_gpus if self.config.use_gpu else 0.0
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
                deterministic_training=self.config.deterministic_training,
                merge_singleton_batch=(self.config.dataset == "har"),
            )
            actor.set_local_epochs.remote(self.config.local_epochs)
            self.client_actors[client_id] = actor

        # 存储每个客户端的样本数（用于样本加权聚合）
        self.client_sample_counts = [len(train_data[i]['labels'])
                                     for i in range(self.config.num_clients)]
        set_tree_sample_counts(
            self.tree_mapping,
            self.client_sample_counts,
            self.clustering_results.get('cluster_heads', []),
        )

        # 保存训练数据引用（供后续 join 时创建新 actor 使用）
        self._reserve_train_data = train_data
        self._reserve_val_data = val_data

        active_count = sum(1 for a in self.client_actors if a is not None)
        print(f"Initialized {active_count} Ray client actors "
              f"across {num_gpus} GPU(s) (explicit round-robin assignment)")

    def _evaluate_global_params(self, params: Dict[str, torch.Tensor]) -> tuple:
        """Evaluate one global state once on the complete shared test set."""
        if self.global_test_data is None:
            raise RuntimeError("Global test data is not initialized")

        device = torch.device(
            "cuda:0" if self.config.use_gpu and torch.cuda.is_available() else "cpu"
        )
        with torch.random.fork_rng(devices=[]):
            model = self._create_model()
        model.load_state_dict(params)
        model = model.to(device)
        model.eval()
        loader = DataLoader(
            TensorDataset(
                self.global_test_data['images'],
                self.global_test_data['labels'],
            ),
            batch_size=max(1, self.config.batch_size),
            shuffle=False,
        )
        criterion = nn.CrossEntropyLoss()
        loss_sum = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss_sum += criterion(logits, yb).item() * yb.size(0)
                correct += (logits.argmax(dim=1) == yb).sum().item()
                total += yb.size(0)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return loss_sum / max(1, total), correct / max(1, total)

    def _evaluate_params_on_dataset(self, params: Dict[str, torch.Tensor], dataset) -> tuple:
        """Evaluate an intermediate state without mutating any client actor."""
        device = torch.device(
            "cuda:0" if self.config.use_gpu and torch.cuda.is_available() else "cpu"
        )
        with torch.random.fork_rng(devices=[]):
            model = self._create_model()
        model.load_state_dict(params)
        model = model.to(device)
        model.eval()
        loader = DataLoader(
            dataset, batch_size=max(1, self.config.batch_size), shuffle=False
        )
        criterion = nn.CrossEntropyLoss()
        loss_sum = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss_sum += criterion(logits, yb).item() * yb.size(0)
                correct += (logits.argmax(dim=1) == yb).sum().item()
                total += yb.size(0)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return loss_sum / max(1, total), correct / max(1, total), total

    def _cluster_members(self, head_id: int) -> list[int]:
        """Return live members represented by one cluster tree."""
        members = []

        def visit(node_id: int) -> None:
            if node_id in self.failed_nodes:
                return
            members.append(node_id)
            for child_id in (self.tree_mapping or {}).get(node_id, {}).get("children", []):
                visit(child_id)

        visit(head_id)
        return members

    def _pairwise_label_jsd_stats(self, members: list[int]) -> tuple[float, int]:
        """Natural-log JSD mean and number of real client pairs in one cluster."""
        distributions = getattr(self, "label_distributions", None)
        if distributions is None or len(members) < 2:
            return 0.0, 0
        values = []
        epsilon = 1e-10
        for left_index, left_id in enumerate(members):
            p = np.asarray(distributions[left_id], dtype=np.float64) + epsilon
            p /= p.sum()
            for right_id in members[left_index + 1:]:
                q = np.asarray(distributions[right_id], dtype=np.float64) + epsilon
                q /= q.sum()
                midpoint = (p + q) / 2.0
                values.append(float(
                    (np.sum(p * np.log(p / midpoint))
                     + np.sum(q * np.log(q / midpoint))) / 2.0
                ))
        return (float(np.mean(values)), len(values)) if values else (0.0, 0)

    def _mean_pairwise_label_jsd(self, members: list[int]) -> float:
        """Compatibility helper returning only the pairwise JSD mean."""
        return self._pairwise_label_jsd_stats(members)[0]

    def _pooled_label_stats(self, members: list[int]) -> tuple[int, float, float]:
        """Return class coverage, normalized entropy, and largest-class share."""
        distributions = getattr(self, "label_distributions", None)
        if distributions is None or not members:
            return 0, 0.0, 0.0
        pooled = np.zeros_like(np.asarray(distributions[members[0]], dtype=np.float64))
        for node_id in members:
            distribution = np.asarray(distributions[node_id], dtype=np.float64)
            client = self.clients[node_id]
            sample_count = len(
                client.get("train_data", client.get("val_data", {"labels": [0]}))["labels"]
            )
            pooled += distribution * sample_count
        total = float(pooled.sum())
        if total <= 0:
            return 0, 0.0, 0.0
        probabilities = pooled / total
        positive = probabilities[probabilities > 0]
        entropy = float(-np.sum(positive * np.log(positive)))
        normalized_entropy = entropy / math.log(len(probabilities)) if len(probabilities) > 1 else 0.0
        return int(np.count_nonzero(pooled)), normalized_entropy, float(probabilities.max())

    def _tree_edge_label_jsd_stats(self, head_id: int) -> tuple[float, int]:
        """Mean label JSD over the actual parent-child edges of one cluster."""
        distributions = getattr(self, "label_distributions", None)
        if distributions is None:
            return 0.0, 0
        values = []
        epsilon = 1e-10

        def visit(parent_id: int) -> None:
            p = np.asarray(distributions[parent_id], dtype=np.float64) + epsilon
            p /= p.sum()
            for child_id in (self.tree_mapping or {}).get(parent_id, {}).get("children", []):
                if child_id in self.failed_nodes:
                    continue
                q = np.asarray(distributions[child_id], dtype=np.float64) + epsilon
                q /= q.sum()
                midpoint = (p + q) / 2.0
                values.append(float(
                    (np.sum(p * np.log(p / midpoint))
                     + np.sum(q * np.log(q / midpoint))) / 2.0
                ))
                visit(child_id)

        visit(head_id)
        return (float(np.mean(values)), len(values)) if values else (0.0, 0)

    def _evaluate_cluster_diagnostics(
        self, round_id: int, cluster_heads: list[int],
        aggregated_weighted: Dict[int, Optional[WeightedState]],
    ) -> list[dict]:
        """Evaluate CH models on cluster-local validation and optional global test data."""
        if not self.config.enable_cluster_diagnostics:
            return []
        interval = max(1, int(self.config.cluster_diagnostics_interval))
        is_final_round = round_id == int(self.config.rounds) - 1
        if round_id % interval != 0 and not is_final_round:
            return []

        rows = []
        global_dataset = None
        if self.config.cluster_diagnostics_global_test:
            if self.global_test_data is None:
                raise RuntimeError("Global test data is required for cluster diagnostics")
            global_dataset = TensorDataset(
                self.global_test_data["images"], self.global_test_data["labels"]
            )

        for head_id in sorted(cluster_heads):
            weighted_state = aggregated_weighted.get(head_id)
            if weighted_state is None:
                continue
            members = self._cluster_members(head_id)
            validation_sets = [
                TensorDataset(
                    self.clients[node_id]["val_data"]["images"],
                    self.clients[node_id]["val_data"]["labels"],
                )
                for node_id in members
                if len(self.clients[node_id]["val_data"]["labels"]) > 0
            ]
            if not validation_sets:
                continue
            params = finalize_weighted_state(weighted_state)
            local_loss, local_acc, validation_samples = self._evaluate_params_on_dataset(
                params, ConcatDataset(validation_sets)
            )
            global_loss = None
            global_acc = None
            if global_dataset is not None:
                global_loss, global_acc, _ = self._evaluate_params_on_dataset(
                    params, global_dataset
                )
            mean_jsd, label_pair_count = self._pairwise_label_jsd_stats(members)
            tree_edge_jsd, tree_edge_count = self._tree_edge_label_jsd_stats(head_id)
            label_coverage, label_entropy, largest_label_share = self._pooled_label_stats(members)
            rows.append({
                "round_id": round_id,
                "cluster_head": head_id,
                "member_ids": ",".join(str(node_id) for node_id in members),
                "num_members": len(members),
                "train_samples": weighted_state.sample_count,
                "validation_samples": validation_samples,
                "mean_pairwise_label_jsd": mean_jsd,
                "label_pair_count": label_pair_count,
                "mean_tree_edge_label_jsd": tree_edge_jsd,
                "tree_edge_count": tree_edge_count,
                "pooled_label_coverage": label_coverage,
                "pooled_label_entropy": label_entropy,
                "largest_pooled_label_share": largest_label_share,
                "cluster_val_loss": local_loss,
                "cluster_val_accuracy": local_acc,
                "cluster_personalized_loss": (
                    local_loss if getattr(
                        self.config, "enable_cluster_personalized_evaluation", False
                    ) else None
                ),
                "cluster_personalized_accuracy": (
                    local_acc if getattr(
                        self.config, "enable_cluster_personalized_evaluation", False
                    ) else None
                ),
                "client_test_partition_mode": getattr(
                    self.config, "client_test_partition_mode", "global_balanced"
                ),
                "cluster_global_loss": global_loss,
                "cluster_global_accuracy": global_acc,
                "final_global_loss": None,
                "final_global_accuracy": None,
                "distribution_similarity_threshold": (
                    self.config.distribution_similarity_threshold
                ),
                "intra_cluster_rounds": int(
                    getattr(self.config, "intra_cluster_rounds", 1)
                ),
                "clustering_assignment_mode": getattr(
                    self.config, "clustering_assignment_mode", "distributed"
                ),
            })
        return rows

    def _save_experiment_inputs(self, output_dir: str) -> None:
        if self._initial_model_state is None:
            raise RuntimeError("Initial model state is unavailable")
        client_indices = [client['indices'] for client in self.clients]
        node_resources = (
            self.election_results.get('all_nodes_info', [])
            if self.election_results else []
        )
        partition_metadata = {
            "dataset": self.config.dataset,
            "split_policy": resolve_dataset_split_policy(
                self.config.dataset, getattr(self.config, "dataset_split_policy", None)
            ),
            "local_val_seed": getattr(self.config, "har_local_val_seed", 42),
        }
        if self.config.dataset == "har":
            from fl_core.data_loader import har_dataset_fingerprint, tensor_fingerprint
            partition_count = sum(
                len(client["train_data"]["labels"]) for client in self.clients
            )
            local_validation_count = sum(
                len(item["labels"]) for item in getattr(self, "_reserve_val_data", [])
            )
            partition_metadata.update({
                "partition_train_sample_count": partition_count,
                "official_train_sample_count": partition_count + local_validation_count,
                "local_validation_sample_count": local_validation_count,
                "partition_index_space_size": partition_count,
                "global_test_sample_count": (
                    len(self.global_test_data["labels"])
                    if self.global_test_data is not None else None
                ),
                "num_classes": self.config.num_classes,
                "preprocessing": "official_train_mean_std_v1",
                "dataset_fingerprint": har_dataset_fingerprint(self.config.data_dir),
                "global_test_fingerprint": (
                    tensor_fingerprint(
                        self.global_test_data["images"], self.global_test_data["labels"]
                    ) if self.global_test_data is not None else None
                ),
            })
        write_experiment_inputs(
            output_dir=output_dir,
            config=self.config,
            client_indices=client_indices,
            initial_model=self._initial_model_state,
            physical_graph=self.topology,
            node_resources=self.node_resources_input,
            link_profiles=(self.frozen_inputs["link_profiles.json"]
                           if self.frozen_inputs else {
                "bandwidth": self.config.bandwidth,
                "latency": self.config.latency,
                "drop_rate": self.config.drop_rate,
            }),
            failure_trace=(self.frozen_inputs["failure_trace.json"]
                           if self.frozen_inputs else {
                "mode": "stochastic_dynamic_nodes",
                "enabled": self.config.enable_dynamic_nodes,
                "failure_rate": self.config.failure_rate,
                "failure_start_round": self.config.failure_start_round,
                "failure_max_nodes": self.config.failure_max_nodes,
                "failure_pattern": self.config.failure_pattern,
                "seed": self.config.seed,
            }),
            partition_metadata=partition_metadata,
        )

    def _save_completion_status(
        self, output_dir: str, requested_rounds: int,
        completed_rounds: int, status: str, error: str | None = None,
    ) -> None:
        if status == "COMPLETE" and completed_rounds != requested_rounds:
            status = "INCOMPLETE"
        write_completion_status(
            output_dir, requested_rounds, completed_rounds, status, error
        )

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
        optimizer = self.optimizers[client_id]
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
    
    # ==========================================================================
    #                    动态节点管理（Node Failure / Join）
    # ==========================================================================

    def _simulate_failures(self, round_id: int) -> list:
        """模拟节点故障，返回本轮新故障的节点 ID 列表"""
        cfg = self.config
        newly_failed = []
        self._last_recovered_nodes = []
        self._last_link_state_changed = False

        frozen = frozen_failure_events(self.frozen_inputs, round_id, "node_failure")
        trace = self.frozen_inputs and self.frozen_inputs["failure_trace.json"]
        has_explicit_trace = isinstance(trace, list) or (
            isinstance(trace, dict) and "events" in trace
        )
        if has_explicit_trace:
            target_failed, target_edges = failure_state_at_round(
                self.frozen_inputs, round_id
            )
            previous_failed = set(self.failed_nodes)
            previous_edges = set(self.failed_edges)
            newly_failed = sorted(target_failed - previous_failed)
            self._last_recovered_nodes = sorted(previous_failed - target_failed)
            self._last_link_state_changed = target_edges != previous_edges
            self.failed_nodes = set(target_failed)
            self.failed_edges = set(target_edges)
            if newly_failed:
                self.failure_log.append({"round": round_id,
                                         "failed_nodes": newly_failed.copy()})
                self._repair_tree_after_failures(newly_failed)
            if self._last_recovered_nodes:
                for node in self._last_recovered_nodes:
                    if hasattr(self, "client_sample_counts"):
                        self.client_sample_counts[node] = len(self.clients[node]["indices"])
                self.recovery_events.append({
                    "event": "node_recovery", "round": round_id,
                    "failed_nodes": self._last_recovered_nodes.copy(),
                    "time_seconds": 0.0, "method": "failure_trace",
                })
            return newly_failed

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
            'round': getattr(self, '_current_round_id', None),
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
                    n for n in self.topology[i]
                    if n not in excluded
                    and tuple(sorted((i, n))) not in self.failed_edges
                ])

        # 重新选举（仅活跃节点参与）
        # Managers index clients, topology and resource tables by physical node
        # id.  Preserve the full index space and isolate unavailable nodes in
        # ``modified_topology``; compacting this list would renumber nodes.
        clients = [{'id': i} for i in range(self.config.num_clients)]
        election_manager = ElectionManager(
            clients=clients,
            topology=modified_topology,
            base_trigger_time=self.config.election_base_trigger_time,
            trigger_factor=self.config.election_trigger_factor,
            seed=self.config.seed,
            resource_params=self._frozen_election_resources(),
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
            'round': getattr(self, '_current_round_id', None),
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
            'global_losses': [],
            'global_accuracies': [],
            'final_client_accuracies': [],
            'training_time': 0.0,
            'failure_log': [],      # 动态节点故障日志
            'join_log': [],         # 动态节点加入日志
            'cluster_diagnostics': [],
            'global_sync_rounds': [],
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
        self._save_experiment_inputs(output_dir)
        self._save_completion_status(
            output_dir, num_rounds, 0, status="RUNNING"
        )

        # 动态节点模式：打印配置信息
        if self.config.enable_dynamic_nodes:
            print(f"\n[动态节点模式]")
            print(f"  故障率: {self.config.failure_rate}, 起始轮: {self.config.failure_start_round}")
            print(f"  最大故障节点: {self.config.failure_max_nodes}, 模式: {self.config.failure_pattern}")
            print(f"  恢复策略: {self.config.recovery_strategy}")

        method_name = (self.config.method_name or
                       ("drift" if self.config.resource_awareness
                        else "drift_no_resource_awareness"))
        monitor = Monitor()
        monitor.configure(run_name, method_name, self.config.seed)
        unified_comm_events = []

        total_start_time = time.time()
        latest_global_params = None

        for round_id in range(num_rounds):
            self._current_round_id = round_id
            round_start_time = time.time()
            round_losses = []
            global_loss = None
            global_acc = None
            aggregated_nodes_this_round: set[int] = set()

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
            recovered_nodes = list(self._last_recovered_nodes)
            link_state_changed = self._last_link_state_changed
            if recovered_nodes:
                print(f"\n✅ 节点恢复: {recovered_nodes}")
                if use_ray and latest_global_params is not None:
                    recovered_actors = [
                        self.client_actors[node] for node in recovered_nodes
                        if self.client_actors[node] is not None
                    ]
                    if recovered_actors:
                        restored = {
                            key: value.detach().cpu() for key, value in latest_global_params.items()
                        }
                        ray.get([
                            actor.set_model_params.remote(restored)
                            for actor in recovered_actors
                        ])
            if recovered_nodes or link_state_changed:
                reason = "node_recovery" if recovered_nodes else "link_state_change"
                if link_state_changed:
                    print(f"\n⚠️  链路状态变化，当前故障链路: {sorted(self.failed_edges)}")
                self._reelect_and_recluster(reason=reason)
                if use_ray:
                    live_actors, live_ids = self._get_live_ray_actors()
                    new_lr = self._calc_learning_rate(round_id)
                    ray.get([a.set_learning_rate.remote(new_lr) for a in live_actors])
            if newly_failed:
                print(f"\n⚠️  节点故障: {newly_failed}")
                print(f"  存活节点: {len(self.active_nodes - self.failed_nodes)}/{self.total_nodes}")
                # 刷新存活 actor 列表（故障节点不再参与后续操作）
                if use_ray:
                    live_actors, live_ids = self._get_live_ray_actors()
                # 更新 cluster_heads
                self._update_cluster_heads_from_tree()

                has_frozen_trace = bool(
                    self.frozen_inputs
                    and isinstance(self.frozen_inputs.get("failure_trace.json"), (list, dict))
                )
                if has_frozen_trace:
                    self._reelect_and_recluster(reason="frozen_node_failure")
                    if use_ray:
                        live_actors, live_ids = self._get_live_ray_actors()

                # reelect_recluster: 簇首故障时触发完整重新选举+聚类
                elif self.config.recovery_strategy == "reelect_recluster":
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
                train_results = []
                limit = max(1, self.config.max_concurrent_gpu_clients)
                for start in range(0, len(live_actors), limit):
                    batch = live_actors[start:start + limit]
                    train_results.extend(ray.get([
                        actor.local_train.remote(round_id) for actor in batch
                    ]))
                round_losses = [r['train_loss'] for r in train_results]
                print(f"  {len(live_ids)} 个存活客户端训练完成")
                if self.failed_nodes:
                    print(f"  已故障节点（跳过）: {sorted(self.failed_nodes)}")
            else:
                for client_id in range(self.config.num_clients):
                    if client_id in self.failed_nodes:
                        continue
                    loss = self._train_single_client(client_id, round_id)
                    round_losses.append(loss)

            # ---- 阶段2-7: 分层聚合 ----
            link_profile = (self.frozen_inputs or {}).get("link_profiles.json", {})
            link_default = link_profile.get("default", link_profile)
            bandwidth_mbps = link_default.get(
                "bandwidth_mbps", link_default.get("default_bandwidth_mbps")
            )
            if bandwidth_mbps is None and self.config.bandwidth is not None:
                bandwidth_mbps = float(self.config.bandwidth) * 8.0
            latency_ms = link_default.get(
                "latency_ms", link_default.get("default_latency_ms", self.config.latency)
            )
            comm = CommStats(
                round_id, bandwidth_mbps=bandwidth_mbps,
                latency_ms=latency_ms,
                drop_rate=float(link_default.get("drop_rate", self.config.drop_rate)),
            )
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
                    if live_params:
                        comm.model_bytes = compute_model_bytes(live_params[0])
                else:
                    wrapped_models = self.models

                agg_device = torch.device('cpu')

                print("\n[阶段2] 簇内分层聚合")
                aggregated_weighted = hierarchical_weighted_reduce(
                    wrapped_models,
                    self.tree_mapping,
                    cluster_heads,
                    comm=comm,
                    skip_nodes=self.failed_nodes,
                )
                print(f"  簇内分层聚合完成，共 {len(cluster_heads)} 个簇")
                print(f"  📡 簇内上行: {comm.intra_cluster_msgs} 条")

                intra_cluster_rounds = max(
                    1, int(getattr(self.config, "intra_cluster_rounds", 1))
                )
                for intra_round_id in range(1, intra_cluster_rounds):
                    print(
                        f"\n[阶段2.{intra_round_id}] 簇模型下发并执行额外簇内训练 "
                        f"({intra_round_id + 1}/{intra_cluster_rounds})"
                    )
                    cluster_params = {
                        head_id: finalize_weighted_state(weighted_state)
                        for head_id, weighted_state in aggregated_weighted.items()
                        if weighted_state is not None
                    }
                    distribute_returned_models_to_clusters(
                        returned_models=cluster_params,
                        tree_mapping=self.tree_mapping,
                        models=wrapped_models,
                        cluster_heads=cluster_heads,
                        skip_nodes=self.failed_nodes,
                        comm=comm,
                        allow_unreachable=bool(self.failed_edges),
                    )

                    if use_ray:
                        live_actors, live_ids = self._get_live_ray_actors()
                        ray.get([
                            actor.set_model_params.remote({
                                key: value.detach().cpu()
                                for key, value in wrapped_models[node_id].state_dict().items()
                            })
                            for actor, node_id in zip(live_actors, live_ids)
                        ])
                        extra_train_results = []
                        limit = max(1, self.config.max_concurrent_gpu_clients)
                        for start in range(0, len(live_actors), limit):
                            batch = live_actors[start:start + limit]
                            extra_train_results.extend(ray.get([
                                actor.local_train.remote(round_id, intra_round_id)
                                for actor in batch
                            ]))
                        round_losses.extend(
                            result['train_loss'] for result in extra_train_results
                        )
                        live_params = ray.get([
                            actor.get_model_params.remote() for actor in live_actors
                        ])
                        all_params = [None] * self.config.num_clients
                        for node_id, params in zip(live_ids, live_params):
                            all_params[node_id] = params
                        wrapped_models = [
                            ModelWrapper(params) if params is not None else None
                            for params in all_params
                        ]
                    else:
                        for client_id in range(self.config.num_clients):
                            if client_id in self.failed_nodes:
                                continue
                            round_losses.append(
                                self._train_single_client(client_id, round_id)
                            )

                    aggregated_weighted = hierarchical_weighted_reduce(
                        wrapped_models,
                        self.tree_mapping,
                        cluster_heads,
                        comm=comm,
                        skip_nodes=self.failed_nodes,
                    )
                    print(
                        f"  第 {intra_round_id + 1}/{intra_cluster_rounds} 次簇内聚合完成"
                    )

                round_cluster_diagnostics = self._evaluate_cluster_diagnostics(
                    round_id, cluster_heads, aggregated_weighted
                )

                sync_interval = max(
                    1, int(getattr(self.config, "global_sync_interval", 1))
                )
                perform_global_sync = global_sync_due(
                    round_id, num_rounds, sync_interval
                )
                comm.global_sync_performed = perform_global_sync
                message_queue = {}
                path_stacks = {}

                if perform_global_sync:
                    print("\n[阶段3] 簇首沿选举路由上传到 GEA")
                    cluster_sizes, routing_tables = build_cluster_info(
                        self.tree_mapping,
                        self.clustering_results,
                        self.election_results,
                    )
                    for head_id in cluster_heads:
                        forward_model_to_best_client(
                            source_cluster_head=head_id,
                            weighted_state=aggregated_weighted[head_id],
                            routing_tables=routing_tables,
                            best_client_id=best_client_id,
                            message_queue=message_queue,
                            comm=comm,
                        )
                    print(f"  GEA: Node {best_client_id}; 簇样本数: {cluster_sizes}")
                    print(f"  📡 簇间上行: {comm.inter_cluster_forward_hops} 跳")

                    print("\n[阶段4] GEA 按真实样本数聚合所有簇模型")
                    final_model, path_stacks = aggregate_models_at_best_client(
                        best_client_id=best_client_id,
                        models=wrapped_models,
                        message_queue=message_queue,
                        device=agg_device,
                    )
                    latest_global_params = final_model
                    results['global_sync_rounds'].append(round_id)
                def collect_cluster_nodes(node_id: int) -> None:
                    if node_id in self.failed_nodes or node_id in aggregated_nodes_this_round:
                        return
                    aggregated_nodes_this_round.add(node_id)
                    for child_id in (self.tree_mapping or {}).get(node_id, {}).get("children", []):
                        collect_cluster_nodes(child_id)

                if perform_global_sync:
                    for reachable_head in path_stacks:
                        collect_cluster_nodes(reachable_head)
                else:
                    aggregated_nodes_this_round.update(
                        self.active_nodes - self.failed_nodes
                    )

                if perform_global_sync:
                    print("\n[阶段5] 全局模型沿上传路径原路返回簇首")
                    return_queue = {}
                    return_model_to_cluster_heads(
                        final_model=final_model,
                        path_stacks=path_stacks,
                        return_queue=return_queue,
                        comm=comm,
                    )
                    returned_models = {
                        head_id: return_queue[head_id][-1]['model_dict']
                        for head_id in cluster_heads
                        if head_id in return_queue and return_queue[head_id]
                    }
                    print(f"  📡 簇间下行: {comm.inter_cluster_return_hops} 跳")
                else:
                    print(
                        f"\n[阶段3-5] 跳过全局同步 "
                        f"(间隔={sync_interval})"
                    )
                    returned_models = {
                        head_id: finalize_weighted_state(weighted_state)
                        for head_id, weighted_state in aggregated_weighted.items()
                        if weighted_state is not None
                    }

                model_scope = "全局" if perform_global_sync else "簇"
                print(f"\n[阶段6] 簇首沿原簇树下发{model_scope}模型")
                distribute_returned_models_to_clusters(
                    returned_models=returned_models,
                    tree_mapping=self.tree_mapping,
                    models=wrapped_models,
                    cluster_heads=cluster_heads,
                    skip_nodes=self.failed_nodes,
                    comm=comm,
                    allow_unreachable=bool(self.failed_edges),
                )
                if use_ray:
                    live_actors, live_ids = self._get_live_ray_actors()
                    if self.failed_edges:
                        ray.get([
                            actor.set_model_params.remote({
                                key: value.detach().cpu()
                                for key, value in wrapped_models[node_id].state_dict().items()
                            })
                            for actor, node_id in zip(live_actors, live_ids)
                        ])
                    elif perform_global_sync:
                        final_model_cpu = {k: v.cpu() for k, v in final_model.items()}
                        ray.get([a.set_model_params.remote(final_model_cpu) for a in live_actors])
                    else:
                        ray.get([
                            actor.set_model_params.remote({
                                key: value.detach().cpu()
                                for key, value in wrapped_models[node_id].state_dict().items()
                            })
                            for actor, node_id in zip(live_actors, live_ids)
                        ])
                print(f"  📡 簇内下行: {comm.intra_cluster_dist_msgs} 条")
                print(f"  📡 本轮总通信: {comm.total_msgs} 条")

                if perform_global_sync:
                    print("\n[阶段7] 在完整全局测试集上评估一次")
                    global_loss, global_acc = self._evaluate_global_params(final_model)
                    results['global_losses'].append(global_loss)
                    results['global_accuracies'].append(global_acc)
                    print(f"  全局测试损失: {global_loss:.4f}, 全局测试准确率: {global_acc:.4f}")
                for row in round_cluster_diagnostics:
                    row["global_sync_performed"] = perform_global_sync
                    row["final_global_loss"] = global_loss if perform_global_sync else np.nan
                    row["final_global_accuracy"] = global_acc if perform_global_sync else np.nan
                results['cluster_diagnostics'].extend(round_cluster_diagnostics)

            elif use_ray:
                # ---- 阶段2-4: 扁平聚合（消融实验：无聚类模式） ----
                print("\n[阶段2] 扁平聚合（存活客户端 → 聚合节点）")
                live_actors, live_ids = self._get_live_ray_actors()
                aggregated_nodes_this_round = set(live_ids)
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
                latest_global_params = final_params
                comm.flat_upload_msgs = sum(client_id != best_id for client_id in live_ids)
                if all_params:
                    comm.model_bytes = compute_model_bytes(all_params[0])
                for client_id in live_ids:
                    if client_id != best_id:
                        comm.record_transfer(
                            client_id, best_id, "MODEL_UPLOAD", comm.model_bytes,
                            phase="MODEL_UPLOAD", flow_id="fedavg:up",
                        )
                print(f"  聚合了 {len(all_params)} 个客户端的模型 (聚合节点: Node {best_id})")
                print(f"  📡 通信: 上传 {comm.flat_upload_msgs} 条")

                print("\n[阶段3] 分发聚合模型到所有存活节点")
                final_params_cpu = {k: v.cpu() for k, v in final_params.items()}
                ray.get([a.set_model_params.remote(final_params_cpu) for a in live_actors])
                comm.flat_download_msgs = sum(client_id != best_id for client_id in live_ids)
                for client_id in live_ids:
                    if client_id != best_id:
                        comm.record_transfer(
                            best_id, client_id, "GLOBAL_MODEL", comm.model_bytes,
                            phase="MODEL_DISTRIBUTION", flow_id="fedavg:down",
                        )
                print(f"  📡 通信: 下载 {comm.flat_download_msgs} 条")

                print(f"  📡 本轮总通信: {comm.total_msgs} 条消息, {comm.total_bytes:,} 字节")

                print("\n[阶段4] 在完整全局测试集上评估一次")
                global_loss, global_acc = self._evaluate_global_params(final_params)
                results['global_losses'].append(global_loss)
                results['global_accuracies'].append(global_acc)
                print(f"  全局测试损失: {global_loss:.4f}, 全局测试准确率: {global_acc:.4f}")

            avg_loss = np.mean(round_losses)
            results['round_losses'].append(avg_loss)
            if global_acc is not None:
                results['round_accuracies'].append(global_acc)

            round_time = time.time() - round_start_time
            print(f"\n  本轮平均训练损失: {avg_loss:.4f}")
            if global_acc is not None:
                print(f"  本轮全局准确率: {results['global_accuracies'][-1]:.4f}")
            elif use_hierarchical:
                print("  本轮未执行全局同步/全局评估")
            print(f"  本轮耗时: {round_time:.2f}s")

            # 记录本轮指标（使用分层聚合后的全局准确率作为 val_acc）
            recorded_global_acc = results['global_accuracies'][-1] if results['global_accuracies'] else 0.0
            recorded_global_loss = results['global_losses'][-1] if results['global_losses'] else 0.0
            live_count = len(self.active_nodes - self.failed_nodes)
            failed_count = len(self.failed_nodes)
            monitor.record_round(
                round_id=round_id,
                avg_train_loss=avg_loss,
                val_loss=recorded_global_loss,
                val_acc=recorded_global_acc,
                selected_count=live_count,
                dropped_count=failed_count,
                comm_success_count=comm.total_msgs if use_ray else 0,
                comm_failure_count=0,
                comm_timeout_count=0,
                broadcast_success_count=(
                    (comm.inter_cluster_return_hops + comm.intra_cluster_dist_msgs
                     if use_hierarchical else comm.flat_download_msgs)
                    if use_ray else 0
                ),
                broadcast_failure_count=0,
                upload_success_count=(
                    (comm.intra_cluster_msgs + comm.inter_cluster_forward_hops
                     if use_hierarchical else comm.flat_upload_msgs)
                    if use_ray else 0
                ),
                upload_failure_count=0,
                upload_storage_blocked_count=0,
                gossip_success_count=0,
                gossip_failure_count=0,
                round_wall_clock_seconds=round_time,
            )
            # 保存每轮通信统计
            if 'round_comm_stats' not in results:
                results['round_comm_stats'] = []
            results['round_comm_stats'].append(comm.to_dict())
            for event_index, event in enumerate(comm.events):
                unified_comm_events.append({
                    "experiment_id": run_name,
                    "method": method_name,
                    "seed": self.config.seed,
                    "timestamp_start": None,
                    "timestamp_end": None,
                    "message_id": f"r{round_id}-m{event_index}",
                    **event,
                })

            live_nodes = sorted(self.active_nodes - self.failed_nodes)
            total_samples = sum(len(client["indices"]) for client in self.clients)
            live_samples = sum(
                len(self.clients[node_id]["indices"]) for node_id in live_nodes
            )
            roles = {}
            parents = {}
            children_counts = {}
            best_id = (self.election_results or {}).get("best_client_id", -1)
            for node_id in range(self.config.num_clients):
                info = (self.tree_mapping or {}).get(node_id, {})
                parent = info.get("parent")
                children = [c for c in info.get("children", []) if c in live_nodes]
                parents[node_id] = parent
                children_counts[node_id] = len(children)
                if node_id == best_id:
                    roles[node_id] = "ROOT"
                elif parent is None and node_id in live_nodes:
                    roles[node_id] = "CLUSTER_HEAD"
                elif children:
                    roles[node_id] = "INTERMEDIATE"
                else:
                    roles[node_id] = "CLIENT"
            # Record semantic aggregation inputs separately from communication
            # messages: each live node contributes its own model, and the GEA
            # also consumes the zero-hop model from its local cluster head.
            aggregation_inputs_by_phase = {
                "INTRA_CLUSTER_UPLOAD": {},
                "INTER_CLUSTER_UPLOAD": {},
            }

            def count_tree_parts(node_id: int) -> int:
                info = (self.tree_mapping or {}).get(node_id, {})
                # In the Ray path model objects live inside actors and
                # ``self.models`` is intentionally None.  A live participant
                # nevertheless contributes exactly one local weighted state.
                own = int(node_id in live_nodes)
                parts = own
                for child_id in info.get("children", []):
                    child_parts = count_tree_parts(child_id)
                    if child_parts > 0:
                        parts += 1
                if parts > 1:
                    aggregation_inputs_by_phase["INTRA_CLUSTER_UPLOAD"][str(node_id)] = parts
                return parts

            for head_id in cluster_heads:
                count_tree_parts(head_id)
            gea_inputs = len(message_queue.get(best_id, []))
            if gea_inputs > 1:
                aggregation_inputs_by_phase["INTER_CLUSTER_UPLOAD"][str(best_id)] = gea_inputs
            aggregation_operation_ids_by_phase = {
                phase: {
                    node: f"r{round_id}:aggregate:{phase.lower()}:{node}"
                    for node in nodes
                }
                for phase, nodes in aggregation_inputs_by_phase.items()
            }
            monitor.record_round_context(
                experiment_id=run_name, method=method_name, seed=self.config.seed,
                round=round_id, node_ids=list(range(self.config.num_clients)),
                roles=roles, parents=parents, children_counts=children_counts,
                active_nodes=live_nodes, failed_nodes=sorted(self.failed_nodes),
                eligible_node_ratio=len(live_nodes) / max(1, self.config.num_clients),
                reachable_node_ratio=(len(aggregated_nodes_this_round)
                                      / max(1, len(live_nodes))),
                eligible_sample_ratio=live_samples / max(1, total_samples),
                aggregated_sample_ratio=(sum(
                    len(self.clients[node_id]["indices"])
                    for node_id in aggregated_nodes_this_round
                ) / max(1, live_samples)),
                aggregation_inputs_by_phase=aggregation_inputs_by_phase,
                aggregation_operation_ids_by_phase=aggregation_operation_ids_by_phase,
                round_completion_ratio=1.0 if global_acc is not None else 0.0,
                round_wall_clock_seconds=round_time,
                round_latency_sim_ms=None,
            )

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

        print(f"  总训练耗时: {results['training_time']:.2f}s")

        if results['global_accuracies']:
            print(f"\n完整全局测试集准确率统计:")
            print(f"  最终准确率: {results['global_accuracies'][-1]:.4f}")
            print(f"  最高准确率: {np.max(results['global_accuracies']):.4f}")
            print(f"  最低准确率: {np.min(results['global_accuracies']):.4f}")

        if self.config.save_model and latest_global_params is not None:
            model_path = os.path.join(output_dir, f"{run_name}_final_model.pt")
            torch.save(
                {key: value.detach().cpu() for key, value in latest_global_params.items()},
                model_path,
            )
            print(f"  最终全局模型已保存: {model_path}")

        # ---- 保存指标并生成图表 ----
        print(f"\n{'='*60}")
        print("保存训练指标和图表")
        print(f"{'='*60}")

        monitor.dump(output_dir, run_name)
        if results['cluster_diagnostics']:
            import json as _json
            import pandas as pd
            cluster_df = pd.DataFrame(results['cluster_diagnostics'])
            cluster_csv = os.path.join(output_dir, "cluster_round_metrics.csv")
            cluster_df.to_csv(cluster_csv, index=False)
            per_round = []
            for diagnostic_round, frame in cluster_df.groupby("round_id"):
                weights = frame["validation_samples"].astype(float)
                weight_total = max(1.0, float(weights.sum()))
                pair_weights = frame["label_pair_count"].astype(float)
                pair_total = float(pair_weights.sum())
                edge_weights = frame["tree_edge_count"].astype(float)
                edge_total = float(edge_weights.sum())
                global_accuracy = frame["cluster_global_accuracy"]
                per_round.append({
                    "round_id": int(diagnostic_round),
                    "num_clusters": int(len(frame)),
                    "mean_cluster_size": float(frame["num_members"].mean()),
                    "singleton_cluster_ratio": float(
                        (frame["num_members"] == 1).mean()
                    ),
                    "weighted_cluster_val_accuracy": float(
                        (frame["cluster_val_accuracy"] * weights).sum() / weight_total
                    ),
                    "weighted_cluster_personalized_accuracy": (
                        float(
                            (frame["cluster_personalized_accuracy"] * weights).sum()
                            / weight_total
                        )
                        if frame["cluster_personalized_accuracy"].notna().any() else None
                    ),
                    "pair_weighted_mean_label_jsd": (
                        float((frame["mean_pairwise_label_jsd"] * pair_weights).sum()
                              / pair_total)
                        if pair_total > 0 else None
                    ),
                    "edge_weighted_mean_label_jsd": (
                        float((frame["mean_tree_edge_label_jsd"] * edge_weights).sum()
                              / edge_total)
                        if edge_total > 0 else None
                    ),
                    "sample_weighted_pooled_label_coverage": float(
                        (frame["pooled_label_coverage"] * frame["train_samples"]).sum()
                        / max(1.0, float(frame["train_samples"].sum()))
                    ),
                    "sample_weighted_pooled_label_entropy": float(
                        (frame["pooled_label_entropy"] * frame["train_samples"]).sum()
                        / max(1.0, float(frame["train_samples"].sum()))
                    ),
                    "sample_weighted_largest_label_share": float(
                        (frame["largest_pooled_label_share"] * frame["train_samples"]).sum()
                        / max(1.0, float(frame["train_samples"].sum()))
                    ),
                    "mean_cluster_global_accuracy": (
                        float(global_accuracy.mean())
                        if global_accuracy.notna().any() else None
                    ),
                    "sample_weighted_cluster_global_accuracy": (
                        float((global_accuracy * frame["train_samples"]).sum()
                              / max(1.0, float(frame["train_samples"].sum())))
                        if global_accuracy.notna().any() else None
                    ),
                    "final_global_accuracy": float(
                        frame["final_global_accuracy"].iloc[0]
                    ),
                })
            with open(os.path.join(output_dir, "clustering_summary.json"), "w") as handle:
                _json.dump({
                    "definition": "pairwise-neighbor-label-JSD acceptance constraint",
                    "threshold": self.config.distribution_similarity_threshold,
                    "intra_cluster_rounds": int(
                        getattr(self.config, "intra_cluster_rounds", 1)
                    ),
                    "clustering_assignment_mode": getattr(
                        self.config, "clustering_assignment_mode", "distributed"
                    ),
                    "diagnostic_interval": self.config.cluster_diagnostics_interval,
                    "global_test_enabled": self.config.cluster_diagnostics_global_test,
                    "personalized_evaluation_enabled": getattr(
                        self.config, "enable_cluster_personalized_evaluation", False
                    ),
                    "client_test_partition_mode": getattr(
                        self.config, "client_test_partition_mode", "global_balanced"
                    ),
                    "records": len(results['cluster_diagnostics']),
                    "per_round": per_round,
                }, handle, indent=2, ensure_ascii=False)
            print(f"  簇级诊断已保存: {cluster_csv}")
        topology_events = []
        for failure in self.failure_log:
            topology_events.append({
                "experiment_id": run_name, "method": method_name,
                "seed": self.config.seed, "round": failure.get("round"),
                "event_time": None, "event_type": "NODE_FAILURE",
                "target_type": "NODE", "target_id": failure.get("failed_nodes"),
                "detected_time": None, "repair_start_time": None,
                "repair_end_time": None, "repair_time_wall_clock_seconds": None,
                "affected_nodes": len(failure.get("failed_nodes", [])),
                "control_messages": None, "control_bytes": None,
            })
        for join in self.join_log:
            topology_events.append({
                "experiment_id": run_name, "method": method_name,
                "seed": self.config.seed, "round": join.get("round"),
                "event_time": None, "event_type": "NODE_JOIN",
                "target_type": "NODE", "target_id": join.get("joined_nodes"),
                "detected_time": None, "repair_start_time": None,
                "repair_end_time": None, "repair_time_wall_clock_seconds": None,
                "affected_nodes": len(join.get("joined_nodes", [])),
                "control_messages": None, "control_bytes": None,
            })
        frozen_trace = ((self.frozen_inputs or {}).get("failure_trace.json", {}))
        frozen_events = (frozen_trace if isinstance(frozen_trace, list)
                         else frozen_trace.get("events", []))
        for event in frozen_events:
            event_type = str(event.get("event_type", "")).strip().lower()
            if event_type not in {"link_failure", "link_recovery"}:
                continue
            edges = event.get("edges", [])
            topology_events.append({
                "experiment_id": run_name, "method": method_name,
                "seed": self.config.seed, "round": event.get("round"),
                "event_time": None, "event_type": event_type.upper(),
                "target_type": "LINK", "target_id": edges,
                "detected_time": None, "repair_start_time": None,
                "repair_end_time": None,
                "repair_time_wall_clock_seconds": 0.0,
                "affected_nodes": len({node for edge in edges for node in edge}),
                "control_messages": None, "control_bytes": None,
            })
        for index, event in enumerate(getattr(self, "recovery_events", [])):
            topology_events.append({
                "experiment_id": run_name, "method": method_name,
                "seed": self.config.seed, "round": event.get("round"),
                "event_time": None,
                "event_type": str(event.get("event", "TOPOLOGY_REPAIR")).upper(),
                "target_type": "NODE", "target_id": event.get("failed_nodes"),
                "detected_time": None, "repair_start_time": None,
                "repair_end_time": None,
                "repair_time_wall_clock_seconds": event.get("time_seconds"),
                "affected_nodes": len(event.get("failed_nodes", [])),
                "control_messages": event.get("control_messages"),
                "control_bytes": event.get("control_bytes"),
            })
        dump_system_metrics(
            output_dir, unified_comm_events, monitor.round_contexts, topology_events
        )
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
            'global_sync_interval': int(
                getattr(self.config, 'global_sync_interval', 1)
            ),
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

        self._save_completion_status(
            output_dir=output_dir,
            requested_rounds=num_rounds,
            completed_rounds=len(monitor.round_metrics),
            status="COMPLETE",
        )

        return results
