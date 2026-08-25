"""
客户端Actor模块：定义分布式客户端的行为

该模块实现了联邦学习中的客户端角色，使用Ray框架实现分布式计算。

核心功能：
1. 客户端初始化：加载模型、数据、优化器
2. 本地训练：执行本地SGD更新
3. 模型参数管理：获取/设置模型参数
4. 模型评估：在本地验证集上评估模型性能
5. 显存管理：清理GPU缓存

关键概念：
- Ray Actor：分布式对象，每个客户端是一个独立的Actor
- @ray.remote：装饰器，将类标记为Ray Actor
- num_cpus/num_gpus：为Actor分配的资源比例
- .remote()：异步调用Actor方法，返回ObjectRef
"""

import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from typing import List, Dict, Optional
import torchvision.transforms as T

@ray.remote(num_cpus=0.05, max_concurrency=4)
class NewClientActor:
    """
    联邦学习客户端Actor类
    
    每个客户端负责：
    1. 维护本地模型副本
    2. 在本地数据上训练模型
    3. 评估模型性能
    4. 与服务器/其他客户端交换模型参数
    
    参数说明：
        client_id: 客户端唯一标识
        model: 模型实例（ResNet18等）
        train_dataset: 本地训练数据集（Subset）
        val_dataset: 验证数据集
        batch_size: 训练批次大小
        lr: 学习率
        weight_decay: 权重衰减
        device: 计算设备（cpu/cuda）
    """
    
    def __init__(self, client_id: int, model, train_dataset, val_dataset,
                 batch_size: int, lr: float, weight_decay: float,
                 device=None, neighbors: Optional[List[int]] = None,
                 use_image_transform: bool = True, seed: int = 0):
        # 客户端基本信息
        self.client_id = client_id
        self.seed = seed
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model = model
        self._model_on_gpu = False
        self.batch_size = batch_size

        # 邻居管理（去中心化关键）
        self.neighbors = neighbors if neighbors is not None else []
        self.neighbor_handles = {}  # 存储邻居的Ray Actor句柄
        self.received_params_buffer = []  # 接收到的模型参数缓冲

        # 性能跟踪（用于选择性邻居通信）
        self.performance_history = {}
        self.trusted_clients = set()
        
        # 创建数据加载器
        # DataLoader负责将数据集分成批次，支持多线程加载
        loader_generator = torch.Generator().manual_seed(seed)
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            generator=loader_generator,
        )
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False  # 验证时不需要打乱
        )
        
        # 优化器参数（每轮 local_train 内重建，对齐 FedAvg/Gossip）
        self.lr = lr
        self.weight_decay = weight_decay
        
        # 损失函数：交叉熵损失（用于分类任务）
        self.criterion = nn.CrossEntropyLoss()
        
        # 本地训练轮数（每轮联邦学习中的本地训练次数）
        self.local_epochs = 5

        # 数据增强（仅用于图像数据，HAR等向量数据跳过）
        if use_image_transform:
            self.train_transform = T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
            ])
        else:
            self.train_transform = None
    
    def _ensure_cuda(self):
        """延迟CUDA初始化：模型首次使用时才迁移到GPU"""
        if not self._model_on_gpu and self.device.type == 'cuda':
            self.model = self.model.to(self.device)
            self._model_on_gpu = True

    def set_local_epochs(self, epochs: int):
        """
        设置本地训练轮数

        参数：
            epochs: 本地训练轮数
        """
        self.local_epochs = epochs

    def set_learning_rate(self, lr: float):
        """
        动态设置学习率（每轮训练开始前由 runtime 调用）
        """
        self.lr = lr
    
    def get_model_params(self):
        """
        获取当前模型参数
        
        返回：
            字典，key为参数名称，value为参数张量（CPU上）
            
        注意：
            使用detach()分离计算图，避免内存泄漏
            使用cpu()将参数移动到CPU，便于网络传输
        """
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
    
    def set_model_params(self, params):
        """
        设置模型参数
        
        参数：
            params: 模型参数字典
            
        原理：
            load_state_dict()是PyTorch的标准方法，用于加载模型权重
        """
        self.model.load_state_dict(params)
    
    def local_train(self, round_id: int):
        """
        执行本地训练

        参数：
            round_id: 当前联邦学习轮数

        返回：
            字典，包含客户端ID、轮数、训练损失、样本数

        训练流程：
        1. 创建新的优化器（对齐 FedAvg/Gossip，每轮重建 adam 状态）
        2. 设置模型为训练模式
        3. 遍历多个本地epoch，执行 SGD 更新
        4. 返回训练统计信息
        """
        self._ensure_cuda()
        torch.manual_seed(self.seed + round_id)
        # 每轮重建优化器（对齐 FedAvg/Gossip）
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.model.train()  # 设置模型为训练模式
        total_loss = 0.0    # 累计损失
        total_samples = 0   # 累计样本数

        for epoch in range(self.local_epochs):
            for xb, yb in self.train_loader:
                # 将数据移动到指定设备
                xb, yb = xb.to(self.device), yb.to(self.device)

                # 训练步骤：
                optimizer.zero_grad()  # 清零梯度（防止累加）
                if self.train_transform is not None:
                    xb = self.train_transform(xb)  # 数据增强（仅图像）
                output = self.model(xb)     # 前向传播：输入→模型→输出
                loss = self.criterion(output, yb)  # 计算损失
                loss.backward()             # 反向传播：计算梯度
                optimizer.step()       # 更新参数

                # 累计统计（在释放前获取值）
                total_loss += loss.item() * xb.size(0)
                total_samples += xb.size(0)

                # 显存优化：立即释放临时变量
                del output, loss
                torch.cuda.empty_cache()

        # 释放本轮优化器
        optimizer.zero_grad(set_to_none=True)
        del optimizer
        torch.cuda.empty_cache()  # 清理所有未使用的显存
        
        # 计算平均损失
        avg_loss = total_loss / total_samples
        
        return {
            'client_id': self.client_id,
            'round_id': round_id,
            'train_loss': avg_loss,
            'num_samples': total_samples
        }
    
    def evaluate(self):
        """
        在本地验证集上评估模型
        
        返回：
            字典，包含客户端ID、验证损失、验证准确率
            
        评估流程：
        1. 设置模型为评估模式（禁用Dropout、固定BatchNorm）
        2. 使用torch.no_grad()禁用梯度计算（节省内存和时间）
        3. 遍历验证集，计算损失和准确率
        """
        self._ensure_cuda()
        self.model.eval()  # 设置模型为评估模式
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():  # 禁用梯度计算
            for xb, yb in self.val_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                output = self.model(xb)
                loss = self.criterion(output, yb)
                
                # 计算损失
                total_loss += loss.item() * xb.size(0)
                
                # 计算准确率
                _, predicted = torch.max(output.data, 1)  # 获取预测类别
                total += yb.size(0)
                correct += (predicted == yb).sum().item()
                
                # 清理临时变量
                del output, loss
        
        # 计算平均损失和准确率
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return {
            'client_id': self.client_id,
            'val_loss': avg_loss,
            'val_acc': accuracy,
            'num_samples': total,
        }
    
    def clear_cache(self):
        """
        清理GPU显存缓存

        原理：
            torch.cuda.empty_cache()释放PyTorch未使用的缓存显存
            torch.cuda.synchronize()等待所有CUDA操作完成
        """
        if torch.cuda.is_available() and self._model_on_gpu:
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.device)
        return self.client_id

    def set_neighbors(self, neighbors: List[int]):
        """
        设置静态邻居列表

        参数：
            neighbors: 邻居客户端ID列表
        """
        self.neighbors = neighbors
        print(f"Client {self.client_id}: neighbors set to {neighbors}")

    def register_neighbor_handle(self, neighbor_id: int, handle):
        """
        注册邻居的Ray Actor句柄

        参数：
            neighbor_id: 邻居客户端ID
            handle: Ray Actor句柄
        """
        self.neighbor_handles[neighbor_id] = handle

    def select_neighbors_by_performance(self, k: int = 3) -> List[int]:
        """
        根据历史性能选择最优邻居

        参数：
            k: 选择前k个最优邻居

        返回：
            选中的邻居ID列表
        """
        if not self.performance_history:
            return self.neighbors[:k] if self.neighbors else []

        sorted_clients = sorted(
            self.performance_history.items(),
            key=lambda x: x[1].get('val_acc', 0),
            reverse=True
        )
        selected = [cid for cid, _ in sorted_clients[:k] if cid in self.neighbors]
        return selected if selected else self.neighbors[:k]

    def select_neighbors_by_trust(self, trust_scores: Dict[int, float], k: int = 3) -> List[int]:
        """
        根据信任度选择邻居

        参数：
            trust_scores: 信任度字典 {client_id: trust_score}
            k: 选择前k个最可信邻居

        返回：
            选中的邻居ID列表
        """
        if not trust_scores:
            return self.neighbors[:k]

        neighbors_with_trust = [
            (cid, trust_scores.get(cid, 0)) for cid in self.neighbors
        ]
        sorted_neighbors = sorted(neighbors_with_trust, key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in sorted_neighbors[:k]]

    def send_model_to_neighbors(self, params: Dict) -> List[Dict]:
        """
        发送模型参数给所有已注册的邻居

        参数：
            params: 模型参数字典

        返回：
            发送结果列表
        """
        results = []
        for neighbor_id, handle in self.neighbor_handles.items():
            try:
                result = handle.receive_model_from_peer.remote(
                    sender_id=self.client_id,
                    params=params
                )
                results.append({'neighbor_id': neighbor_id, 'status': 'sent'})
            except Exception as e:
                results.append({'neighbor_id': neighbor_id, 'status': 'failed', 'error': str(e)})
        return results

    def receive_model_from_peer(self, sender_id: int, params: Dict) -> Dict:
        """
        从对等节点接收模型参数并缓存

        参数：
            sender_id: 发送方客户端ID
            params: 模型参数字典

        返回：
            接收确认信息
        """
        self.received_params_buffer.append({
            'sender_id': sender_id,
            'params': params,
            'round': getattr(self, 'current_round', 0)
        })
        return {
            'receiver_id': self.client_id,
            'sender_id': sender_id,
            'status': 'received'
        }

    def aggregate_from_buffer(self, strategy: str = 'avg') -> Dict:
        """
        从缓冲池聚合接收到的模型

        参数：
            strategy: 聚合策略，'avg'或'weighted'

        返回：
            聚合后的模型参数字典
        """
        if not self.received_params_buffer:
            return self.get_model_params()

        received = [entry['params'] for entry in self.received_params_buffer]
        received.append(self.get_model_params())

        if strategy == 'avg':
            aggregated = self._average_models(received)
        elif strategy == 'weighted':
            weights = [1.0] * len(received)
            aggregated = self._weighted_average_models(received, weights)
        else:
            aggregated = self._average_models(received)

        self.received_params_buffer.clear()
        return aggregated

    def _average_models(self, model_list: List[Dict]) -> Dict:
        """
        简单平均多个模型参数

        参数：
            model_list: 模型参数字典列表

        返回：
            平均后的模型参数字典
        """
        avg_params = {}
        for key in model_list[0].keys():
            avg_params[key] = torch.stack([m[key] for m in model_list]).mean(dim=0)
        return avg_params

    def _weighted_average_models(self, model_list: List[Dict], weights: List[float]) -> Dict:
        """
        加权平均模型参数

        参数：
            model_list: 模型参数字典列表
            weights: 权重列表

        返回：
            加权平均后的模型参数字典
        """
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        avg_params = {}
        for key in model_list[0].keys():
            weighted_sum = torch.zeros_like(model_list[0][key])
            for m, w in zip(model_list, normalized_weights):
                weighted_sum += m[key] * w
            avg_params[key] = weighted_sum
        return avg_params

    def update_performance_history(self, round_id: int, metrics: Dict):
        """
        更新性能历史记录

        参数：
            round_id: 轮次ID
            metrics: 性能指标字典
        """
        self.performance_history[round_id] = metrics

    def get_neighbors(self) -> List[int]:
        """
        获取当前邻居列表

        返回：
            邻居ID列表
        """
        return self.neighbors

    def gossip_sync(self, round_id: int, active_neighbors: List[int] = None) -> Dict:
        """
        Gossip协议同步：选择性发送模型给指定邻居并聚合

        参数：
            round_id: 当前轮次
            active_neighbors: 本轮要通信的邻居列表，None则使用所有邻居

        返回：
            同步结果字典
        """
        self.current_round = round_id
        target_neighbors = active_neighbors if active_neighbors is not None else self.neighbors

        params = self.get_model_params()
        self.send_model_to_neighbors(params)

        aggregated_params = self.aggregate_from_buffer(strategy='avg')
        self.set_model_params(aggregated_params)

        return {
            'client_id': self.client_id,
            'round_id': round_id,
            'neighbors_contacted': target_neighbors,
            'messages_received': len(self.received_params_buffer)
        }

    def init_election_state(self, neighbors: List[int],
                           resource_params: Optional[Dict] = None,
                           base_trigger_time: float = 10.0,
                           trigger_factor: float = 0.1):
        """
        初始化选举状态

        Args:
            neighbors: 邻居节点ID列表
            resource_params: 资源参数（CPU、内存、带宽等）
            base_trigger_time: 基础触发时间
            trigger_factor: 触发时间因子
        """
        from .election import DistributedElection

        self.election_state = DistributedElection(
            client_id=self.client_id,
            neighbors=neighbors,
            resource_params=resource_params,
            base_trigger_time=base_trigger_time,
            trigger_factor=trigger_factor
        )

        return self.election_state.get_best_client_info()

    def get_resource_score(self) -> float:
        """
        获取本节点的资源得分

        Returns:
            资源得分
        """
        if hasattr(self, 'election_state'):
            return self.election_state.resource_score
        return 0.0

    def get_trigger_time(self) -> float:
        """
        获取触发时间

        Returns:
            触发时间（秒）
        """
        if hasattr(self, 'election_state'):
            return self.election_state.trigger_time
        return 999999.0

    def broadcast_election(self, exclude_neighbor: Optional[int] = None) -> List[Dict]:
        """
        广播选举消息

        Args:
            exclude_neighbor: 排除的邻居ID

        Returns:
            待发送的消息列表
        """
        if not hasattr(self, 'election_state'):
            return []

        messages = self.election_state.broadcast_election_message(excludeNeighbor=exclude_neighbor)

        results = []
        for neighbor_id, msg in messages:
            msg_dict = {
                'type': msg.msg_type.value,
                'candidate_id': msg.candidate_id,
                'candidate_score': msg.candidate_score,
                'hops': msg.hops,
                'source_id': msg.source_id,
                'sequence_id': msg.sequence_id,
                'sender_id': self.client_id
            }
            results.append({'neighbor_id': neighbor_id, 'message': msg_dict})

        return results

    def receive_election_message(self, msg_dict: Dict) -> Optional[List[Dict]]:
        """
        接收并处理选举消息

        Args:
            msg_dict: 选举消息字典

        Returns:
            需要转发的消息列表，None表示丢弃
        """
        if not hasattr(self, 'election_state'):
            return None

        from .election import ElectionMessage, ElectionMessageType

        msg = ElectionMessage(
            msg_type=ElectionMessageType(msg_dict['type']),
            candidate_id=msg_dict['candidate_id'],
            candidate_score=msg_dict['candidate_score'],
            hops=msg_dict['hops'],
            source_id=msg_dict['source_id'],
            sequence_id=msg_dict['sequence_id'],
            sender_id=msg_dict['sender_id']
        )

        outgoing = self.election_state.receive_election_message(msg)

        if outgoing is None:
            return None

        results = []
        for neighbor_id, forward_msg in outgoing:
            msg_dict = {
                'type': forward_msg.msg_type.value,
                'candidate_id': forward_msg.candidate_id,
                'candidate_score': forward_msg.candidate_score,
                'hops': forward_msg.hops,
                'source_id': forward_msg.source_id,
                'sequence_id': forward_msg.sequence_id,
                'sender_id': self.client_id
            }
            results.append({'neighbor_id': neighbor_id, 'message': msg_dict})

        return results

    def get_best_client_info(self) -> Dict:
        """
        获取最优客户端信息

        Returns:
            最优客户端信息字典
        """
        if hasattr(self, 'election_state'):
            return self.election_state.get_best_client_info()
        return {
            'best_client_id': -1,
            'best_client_score': 0.0,
            'hops_to_best': 999999,
            'next_hop': -1,
            'is_best_client': False,
            'my_score': 0.0,
            'my_trigger_time': 999999.0
        }

    def is_best_client(self) -> bool:
        """
        检查本节点是否为最优客户端

        Returns:
            是否为最优客户端
        """
        if hasattr(self, 'election_state'):
            return self.election_state.best_client_table.candidate_id == self.client_id
        return False

    # ==================== 聚类相关方法 ====================

    def init_clustering_state(self, neighbors: List[int], resource_score: float,
                              election_trigger_time: float, max_child: int = 3,
                              max_retry: int = 3, request_timeout: float = 2.0,
                              cluster_offset: float = 30.0):
        """
        初始化聚类状态

        Args:
            neighbors: 邻居节点ID列表
            resource_score: 资源得分（从选举模块获取）
            election_trigger_time: 选举触发时间（从选举模块获取）
            max_child: 最大子节点数
            max_retry: 最大重试次数
            request_timeout: 请求超时时间（秒）
            cluster_offset: 聚类触发时间偏移量（秒）
        """
        from .election import DistributedClustering

        self.clustering_state = DistributedClustering(
            client_id=self.client_id,
            neighbors=neighbors,
            resource_score=resource_score,
            election_trigger_time=election_trigger_time,
            max_child=max_child,
            max_retry=max_retry,
            request_timeout=request_timeout,
            cluster_offset=cluster_offset
        )

        return self.clustering_state.get_cluster_info()

    def get_cluster_trigger_time(self) -> float:
        """
        获取聚类触发时间

        Returns:
            聚类触发时间（秒）
        """
        if hasattr(self, 'clustering_state'):
            return self.clustering_state.cluster_trigger_time
        return 999999.0

    def trigger_clustering(self) -> List[Dict]:
        """
        触发聚类过程

        Returns:
            需要发送的消息列表
        """
        if not hasattr(self, 'clustering_state'):
            return []

        messages = self.clustering_state.trigger()

        results = []
        for neighbor_id, msg in messages:
            msg_dict = {
                'type': msg.msg_type.value,
                'sender_id': msg.sender_id,
                'receiver_id': msg.receiver_id,
                'child_count': msg.child_count,
                'cluster_id': msg.cluster_id
            }
            results.append({'neighbor_id': neighbor_id, 'message': msg_dict})

        return results

    def send_agg_req(self, target_neighbor: int) -> Dict:
        """
        向指定邻居发送聚合请求

        Args:
            target_neighbor: 目标邻居ID

        Returns:
            发送结果
        """
        if not hasattr(self, 'clustering_state'):
            return {'status': 'error', 'message': 'Clustering not initialized'}

        from .election import ClusterMessage, ClusterMessageType

        msg = ClusterMessage(
            msg_type=ClusterMessageType.AGG_REQ,
            sender_id=self.client_id,
            receiver_id=target_neighbor,
            child_count=self.clustering_state.child_count,
            cluster_id=self.clustering_state.cluster_id
        )

        return {
            'status': 'sent',
            'message': {
                'type': msg.msg_type.value,
                'sender_id': msg.sender_id,
                'receiver_id': msg.receiver_id,
                'child_count': msg.child_count,
                'cluster_id': msg.cluster_id
            }
        }

    def receive_agg_req(self, msg_dict: Dict) -> Optional[Dict]:
        """
        处理收到的聚合请求

        Args:
            msg_dict: 聚合请求消息字典

        Returns:
            响应消息（ACCEPT或REJECT），None表示忽略
        """
        if not hasattr(self, 'clustering_state'):
            return None

        from .election import ClusterMessage, ClusterMessageType

        msg = ClusterMessage(
            msg_type=ClusterMessageType(msg_dict['type']),
            sender_id=msg_dict['sender_id'],
            receiver_id=msg_dict['receiver_id'],
            child_count=msg_dict['child_count'],
            cluster_id=msg_dict.get('cluster_id', -1)
        )

        response = self.clustering_state.receive_agg_req(msg)

        if response is None:
            return None

        return {
            'type': response.msg_type.value,
            'sender_id': response.sender_id,
            'receiver_id': response.receiver_id,
            'child_count': response.child_count,
            'cluster_id': response.cluster_id
        }

    def receive_accept(self, msg_dict: Dict):
        """
        处理收到的ACCEPT消息

        Args:
            msg_dict: ACCEPT消息字典
        """
        if not hasattr(self, 'clustering_state'):
            return

        from .election import ClusterMessage, ClusterMessageType

        msg = ClusterMessage(
            msg_type=ClusterMessageType(msg_dict['type']),
            sender_id=msg_dict['sender_id'],
            receiver_id=msg_dict['receiver_id'],
            child_count=msg_dict['child_count'],
            cluster_id=msg_dict.get('cluster_id', -1)
        )

        self.clustering_state.receive_accept(msg)

    def receive_reject(self, msg_dict: Dict) -> List[Dict]:
        """
        处理收到的REJECT消息，尝试下一个邻居

        Args:
            msg_dict: REJECT消息字典

        Returns:
            需要发送的新请求消息列表
        """
        if not hasattr(self, 'clustering_state'):
            return []

        from .election import ClusterMessage, ClusterMessageType

        msg = ClusterMessage(
            msg_type=ClusterMessageType(msg_dict['type']),
            sender_id=msg_dict['sender_id'],
            receiver_id=msg_dict['receiver_id'],
            child_count=msg_dict['child_count'],
            cluster_id=msg_dict.get('cluster_id', -1)
        )

        messages = self.clustering_state.receive_reject(msg)

        results = []
        for neighbor_id, forward_msg in messages:
            msg_dict = {
                'type': forward_msg.msg_type.value,
                'sender_id': forward_msg.sender_id,
                'receiver_id': forward_msg.receiver_id,
                'child_count': forward_msg.child_count,
                'cluster_id': forward_msg.cluster_id
            }
            results.append({'neighbor_id': neighbor_id, 'message': msg_dict})

        return results

    def get_cluster_info(self) -> Dict:
        """
        获取聚类信息

        Returns:
            聚类信息字典
        """
        if hasattr(self, 'clustering_state'):
            return self.clustering_state.get_cluster_info()
        return {
            'client_id': self.client_id,
            'state': 'waiting',
            'parent': None,
            'children': [],
            'child_count': 0,
            'cluster_id': None,
            'trigger_time': 999999.0,
            'resource_score': 0.0
        }

    def is_cluster_head(self) -> bool:
        """
        检查是否为簇首

        Returns:
            是否为簇首
        """
        if hasattr(self, 'clustering_state'):
            return self.clustering_state.is_cluster_head()
        return False
