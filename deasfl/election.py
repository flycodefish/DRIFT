"""
分布式最优客户端选举算法

实现基于资源得分的分布式选举算法，用于在Mesh拓扑中选举出资源最优的节点。

核心机制：
1. 资源得分计算：根据设备异构资源（CPU、内存、通信能力等）计算综合得分
2. 触发时间计算：得分越高，触发时间越早，实现先到先得的选举模式
3. 被动触发机制：节点收到更优节点消息时立即触发，加速收敛
4. 路径记录：记录到最优节点的路由信息，用于后续数据传输

算法流程：
1. 初始化阶段：计算资源得分、触发时间
2. 选举阶段：基于触发时间的广播/被动触发机制
3. 收敛阶段：所有节点就最优节点达成一致
4. 路由阶段：每个节点记录到最优节点的路径和下一跳

参考：/home/yjq/De-ASFL-V8/bestclient.md
"""

import time
import random
import threading
import numpy as np
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class ElectionMessageType(Enum):
    """选举消息类型"""
    ELECTION = "election"
    ELECTED = "elected"


@dataclass
class ElectionMessage:
    """选举消息数据结构"""
    msg_type: ElectionMessageType
    candidate_id: int
    candidate_score: float
    hops: int
    source_id: int
    sequence_id: str
    sender_id: int = -1


@dataclass
class BestClientTable:
    """最优客户端路由表"""
    candidate_id: int = -1
    candidate_score: float = 0.0
    hops: int = 999999
    next_hop: int = -1


class DistributedElection:
    """
    分布式最优客户端选举算法

    使用基于触发时间的广播选举算法，在去中心化Mesh网络中选举资源最优节点。
    """

    def __init__(self, client_id: int, neighbors: List[int],
                 resource_params: Optional[Dict] = None,
                 base_trigger_time: float = 10.0,
                 trigger_factor: float = 0.1):
        """
        初始化选举模块

        Args:
            client_id: 本节点ID
            neighbors: 邻居节点ID列表
            resource_params: 资源参数（CPU核心数、内存、带宽等）
            base_trigger_time: 基础触发时间（秒）
            trigger_factor: 触发时间因子（得分乘数）
        """
        self.client_id = client_id
        self.neighbors = neighbors

        self.base_trigger_time = base_trigger_time
        self.trigger_factor = trigger_factor

        self.has_initiated = False
        self.processed_messages: Set[str] = set()

        self.best_client_table = BestClientTable()

        self.resource_score = self._calculate_resource_score(resource_params)
        self.trigger_time = self._calculate_trigger_time()

        self.election_in_progress = False
        self.election_lock = threading.Lock()

        self.message_handlers: Dict[ElectionMessageType, callable] = {}

    def _calculate_resource_score(self, resource_params: Optional[Dict]) -> float:
        """
        计算本节点的资源得分

        考虑因素：
        1. CPU计算能力（核心数、频率）
        2. 内存大小
        3. 通信带宽
        4. GPU资源（如果有）
        5. 随机噪声（确保唯一性）

        公式：
        score = w1*CPU + w2*Memory + w3*Bandwidth + w4*GPU + noise

        Args:
            resource_params: 资源参数字典

        Returns:
            资源得分（0-100）
        """
        if resource_params is None:
            resource_params = self._generate_random_resources()

        cpu_score = resource_params.get('cpu_cores', 4) * 3       # 6-48
        memory_score = resource_params.get('memory_gb', 8) * 0.8   # 3.2-51.2
        bandwidth_score = resource_params.get('bandwidth_mbps', 100) * 0.02  # 1-20
        gpu_score = resource_params.get('gpu_memory_gb', 0) * 1.5  # 0-48

        noise = random.uniform(-0.5, 0.5)

        total_score = cpu_score + memory_score + bandwidth_score + gpu_score + noise

        return max(0.1, total_score)

    def _generate_random_resources(self) -> Dict:
        """
        生成随机资源参数（模拟异构设备）

        Returns:
            资源参数字典
        """
        return {
            'cpu_cores': random.randint(2, 16),
            'memory_gb': random.randint(4, 64),
            'bandwidth_mbps': random.randint(50, 1000),
            'gpu_memory_gb': random.choice([0, 4, 8, 16, 32])
        }

    def _calculate_trigger_time(self) -> float:
        """
        计算触发时间

        得分越高，触发时间越早
        公式：T_trigger = base_time - score * factor + noise

        添加少量随机噪声确保相同得分也有微小差距

        Returns:
            触发时间（秒）
        """
        base_time = self.base_trigger_time
        score_factor = self.resource_score * self.trigger_factor
        noise = random.uniform(0, 0.1)

        trigger_time = base_time - score_factor + noise

        # 使用更小的最小值，允许触发时间反映真实得分差异
        return max(0.001, trigger_time)

    def register_message_handler(self, msg_type: ElectionMessageType, handler: callable):
        """
        注册消息处理器

        Args:
            msg_type: 消息类型
            handler: 处理函数
        """
        self.message_handlers[msg_type] = handler

    def broadcast_election_message(self, excludeNeighbor: Optional[int] = None) -> List[ElectionMessage]:
        """
        广播选举消息给所有邻居（除指定邻居外）

        Returns:
            待发送的消息列表
        """
        if self.has_initiated:
            return []

        self.has_initiated = True

        msg = ElectionMessage(
            msg_type=ElectionMessageType.ELECTION,
            candidate_id=self.client_id,
            candidate_score=self.resource_score,
            hops=0,
            source_id=self.client_id,
            sequence_id=f"{self.client_id}_{time.time()}",
            sender_id=self.client_id
        )

        self._update_best_client_table(msg)

        messages = []
        for neighbor_id in self.neighbors:
            if neighbor_id != excludeNeighbor:
                messages.append((neighbor_id, msg))

        return messages

    def receive_election_message(self, msg: ElectionMessage) -> Optional[List[tuple]]:
        """
        处理接收到的选举消息

        Args:
            msg: 选举消息

        Returns:
            需要发送的消息列表 [(neighbor_id, msg), ...]，None表示丢弃
        """
        if msg.candidate_id == self.client_id:
            return None

        msg_id = f"{msg.candidate_id}_{msg.sequence_id}"
        if msg_id in self.processed_messages:
            return None
        self.processed_messages.add(msg_id)

        # 创建消息副本，避免修改原始消息
        new_msg = ElectionMessage(
            msg_type=msg.msg_type,
            candidate_id=msg.candidate_id,
            candidate_score=msg.candidate_score,
            hops=msg.hops + 1,  # 跳数+1
            source_id=self.client_id,  # 更新source_id为当前节点
            sequence_id=msg.sequence_id,
            sender_id=msg.sender_id  # 保留原始发送者，用于路由计算
        )

        should_relay = self._update_best_client_table(new_msg)

        if not self.has_initiated:
            self.has_initiated = True

            if msg.candidate_score > self.resource_score:
                should_relay = True
            else:
                self_best_msg = ElectionMessage(
                    msg_type=ElectionMessageType.ELECTION,
                    candidate_id=self.client_id,
                    candidate_score=self.resource_score,
                    hops=0,
                    source_id=self.client_id,
                    sequence_id=f"{self.client_id}_{time.time()}",
                    sender_id=self.client_id
                )
                self._update_best_client_table(self_best_msg)

                messages = []
                for neighbor_id in self.neighbors:
                    if neighbor_id != msg.sender_id:
                        messages.append((neighbor_id, self_best_msg))
                return messages
        elif should_relay:
            messages = []
            for neighbor_id in self.neighbors:
                if neighbor_id != msg.sender_id:
                    # 转发时创建新的消息副本，更新source_id和sender_id为当前节点
                    relay_msg = ElectionMessage(
                        msg_type=new_msg.msg_type,
                        candidate_id=new_msg.candidate_id,
                        candidate_score=new_msg.candidate_score,
                        hops=new_msg.hops,
                        source_id=self.client_id,
                        sequence_id=new_msg.sequence_id,
                        sender_id=self.client_id  # 更新sender_id为当前节点，使接收者知道直接发送者
                    )
                    messages.append((neighbor_id, relay_msg))
            return messages

        return None

    def _update_best_client_table(self, msg: ElectionMessage) -> bool:
        """
        更新最优客户端路由表

        Args:
            msg: 选举消息

        Returns:
            是否需要转发消息
        """
        should_relay = False

        if self.best_client_table.candidate_id == -1:
            should_relay = True
            self.best_client_table = BestClientTable(
                candidate_id=msg.candidate_id,
                candidate_score=msg.candidate_score,
                hops=msg.hops,
                next_hop=msg.sender_id if msg.hops > 0 else msg.candidate_id
            )
        elif msg.candidate_score > self.best_client_table.candidate_score:
            should_relay = True
            self.best_client_table = BestClientTable(
                candidate_id=msg.candidate_id,
                candidate_score=msg.candidate_score,
                hops=msg.hops,
                next_hop=msg.sender_id if msg.hops > 0 else msg.candidate_id
            )
        elif (msg.candidate_score == self.best_client_table.candidate_score and
              msg.hops < self.best_client_table.hops):
            should_relay = True
            self.best_client_table = BestClientTable(
                candidate_id=msg.candidate_id,
                candidate_score=msg.candidate_score,
                hops=msg.hops,
                next_hop=msg.sender_id if msg.hops > 0 else msg.candidate_id
            )

        return should_relay

    def get_best_client_info(self) -> Dict:
        """
        获取最优客户端信息

        Returns:
            最优客户端信息字典
        """
        return {
            'best_client_id': self.best_client_table.candidate_id,
            'best_client_score': self.best_client_table.candidate_score,
            'hops_to_best': self.best_client_table.hops,
            'next_hop': self.best_client_table.next_hop,
            'is_best_client': self.best_client_table.candidate_id == self.client_id,
            'my_score': self.resource_score,
            'my_trigger_time': self.trigger_time
        }

    def is_election_complete(self) -> bool:
        """
        检查选举是否完成（所有节点都已初始化）

        Returns:
            是否完成
        """
        return self.has_initiated

    def reset(self):
        """重置选举状态"""
        with self.election_lock:
            self.has_initiated = False
            self.processed_messages.clear()
            self.best_client_table = BestClientTable()
            self.election_in_progress = False

    def set_neighbors(self, neighbors: List[int]):
        """
        更新邻居列表

        Args:
            neighbors: 新的邻居列表
        """
        self.neighbors = neighbors


class ElectionManager:
    """
    选举管理器：协调所有节点的选举过程

    负责：
    1. 触发选举过程
    2. 管理消息传递
    3. 检测选举收敛
    4. 提供选举结果查询
    """

    def __init__(self, clients: List, topology: List[List[int]],
                 base_trigger_time: float = 10.0,
                 trigger_factor: float = 0.1):
        """
        初始化选举管理器

        Args:
            clients: 客户端Actor列表
            topology: 拓扑信息 [[neighbors], ...]
            base_trigger_time: 基础触发时间
            trigger_factor: 触发时间因子
        """
        self.clients = clients
        self.topology = topology
        self.num_clients = len(clients)

        self.election_states: List[DistributedElection] = []

        self._init_election_states(base_trigger_time, trigger_factor)

        self.best_client_id: Optional[int] = None
        self.election_complete = False

    def _init_election_states(self, base_trigger_time: float, trigger_factor: float):
        """初始化所有节点的选举状态"""
        for i in range(self.num_clients):
            neighbors = self.topology[i] if i < len(self.topology) else []

            state = DistributedElection(
                client_id=i,
                neighbors=neighbors,
                base_trigger_time=base_trigger_time,
                trigger_factor=trigger_factor
            )

            self.election_states.append(state)

    def start_election(self) -> Dict:
        """
        启动选举过程

        Returns:
            选举结果信息
        """
        print(f"\n{'='*60}")
        print(f"Starting Distributed Election with {self.num_clients} nodes")
        print(f"{'='*60}")

        for i in range(self.num_clients):
            info = self.election_states[i].get_best_client_info()
            print(f"Node {i}: score={info['my_score']:.2f}, "
                  f"trigger_time={info['my_trigger_time']:.2f}s")

        election_results = self._run_election()

        self.election_complete = True

        return election_results

    def _run_election(self) -> Dict:
        """
        运行选举过程

        Returns:
            选举结果
        """
        message_queues = {i: [] for i in range(self.num_clients)}

        pending_triggers = []
        for i in range(self.num_clients):
            state = self.election_states[i]
            pending_triggers.append((state.trigger_time, i, 'trigger'))

        pending_triggers.sort(key=lambda x: x[0])

        all_initiated = [False] * self.num_clients
        max_iterations = self.num_clients * 2
        iteration = 0

        while pending_triggers and iteration < max_iterations:
            current_time, node_id, event_type = pending_triggers.pop(0)

            if event_type == 'trigger':
                if not all_initiated[node_id]:
                    all_initiated[node_id] = True

                    messages = self.election_states[node_id].broadcast_election_message()

                    for neighbor_id, msg in messages:
                        message_queues[neighbor_id].append(msg)

            elif event_type == 'message':
                msg = node_id
                receiving_node = event_type

                if isinstance(node_id, int) and isinstance(event_type, str):
                    receiving_node = int(event_type)
                    msg = message_queues[node_id][0] if message_queues[node_id] else None
                    if msg is None:
                        continue

            iteration += 1

        for node_id in range(self.num_clients):
            if not all_initiated[node_id]:
                state = self.election_states[node_id]
                messages = state.broadcast_election_message()
                for neighbor_id, msg in messages:
                    message_queues[neighbor_id].append(msg)

        self._process_all_messages(message_queues)

        return self._collect_election_results()

    def _process_all_messages(self, message_queues: Dict[int, List]):
        """处理所有消息队列中的消息"""
        for _ in range(self.num_clients):
            any_processed = False

            for node_id in range(self.num_clients):
                while message_queues[node_id]:
                    msg = message_queues[node_id].pop(0)

                    outgoing = self.election_states[node_id].receive_election_message(msg)

                    if outgoing:
                        any_processed = True
                        for neighbor_id, forward_msg in outgoing:
                            message_queues[neighbor_id].append(forward_msg)

            if not any_processed:
                break

    def _collect_election_results(self) -> Dict:
        """收集选举结果"""
        best_client_id = -1
        best_score = -1
        all_info = []

        for i in range(self.num_clients):
            info = self.election_states[i].get_best_client_info()
            all_info.append(info)

            if info['best_client_score'] > best_score:
                best_score = info['best_client_score']
                best_client_id = info['best_client_id']

        self.best_client_id = best_client_id

        result = {
            'best_client_id': best_client_id,
            'best_client_score': best_score,
            'all_nodes_info': all_info,
            'convergence_hops': max(info['hops_to_best'] for info in all_info) if all_info else 0
        }

        print(f"\n{'='*60}")
        print(f"Election Complete!")
        print(f"Best Client: Node {best_client_id} with score {best_score:.2f}")
        print(f"Convergence hops: {result['convergence_hops']}")
        print(f"{'='*60}\n")

        for i, info in enumerate(all_info):
            status = "*** BEST ***" if info['best_client_id'] == best_client_id else ""
            print(f"Node {i}: "
                  f"score={info['my_score']:.2f}, "
                  f"best={info['best_client_id']} (hops={info['hops_to_best']}), "
                  f"next_hop={info['next_hop']} {status}")

        return result

    def get_best_client_id(self) -> Optional[int]:
        """
        获取最优客户端ID

        Returns:
            最优客户端ID，未选举完成返回None
        """
        return self.best_client_id if self.election_complete else None

    def get_routing_info(self, node_id: int) -> Dict:
        """
        获取指定节点到最优客户端的路由信息

        Args:
            node_id: 节点ID

        Returns:
            路由信息
        """
        if 0 <= node_id < len(self.election_states):
            return self.election_states[node_id].get_best_client_info()
        return {}


# ==============================================================================
#                    分布式聚类算法（基于触发时间偏移）
# ==============================================================================
# 
# 本实现基于资源驱动定时触发聚类算法，通过触发时间偏移方案与选举算法解耦
# 聚类触发时间 = 选举触发时间 + 偏移量
# 不影响已有的最优客户端选举功能
# ==============================================================================

from enum import Enum


class ClusterMessageType(Enum):
    """聚类消息类型"""
    AGG_REQ = "agg_req"      # 聚合请求
    ACCEPT = "accept"        # 接受请求
    REJECT = "reject"        # 拒绝请求


class ClusterState(Enum):
    """节点聚类状态"""
    WAITING = "waiting"      # 等待触发
    TRIGGERED = "triggered"  # 已触发
    CLUSTER_HEAD = "cluster_head"  # 簇首
    MEMBER = "member"        # 成员节点


@dataclass
class ClusterMessage:
    msg_type: ClusterMessageType
    sender_id: int
    receiver_id: int
    child_count: int = 0
    cluster_id: int = -1
    request_seq: int = 0  # 请求序号，用于支持重试和去重
    sender_trigger_time: Optional[float] = None  # 发送方的触发时间，用于判断同时触发
    sender_label_distribution: object = None  # 发送方的标签分布（用于分布感知聚类）


class DistributedClustering:
    """
    分布式聚类算法

    使用基于资源得分的触发时间偏移方案，在选举完成后自动触发聚类。
    资源越多的节点触发越早，优先成为簇首。
    """

    # 邻居状态枚举
    class NeighborState(Enum):
        UNTRIED = "untried"
        PENDING = "pending"
        FAILED = "failed"
        ACCEPTED = "accepted"

    def __init__(self, client_id: int, neighbors: List[int],
                 resource_score: float, election_trigger_time: float,
                 max_child: int = 3, max_retry: int = 3,
                 request_timeout: float = 2.0, cluster_offset: float = 30.0,
                 simultaneous_threshold: float = 1.0,
                 label_distribution = None,
                 distribution_similarity_threshold: float = 0.5):
        """
        初始化聚类状态

        Args:
            client_id: 本节点ID
            neighbors: 邻居节点ID列表
            resource_score: 资源得分（从选举模块获取）
            election_trigger_time: 选举触发时间（从选举模块获取）
            max_child: 最大子节点数
            max_retry: 最大重试次数
            request_timeout: 请求超时时间（秒）
            cluster_offset: 聚类触发时间偏移量（秒）
            simultaneous_threshold: 同时触发判定阈值（秒）
            label_distribution: 本节点的标签分布（概率向量，shape=(num_classes,)）
            distribution_similarity_threshold: 分布相似度阈值（JSD），低于此值才接受聚合
        """
        self.client_id = client_id
        self.neighbors = neighbors
        self.resource_score = resource_score
        self.election_trigger_time = election_trigger_time

        self.max_child = max_child
        self.max_retry = max_retry
        self.request_timeout = request_timeout
        self.simultaneous_threshold = simultaneous_threshold

        # 分布感知聚类
        self.label_distribution = label_distribution
        self.distribution_similarity_threshold = distribution_similarity_threshold

        # 计算聚类触发时间（触发时间偏移方案）
        self.cluster_trigger_time = self._calculate_cluster_trigger(cluster_offset)

        # 节点状态
        self.state = ClusterState.WAITING
        self.parent = None
        self.children = []
        self.child_count = 0
        self.cluster_id = None

        # 重试计数器
        self.retry_count = {n: 0 for n in neighbors}

        # 消息去重
        self.processed_messages: Set[str] = set()

        # 邻居状态（改进的状态机）
        self.neighbor_state: Dict[int, 'DistributedClustering.NeighborState'] = {
            n: DistributedClustering.NeighborState.UNTRIED for n in neighbors
        }

        # 请求序号（用于支持重试和去重）
        self.request_seq: int = 0

        # 挂起请求状态（保证串行发送）
        self.pending_request: Optional[int] = None

    def _calculate_cluster_trigger(self, cluster_offset: float) -> float:
        """
        计算聚类触发时间

        公式：cluster_trigger = election_trigger + offset - score * factor
        资源越高，聚类触发也越早（保持与选举一致的优先级）

        Args:
            cluster_offset: 基础偏移量

        Returns:
            聚类触发时间
        """
        factor = 0.1  # 增加资源得分影响因子，使触发时间更分散
        return self.election_trigger_time + cluster_offset - self.resource_score * factor

    def trigger(self) -> List[tuple]:
        """
        触发聚类过程（到达触发时间时调用）

        修改后逻辑：
        - 触发时间到达后，无论是否已有父节点，都可以向邻居发起聚合请求
        - 但不能请求自己的父节点（避免回环）
        - 触发时不立即成为簇首，而是先尝试向邻居发送请求
        - 只有当所有请求都被拒绝且没有父节点时，才成为簇首
        - 真正串行发送：一次只发送一个请求，等待响应后再发送下一个
        - 检查挂起请求状态，确保真正串行

        Returns:
            需要发送的消息列表 [(neighbor_id, msg), ...]
        """
        if self.state == ClusterState.TRIGGERED or self.state == ClusterState.CLUSTER_HEAD:
            # 已经触发过，不再重复触发
            return []

        self.state = ClusterState.TRIGGERED

        # 检查是否有挂起请求（保证串行）
        if self.pending_request is not None:
            # 已有挂起请求，等待响应后再继续
            return []

        # 获取可请求的邻居（排除父节点、已接受和挂起的邻居）
        available_neighbors = [
            n for n in self.neighbors 
            if n != self.parent and 
               self.neighbor_state.get(n) != DistributedClustering.NeighborState.ACCEPTED and
               self.neighbor_state.get(n) != DistributedClustering.NeighborState.PENDING
        ]
        
        if available_neighbors:
            # 还有可请求的邻居，按资源排序（优先选择资源低的作为子节点）
            sorted_neighbors = self._sort_neighbors_by_resource(available_neighbors)
            # 真正串行发送：只发送第一个请求
            return self._send_next_request(sorted_neighbors)
        else:
            # 没有可请求的邻居了
            if self.parent is None:
                # 没有父节点，成为簇首
                self.state = ClusterState.CLUSTER_HEAD
                self.cluster_id = self.client_id
            return []

    def _sort_neighbors_by_resource(self, neighbors: Optional[List[int]] = None) -> List[int]:
        """
        按资源得分排序邻居（用于决定发送顺序）

        Args:
            neighbors: 可选，指定要排序的邻居列表，默认为self.neighbors

        Returns:
            排序后的邻居列表
        """
        target_neighbors = neighbors if neighbors is not None else self.neighbors
        return sorted(target_neighbors)

    def _send_next_request(self, targets: List[int]) -> List[tuple]:
        """
        真正串行发送：向目标邻居发送下一个聚合请求（一次只发一个）

        Args:
            targets: 目标邻居列表（已排序）

        Returns:
            需要发送的消息列表（最多包含一个消息）
        """
        # 检查是否还有容量
        if self.child_count >= self.max_child:
            return []

        # 检查是否有挂起请求（保证串行）
        if self.pending_request is not None:
            return []

        # 找到第一个未尝试或可重试的邻居
        for neighbor_id in targets:
            state = self.neighbor_state.get(neighbor_id)
            if state == DistributedClustering.NeighborState.UNTRIED:
                # 从未尝试过，可以发送
                break
            elif state == DistributedClustering.NeighborState.FAILED:
                # 失败过，检查是否可以重试
                if self.retry_count.get(neighbor_id, 0) < self.max_retry:
                    break
        else:
            # 没有找到可发送的邻居
            return []

        # 更新邻居状态为挂起
        self.neighbor_state[neighbor_id] = DistributedClustering.NeighborState.PENDING
        self.pending_request = neighbor_id
        
        # 增加请求序号
        self.request_seq += 1
        
        msg = ClusterMessage(
            msg_type=ClusterMessageType.AGG_REQ,
            sender_id=self.client_id,
            receiver_id=neighbor_id,
            child_count=self.child_count,
            cluster_id=self.cluster_id,
            request_seq=self.request_seq,
            sender_trigger_time=self.cluster_trigger_time,  # 添加发送方触发时间
            sender_label_distribution=self.label_distribution  # 发送方的标签分布
        )
        return [(neighbor_id, msg)]

    def _compute_js_divergence(self, dist_p: np.ndarray, dist_q: np.ndarray) -> float:
        """
        计算两个概率分布之间的 Jensen-Shannon 散度（对称、有界 [0, ln2]）

        Args:
            dist_p: 分布 P（概率向量）
            dist_q: 分布 Q（概率向量）

        Returns:
            JSD 值，越小越相似
        """
        epsilon = 1e-10
        p = np.asarray(dist_p, dtype=np.float64) + epsilon
        q = np.asarray(dist_q, dtype=np.float64) + epsilon
        p = p / p.sum()
        q = q / q.sum()
        m = (p + q) / 2.0
        return float(np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))) / 2.0

    def receive_agg_req(self, msg: ClusterMessage) -> Optional[ClusterMessage]:
        """
        处理收到的聚合请求

        修改后逻辑：
        - 已有父节点 → 拒绝（保持单父节点约束）
        - CLUSTER_HEAD → 拒绝（簇首永不降级）
        - 已触发且非同时触发 → 拒绝（保证簇首稳定性）
        - 同时触发冲突（触发时间差在阈值内）→ 比较ID，ID小的成为父节点
        - 数据分布不相似 → 拒绝（分布感知聚类）
        - 无父节点 + 未触发 → 接受

        Args:
            msg: 聚合请求消息

        Returns:
            响应消息（ACCEPT或REJECT），None表示忽略
        """
        # 去重检查（使用sender_id、request_seq和cluster_id作为去重键，更稳定）
        msg_cluster_id = msg.cluster_id if msg.cluster_id is not None else -1
        msg_id = f"{msg.sender_id}_{msg.request_seq}_{msg_cluster_id}"
        if msg_id in self.processed_messages:
            return None
        self.processed_messages.add(msg_id)

        # 规则1：已有父节点 → 拒绝（保持单父节点约束）
        if self.parent is not None:
            return self._reject_request(msg)

        # 规则2：CLUSTER_HEAD → 拒绝（簇首永不降级）
        if self.state == ClusterState.CLUSTER_HEAD:
            return self._reject_request(msg)

        # 规则3：已触发（处理同时触发冲突）
        if self.state == ClusterState.TRIGGERED:
            # 检查是否为同时触发（触发时间差在阈值内）
            if msg.sender_trigger_time is not None:
                time_diff = abs(self.cluster_trigger_time - msg.sender_trigger_time)
                is_simultaneous = time_diff <= self.simultaneous_threshold
            else:
                # 无法判断时，保守处理为非同时触发
                is_simultaneous = False

            if is_simultaneous:
                # 同时触发冲突处理：双方都已触发时，ID小的成为父节点
                if self.client_id > msg.sender_id:
                    # 我ID更大，接受对方请求，成为子节点
                    return self._accept_request(msg)
                else:
                    # 我ID更小，拒绝对方请求
                    return self._reject_request(msg)
            else:
                # 非同时触发，已触发的节点拒绝（保证簇首稳定性）
                return self._reject_request(msg)

        # 规则4：数据分布相似度检查（分布感知聚类）
        if (self.label_distribution is not None and
                msg.sender_label_distribution is not None and
                self.distribution_similarity_threshold > 0):
            jsd = self._compute_js_divergence(
                self.label_distribution, msg.sender_label_distribution
            )
            if jsd > self.distribution_similarity_threshold:
                # 分布差异太大，拒绝
                return self._reject_request(msg)

        # 规则5：无父节点 + 未触发 → 接受
        return self._accept_request(msg)

    def _accept_request(self, msg: ClusterMessage) -> ClusterMessage:
        """
        接受聚合请求

        Args:
            msg: 聚合请求消息

        Returns:
            ACCEPT响应消息
        """
        self.parent = msg.sender_id
        # 如果消息中的cluster_id是None，说明发送方还没有加入任何簇
        # 此时发送方（父节点）将成为簇首，cluster_id设置为父节点的ID
        self.cluster_id = msg.cluster_id if msg.cluster_id is not None else msg.sender_id
        self.state = ClusterState.MEMBER

        return ClusterMessage(
            msg_type=ClusterMessageType.ACCEPT,
            sender_id=self.client_id,
            receiver_id=msg.sender_id,
            child_count=self.child_count,
            cluster_id=self.cluster_id
        )

    def _reject_request(self, msg: ClusterMessage) -> ClusterMessage:
        """
        拒绝聚合请求

        Args:
            msg: 聚合请求消息

        Returns:
            REJECT响应消息
        """
        return ClusterMessage(
            msg_type=ClusterMessageType.REJECT,
            sender_id=self.client_id,
            receiver_id=msg.sender_id,
            child_count=self.child_count,
            cluster_id=self.cluster_id if self.cluster_id else -1
        )

    def receive_accept(self, msg: ClusterMessage) -> List[tuple]:
        """
        处理收到的ACCEPT消息，并尝试继续扩展子节点

        Args:
            msg: ACCEPT消息

        Returns:
            需要发送的新请求消息列表（继续扩展子节点）
        """
        # 更新邻居状态
        if msg.sender_id in self.neighbor_state:
            self.neighbor_state[msg.sender_id] = DistributedClustering.NeighborState.ACCEPTED
        
        # 重置挂起请求状态
        self.pending_request = None

        if msg.sender_id not in self.children:
            self.children.append(msg.sender_id)
            self.child_count += 1
        
        # 如果没有父节点且还不是簇首，成为簇首
        if self.parent is None and self.state != ClusterState.CLUSTER_HEAD:
            self.state = ClusterState.CLUSTER_HEAD
            self.cluster_id = self.client_id

        # 尝试继续扩展子节点（如果还有容量）
        if self.child_count < self.max_child:
            # 获取可请求的邻居（排除父节点、已接受和挂起的邻居）
            remaining_neighbors = [
                n for n in self.neighbors
                if n != self.parent and 
                   self.neighbor_state.get(n) != DistributedClustering.NeighborState.ACCEPTED and
                   self.neighbor_state.get(n) != DistributedClustering.NeighborState.PENDING
            ]

            if remaining_neighbors:
                # 按资源排序邻居
                sorted_neighbors = self._sort_neighbors_by_resource(remaining_neighbors)
                # 继续发送下一个请求
                return self._send_next_request(sorted_neighbors)

        return []

    def receive_reject(self, msg: ClusterMessage) -> List[tuple]:
        """
        处理收到的REJECT消息，尝试下一个邻居

        修改后逻辑：
        - 已触发的节点（无论是簇首还是成员）都可以继续尝试其他邻居
        - 使用邻居状态机（UNTRIED/PENDING/FAILED/ACCEPTED）管理
        - 真正串行发送：一次只发送一个请求

        Args:
            msg: REJECT消息

        Returns:
            需要发送的新请求消息列表
        """
        # 只有已触发的节点才能继续尝试
        if self.state != ClusterState.TRIGGERED and self.state != ClusterState.CLUSTER_HEAD:
            return []

        # 更新邻居状态为失败
        if msg.sender_id in self.neighbor_state:
            self.neighbor_state[msg.sender_id] = DistributedClustering.NeighborState.FAILED
        
        # 增加重试计数
        if msg.sender_id in self.retry_count:
            self.retry_count[msg.sender_id] += 1
        
        # 重置挂起请求状态
        self.pending_request = None

        # 获取可请求的邻居（排除父节点、已接受和挂起的邻居）
        remaining_neighbors = [
            n for n in self.neighbors
            if n != self.parent and 
               self.neighbor_state.get(n) != DistributedClustering.NeighborState.ACCEPTED and
               self.neighbor_state.get(n) != DistributedClustering.NeighborState.PENDING
        ]

        if not remaining_neighbors:
            # 没有可请求的邻居了
            if self.parent is None and self.state != ClusterState.CLUSTER_HEAD:
                # 没有父节点且还不是簇首，成为簇首
                self.state = ClusterState.CLUSTER_HEAD
                self.cluster_id = self.client_id
            return []

        # 按资源排序邻居
        sorted_neighbors = self._sort_neighbors_by_resource(remaining_neighbors)

        # 真正串行发送：只发送下一个请求
        return self._send_next_request(sorted_neighbors)

    def get_cluster_info(self) -> Dict:
        """
        获取聚类信息

        Returns:
            聚类信息字典
        """
        return {
            'client_id': self.client_id,
            'state': self.state.value,
            'parent': self.parent,
            'children': self.children.copy(),
            'child_count': self.child_count,
            'cluster_id': self.cluster_id,
            'trigger_time': self.cluster_trigger_time,
            'resource_score': self.resource_score
        }

    def is_cluster_head(self) -> bool:
        """
        检查是否为簇首

        Returns:
            是否为簇首
        """
        return self.state == ClusterState.CLUSTER_HEAD

    def set_neighbors(self, neighbors: List[int]):
        """
        更新邻居列表

        Args:
            neighbors: 新的邻居列表
        """
        self.neighbors = neighbors
        for n in neighbors:
            if n not in self.retry_count:
                self.retry_count[n] = 0
            if n not in self.neighbor_state:
                self.neighbor_state[n] = DistributedClustering.NeighborState.UNTRIED


class ClusteringManager:
    """
    聚类管理器：协调所有节点的聚类过程

    负责：
    1. 初始化聚类状态
    2. 按触发时间顺序触发聚类
    3. 处理聚类消息
    4. 收集聚类结果
    """

    def __init__(self, clients: List, topology: List[List[int]],
                 election_results: Dict, max_child: int = 3,
                 max_retry: int = 3, request_timeout: float = 2.0,
                 cluster_offset: float = 30.0,
                 label_distributions: List = None,
                 distribution_similarity_threshold: float = 0.5):
        """
        初始化聚类管理器

        Args:
            clients: 客户端Actor列表
            topology: 拓扑信息
            election_results: 选举结果（包含资源得分和触发时间）
            max_child: 最大子节点数
            max_retry: 最大重试次数
            request_timeout: 请求超时时间
            cluster_offset: 聚类触发时间偏移量
            label_distributions: 每个客户端的标签分布列表
            distribution_similarity_threshold: 分布相似度阈值（JSD），低于此值才接受聚合
        """
        self.clients = clients
        self.topology = topology
        self.num_clients = len(clients)
        self.election_results = election_results

        self.max_child = max_child
        self.max_retry = max_retry
        self.request_timeout = request_timeout
        self.cluster_offset = cluster_offset

        self.label_distributions = label_distributions
        self.distribution_similarity_threshold = distribution_similarity_threshold

        self.clustering_states: List[DistributedClustering] = []
        self._init_clustering_states()

        self.clustering_complete = False
        self.cluster_heads = []

    def _init_clustering_states(self):
        """初始化所有节点的聚类状态"""
        for i in range(self.num_clients):
            neighbors = self.topology[i] if i < len(self.topology) else []

            # 从选举结果中获取资源得分和选举触发时间
            node_info = self.election_results['all_nodes_info'][i]
            resource_score = node_info['my_score']
            election_trigger = node_info['my_trigger_time']

            # 获取该节点的标签分布
            label_dist = (self.label_distributions[i]
                          if self.label_distributions and i < len(self.label_distributions)
                          else None)

            state = DistributedClustering(
                client_id=i,
                neighbors=neighbors,
                resource_score=resource_score,
                election_trigger_time=election_trigger,
                max_child=self.max_child,
                max_retry=self.max_retry,
                request_timeout=self.request_timeout,
                cluster_offset=self.cluster_offset,
                label_distribution=label_dist,
                distribution_similarity_threshold=self.distribution_similarity_threshold
            )

            self.clustering_states.append(state)

    def run_clustering(self) -> Dict:
        """
        运行分布式聚类

        Returns:
            聚类结果字典
        """
        print(f"\n{'='*60}")
        print(f"Running Distributed Clustering")
        print(f"{'='*60}")

        # 打印聚类触发时间
        print("\nCluster Trigger Times:")
        trigger_times = []
        for i in range(self.num_clients):
            info = self.clustering_states[i].get_cluster_info()
            trigger_times.append((info['trigger_time'], i))
            print(f"Node {i}: score={info['resource_score']:.2f}, "
                  f"trigger_time={info['trigger_time']:.2f}s")

        # 按触发时间排序
        trigger_times.sort(key=lambda x: x[0])

        message_queues = {i: [] for i in range(self.num_clients)}

        # 初始化触发事件
        for t, node_id in trigger_times:
            message_queues[node_id].append(('trigger', None))

        # 处理消息队列（使用事件驱动方式）
        # 使用一个列表来模拟事件队列，按时间顺序处理
        event_queue = []
        
        # 初始化触发事件（带时间戳）
        for t, node_id in trigger_times:
            event_queue.append((t, node_id, 'trigger', None))
        
        # 按时间排序事件队列
        event_queue.sort(key=lambda x: x[0])
        
        current_time = 0.0
        
        while event_queue:
            # 获取最早的事件
            event_time, node_id, event_type, data = event_queue.pop(0)
            current_time = event_time
            
            if event_type == 'trigger':
                messages = self.clustering_states[node_id].trigger()
                for neighbor_id, msg in messages:
                    # 计算消息传递延迟（模拟网络延迟）
                    delay = 0.1  # 固定延迟
                    event_queue.append((current_time + delay, neighbor_id, 'agg_req', msg))
                    # 重新排序事件队列
                    event_queue.sort(key=lambda x: x[0])

            elif event_type == 'agg_req':
                response = self.clustering_states[node_id].receive_agg_req(data)
                if response:
                    delay = 0.1
                    event_queue.append((current_time + delay, response.receiver_id, 
                                       response.msg_type.value, response))
                    event_queue.sort(key=lambda x: x[0])

            elif event_type == 'accept':
                messages = self.clustering_states[node_id].receive_accept(data)
                for neighbor_id, msg in messages:
                    delay = 0.1
                    event_queue.append((current_time + delay, neighbor_id, 'agg_req', msg))
                    event_queue.sort(key=lambda x: x[0])

            elif event_type == 'reject':
                messages = self.clustering_states[node_id].receive_reject(data)
                for neighbor_id, msg in messages:
                    delay = 0.1
                    event_queue.append((current_time + delay, neighbor_id, 'agg_req', msg))
                    event_queue.sort(key=lambda x: x[0])

        return self._collect_clustering_results()

    def _collect_clustering_results(self) -> Dict:
        """收集聚类结果"""
        all_info = []
        cluster_heads = []
        clusters = {}

        for i in range(self.num_clients):
            info = self.clustering_states[i].get_cluster_info()
            all_info.append(info)

            # 簇首的条件：状态为cluster_head 或者 没有父节点（可能从未被触发过）
            if info['state'] == 'cluster_head' or info['parent'] is None:
                cluster_heads.append(i)
                clusters[i] = {'head': i, 'members': []}

        # 统计每个簇的成员（有父节点的节点都是成员）
        for info in all_info:
            # 有父节点的节点都是成员
            if info['parent'] is not None and info['cluster_id'] in clusters:
                clusters[info['cluster_id']]['members'].append(info['client_id'])

        result = {
            'cluster_heads': cluster_heads,
            'num_clusters': len(cluster_heads),
            'clusters': clusters,
            'all_nodes_info': all_info
        }

        # 打印结果
        print(f"\n{'='*60}")
        print(f"Clustering Complete!")
        print(f"Number of clusters: {len(cluster_heads)}")
        print(f"Cluster heads: {cluster_heads}")
        print(f"{'='*60}\n")

        for cluster_id, cluster_info in clusters.items():
            members = cluster_info['members']
            print(f"Cluster {cluster_id} (head):")
            print(f"  Members: {members}")
            print(f"  Size: {len(members) + 1}")

        self.clustering_complete = True
        self.cluster_heads = cluster_heads

        return result

    def get_cluster_heads(self) -> List[int]:
        """
        获取所有簇首ID

        Returns:
            簇首ID列表
        """
        return self.cluster_heads

    def get_cluster_info(self, node_id: int) -> Dict:
        """
        获取指定节点的聚类信息

        Args:
            node_id: 节点ID

        Returns:
            聚类信息
        """
        if 0 <= node_id < len(self.clustering_states):
            return self.clustering_states[node_id].get_cluster_info()
        return {}

    def get_cluster_members(self, cluster_id: int) -> List[int]:
        """
        获取指定簇的所有成员

        Args:
            cluster_id: 簇ID

        Returns:
            成员ID列表
        """
        members = []
        for i in range(self.num_clients):
            info = self.clustering_states[i].get_cluster_info()
            if info['cluster_id'] == cluster_id:
                members.append(i)
        return members
