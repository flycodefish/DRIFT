from __future__ import annotations

import json
import math
import numpy as np
import os
import random
import time
import warnings
from pathlib import Path

# 尝试导入网络可视化库
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

warnings.filterwarnings(
    "ignore",
    message=r"The pynvml package is deprecated\\..*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Tip: In future versions of Ray, Ray will no longer override accelerator visible devices env var.*",
    category=FutureWarning,
)
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

import ray
import torch

from .actors import ClientActor, MLP, estimate_state_dict_bytes, get_model_params
from .config import FLConfig
from .monitor import Monitor
from .topology import build_neighbors


class RayFLRuntime:
    """Low-level runtime: node lifecycle, topology, comm stats, and eval."""

    def __init__(self, cfg: FLConfig) -> None:
        self.cfg = cfg
        self.monitor = Monitor()
        self.clients = []
        self.client_step_delays_ms: list[float] = []
        self.slow_network_client_ids: set[int] = set()
        self.low_storage_client_ids: set[int] = set()
        self.edge_overrides: dict[str, dict[str, float]] = {}
        self.neighbors: dict[int, list[int]] = {}
        self.train_shards = []
        self.client_label_stats: list[dict[int, int]] = []
        self.val_x = None
        self.val_y = None
        self.run_name = f"{cfg.program_name}_{cfg.topology}_{cfg.num_clients}c_{cfg.rounds}r"

    def start(self) -> None:
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        ray.init(
            ignore_reinit_error=True,
            logging_level="ERROR",
            _skip_env_hook=True,
        )
        self._build_data()
        self._build_client_speed_profile()
        self._build_client_network_profile()
        self._build_client_storage_profile()
        self._build_edge_overrides()
        self._create_clients()
        self.neighbors = build_neighbors(
            self.cfg.num_clients,
            self.cfg.topology,
            self.cfg.neighbor_strategy,
            self.cfg.neighbor_k,
        )
        # 生成拓扑图
        self._generate_topology_graph()

    def _build_client_speed_profile(self) -> None:
        self.client_step_delays_ms = [0.0 for _ in range(self.cfg.num_clients)]
        if self.cfg.client_speed_mode == "uniform" or self.cfg.slow_client_delay_ms <= 0:
            return
        if self.cfg.client_speed_mode != "two_group":
            raise ValueError(f"Unsupported client_speed_mode: {self.cfg.client_speed_mode}")

        num_slow = int(self.cfg.num_clients * self.cfg.slow_client_ratio)
        num_slow = min(max(num_slow, 0), self.cfg.num_clients)
        if num_slow == 0:
            return

        rng = random.Random(self.cfg.seed)
        slow_client_ids = set(rng.sample(range(self.cfg.num_clients), k=num_slow))
        for client_id in slow_client_ids:
            self.client_step_delays_ms[client_id] = self.cfg.slow_client_delay_ms

    def get_client_speed_profile(self) -> list[float]:
        return self.client_step_delays_ms

    def _build_client_network_profile(self) -> None:
        self.slow_network_client_ids = set()
        if self.cfg.client_network_mode == "uniform":
            return
        if self.cfg.client_network_mode != "two_group":
            raise ValueError(f"Unsupported client_network_mode: {self.cfg.client_network_mode}")

        num_slow = int(self.cfg.num_clients * self.cfg.slow_network_client_ratio)
        num_slow = min(max(num_slow, 0), self.cfg.num_clients)
        if num_slow == 0:
            return

        rng = random.Random(self.cfg.seed + 17)
        self.slow_network_client_ids = set(rng.sample(range(self.cfg.num_clients), k=num_slow))

    def get_slow_network_client_ids(self) -> list[int]:
        return sorted(self.slow_network_client_ids)

    def _build_client_storage_profile(self) -> None:
        self.low_storage_client_ids = set()
        if self.cfg.client_storage_mode == "uniform" or self.cfg.low_storage_max_upload_bytes <= 0:
            return
        if self.cfg.client_storage_mode != "two_group":
            raise ValueError(f"Unsupported client_storage_mode: {self.cfg.client_storage_mode}")

        num_low_storage = int(self.cfg.num_clients * self.cfg.low_storage_client_ratio)
        num_low_storage = min(max(num_low_storage, 0), self.cfg.num_clients)
        if num_low_storage == 0:
            return

        rng = random.Random(self.cfg.seed + 23)
        self.low_storage_client_ids = set(rng.sample(range(self.cfg.num_clients), k=num_low_storage))

    def get_low_storage_client_ids(self) -> list[int]:
        return sorted(self.low_storage_client_ids)

    def _build_edge_overrides(self) -> None:
        self.edge_overrides = {}
        if not self.cfg.edge_profile_json:
            return

        try:
            raw = json.loads(self.cfg.edge_profile_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid edge_profile_json: {exc}") from exc

        if not isinstance(raw, dict):
            raise ValueError("edge_profile_json must be a JSON object")

        for edge_key, edge_profile in raw.items():
            if not isinstance(edge_key, str) or "->" not in edge_key:
                raise ValueError(f"Invalid edge override key: {edge_key}")
            if not isinstance(edge_profile, dict):
                raise ValueError(f"Invalid edge override profile for {edge_key}")

            normalized = {}
            if "bandwidth_mbps" in edge_profile:
                normalized["bandwidth_mbps"] = max(float(edge_profile["bandwidth_mbps"]), 1e-6)
            if "latency_ms" in edge_profile:
                normalized["latency_ms"] = max(float(edge_profile["latency_ms"]), 0.0)
            if "drop_rate" in edge_profile:
                normalized["drop_rate"] = min(max(float(edge_profile["drop_rate"]), 0.0), 1.0)
            self.edge_overrides[edge_key] = normalized

    def get_edge_override_keys(self) -> list[str]:
        return sorted(self.edge_overrides.keys())

    def shutdown(self) -> None:
        self.monitor.dump(self.cfg.output_dir, self.run_name)
        self._dump_client_profiles()
        ray.shutdown()

    def _dump_client_profiles(self) -> None:
        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        profile_path = out_dir / f"{self.run_name}_client_profiles.json"

        payload = {
            "label_stats": self.client_label_stats,
            "step_delays_ms": self.client_step_delays_ms,
            "slow_network_clients": self.get_slow_network_client_ids(),
            "low_storage_clients": self.get_low_storage_client_ids(),
        }
        profile_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _build_data(self) -> None:
        # 检查是否使用CIFAR-10数据集
        if hasattr(self.cfg, 'dataset') and self.cfg.dataset == 'cifar10':
            self._build_cifar10_data()
            return

        # 检查是否使用NEU数据集
        if hasattr(self.cfg, 'dataset') and self.cfg.dataset == 'neu':
            self._build_neu_data()
            return

        # 检查是否使用HAR数据集
        if hasattr(self.cfg, 'dataset') and self.cfg.dataset == 'har':
            self._build_har_data()
            return

        # 检查是否使用CIFAR-100数据集
        if hasattr(self.cfg, 'dataset') and self.cfg.dataset == 'cifar100':
            self._build_cifar100_data()
            return
        
        # 生成合成数据
        generator = torch.Generator().manual_seed(self.cfg.seed)
        features = torch.randn(self.cfg.num_samples, self.cfg.input_dim, generator=generator)
        true_w = torch.randn(self.cfg.input_dim, self.cfg.num_classes, generator=generator)
        logits = features @ true_w + 0.3 * torch.randn(
            self.cfg.num_samples, self.cfg.num_classes, generator=generator
        )
        labels = torch.argmax(logits, dim=1)

        split = int(0.8 * self.cfg.num_samples)
        train_x, train_y = features[:split], labels[:split]
        self.val_x, self.val_y = features[split:], labels[split:]

        self.train_shards = []
        self.client_label_stats = []

        if self.cfg.data_partition_mode == "iid":
            self._build_iid_shards(train_x, train_y)
            return

        if self.cfg.data_partition_mode == "label_skew":
            self._build_label_skew_shards(train_x, train_y)
            return

        if self.cfg.data_partition_mode == "dirichlet":
            self._build_dirichlet_shards(train_x, train_y)
            return

        raise ValueError(f"Unsupported data_partition_mode: {self.cfg.data_partition_mode}")

    def _build_cifar10_data(self) -> None:
        """Load CIFAR-10 through the partitioner shared with DRIFT."""
        from fl_core.data_loader import load_cifar10_data

        train_data, val_data, _, _, _ = load_cifar10_data(
            data_dir=self.cfg.data_dir,
            num_clients=self.cfg.num_clients,
            partition_mode=self.cfg.data_partition_mode,
            alpha=self.cfg.dirichlet_alpha,
        )
        self.train_shards = [
            (item['images'], item['labels']) for item in train_data
        ]
        self.client_label_stats = [
            self._count_labels(item['labels']) for item in train_data
        ]
        self.val_x = torch.cat([item['images'] for item in val_data], dim=0)
        self.val_y = torch.cat([item['labels'] for item in val_data], dim=0)

    def _build_cifar100_data(self) -> None:
        """Load and partition CIFAR-100 dataset."""
        import torchvision

        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=None)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=None)

        all_data = np.concatenate([trainset.data, testset.data], axis=0)
        all_labels = np.concatenate([np.array(trainset.targets), np.array(testset.targets)], axis=0)

        all_data = torch.tensor(all_data).permute(0, 3, 1, 2).float() / 255.0
        mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(1, 3, 1, 1)
        std = torch.tensor([0.2675, 0.2565, 0.2761]).view(1, 3, 1, 1)
        all_data = (all_data - mean) / std
        all_labels = torch.tensor(all_labels)

        split = int(0.8 * len(all_data))
        train_x, train_y = all_data[:split], all_labels[:split]
        self.val_x, self.val_y = all_data[split:], all_labels[split:]

        self.train_shards = []
        self.client_label_stats = []

        if self.cfg.data_partition_mode == "iid":
            self._build_iid_shards(train_x, train_y)
            return

        if self.cfg.data_partition_mode == "label_skew":
            self._build_label_skew_shards(train_x, train_y)
            return

        if self.cfg.data_partition_mode == "dirichlet":
            self._build_dirichlet_shards(train_x, train_y)
            return

        raise ValueError(f"Unsupported data_partition_mode: {self.cfg.data_partition_mode}")

    def _build_neu_data(self) -> None:
        """Load and partition NEU-DET steel surface defect dataset."""
        from fl_core.data_loader import load_neu_data

        train_data, val_data, _, _, _ = load_neu_data(
            data_dir='./data',
            num_clients=self.cfg.num_clients,
            partition_mode=self.cfg.data_partition_mode,
            alpha=self.cfg.dirichlet_alpha,
        )

        self.train_shards = []
        self.client_label_stats = []
        for client_id in range(self.cfg.num_clients):
            x = train_data[client_id]['images']
            y = train_data[client_id]['labels']
            self.train_shards.append((x, y))
            self.client_label_stats.append(self._count_labels(y))

        # 合并所有客户端验证数据作为全局验证集
        all_val_x = [val_data[cid]['images'] for cid in range(self.cfg.num_clients)]
        all_val_y = [val_data[cid]['labels'] for cid in range(self.cfg.num_clients)]
        self.val_x = torch.cat(all_val_x, dim=0)
        self.val_y = torch.cat(all_val_y, dim=0)

        print(f"NEU dataset loaded: {sum(len(s[0]) for s in self.train_shards)} train → {self.cfg.num_clients} clients, "
              f"{len(self.val_x)} val")

    def _build_har_data(self) -> None:
        """Load and partition UCI HAR human activity recognition dataset."""
        from fl_core.data_loader import load_har_data

        train_data, val_data, _, _, _ = load_har_data(
            data_dir='./data',
            num_clients=self.cfg.num_clients,
            partition_mode=self.cfg.data_partition_mode,
            alpha=self.cfg.dirichlet_alpha,
        )

        self.train_shards = []
        self.client_label_stats = []
        for client_id in range(self.cfg.num_clients):
            x = train_data[client_id]['images']
            y = train_data[client_id]['labels']
            self.train_shards.append((x, y))
            self.client_label_stats.append(self._count_labels(y))

        all_val_x = [val_data[cid]['images'] for cid in range(self.cfg.num_clients)]
        all_val_y = [val_data[cid]['labels'] for cid in range(self.cfg.num_clients)]
        self.val_x = torch.cat(all_val_x, dim=0)
        self.val_y = torch.cat(all_val_y, dim=0)

        print(f"HAR dataset loaded: {sum(len(s[0]) for s in self.train_shards)} train → {self.cfg.num_clients} clients, "
              f"{len(self.val_x)} val")

    def _build_iid_shards(self, train_x: torch.Tensor, train_y: torch.Tensor) -> None:
        indices = torch.randperm(train_x.shape[0])
        shard_size = train_x.shape[0] // self.cfg.num_clients
        for client_id in range(self.cfg.num_clients):
            start = client_id * shard_size
            end = train_x.shape[0] if client_id == self.cfg.num_clients - 1 else start + shard_size
            shard_idx = indices[start:end]
            shard_y = train_y[shard_idx]
            self.train_shards.append((train_x[shard_idx], shard_y))
            self.client_label_stats.append(self._count_labels(shard_y))

    def _build_label_skew_shards(self, train_x: torch.Tensor, train_y: torch.Tensor) -> None:
        class_to_indices: dict[int, list[int]] = {}
        for class_id in range(self.cfg.num_classes):
            mask = train_y == class_id
            class_indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
            random.shuffle(class_indices)
            class_to_indices[class_id] = class_indices

        samples_per_client = train_x.shape[0] // self.cfg.num_clients
        classes_per_client = max(1, min(self.cfg.classes_per_client, self.cfg.num_classes))

        for client_id in range(self.cfg.num_clients):
            allowed_classes = [
                (client_id + offset) % self.cfg.num_classes for offset in range(classes_per_client)
            ]
            shard_indices: list[int] = []
            turn = 0
            while len(shard_indices) < samples_per_client:
                class_id = allowed_classes[turn % len(allowed_classes)]
                if class_to_indices[class_id]:
                    shard_indices.append(class_to_indices[class_id].pop())
                else:
                    fallback_class = next(
                        (candidate for candidate, pool in class_to_indices.items() if pool),
                        None,
                    )
                    if fallback_class is None:
                        break
                    shard_indices.append(class_to_indices[fallback_class].pop())
                turn += 1

            shard_idx_tensor = torch.tensor(shard_indices, dtype=torch.long)
            shard_y = train_y[shard_idx_tensor]
            self.train_shards.append((train_x[shard_idx_tensor], shard_y))
            self.client_label_stats.append(self._count_labels(shard_y))

        remaining_indices = []
        for pool in class_to_indices.values():
            remaining_indices.extend(pool)
        for offset, index in enumerate(remaining_indices):
            client_id = offset % self.cfg.num_clients
            shard_x, shard_y = self.train_shards[client_id]
            extra_x = train_x[index].unsqueeze(0)
            extra_y = train_y[index].unsqueeze(0)
            updated_y = torch.cat([shard_y, extra_y], dim=0)
            self.train_shards[client_id] = (torch.cat([shard_x, extra_x], dim=0), updated_y)
            self.client_label_stats[client_id] = self._count_labels(updated_y)

    def _build_dirichlet_shards(self, train_x: torch.Tensor, train_y: torch.Tensor) -> None:
        alpha = float(max(self.cfg.dirichlet_alpha, 1e-6))
        client_indices: list[list[int]] = [[] for _ in range(self.cfg.num_clients)]

        for class_id in range(self.cfg.num_classes):
            class_mask = train_y == class_id
            class_indices = torch.nonzero(class_mask, as_tuple=False).flatten().tolist()
            if not class_indices:
                continue
            random.shuffle(class_indices)

            concentration = torch.full((self.cfg.num_clients,), alpha)
            proportions = torch.distributions.Dirichlet(concentration).sample()
            raw_counts = proportions * len(class_indices)
            counts = torch.floor(raw_counts).to(torch.int64)
            remain = len(class_indices) - int(counts.sum().item())
            if remain > 0:
                frac = raw_counts - counts.float()
                topk = torch.topk(frac, k=remain).indices.tolist()
                for idx in topk:
                    counts[idx] += 1

            cursor = 0
            for client_id, count in enumerate(counts.tolist()):
                if count <= 0:
                    continue
                selected = class_indices[cursor : cursor + count]
                client_indices[client_id].extend(selected)
                cursor += count

        # Keep clients non-empty by borrowing from the largest shard when needed.
        min_samples = max(self.cfg.batch_size, 1)
        for client_id in range(self.cfg.num_clients):
            deficit = min_samples - len(client_indices[client_id])
            if deficit <= 0:
                continue
            donors = sorted(
                range(self.cfg.num_clients),
                key=lambda idx: len(client_indices[idx]),
                reverse=True
            )
            for donor_id in donors:
                if deficit <= 0:
                    break
                if donor_id == client_id:
                    continue
                take = min(deficit, len(client_indices[donor_id]) - min_samples)
                if take > 0:
                    client_indices[client_id].extend(client_indices[donor_id][-take:])
                    client_indices[donor_id] = client_indices[donor_id][:-take]
                    deficit -= take

        for client_id in range(self.cfg.num_clients):
            shard_idx = torch.tensor(client_indices[client_id], dtype=torch.long)
            shard_y = train_y[shard_idx]
            self.train_shards.append((train_x[shard_idx], shard_y))
            self.client_label_stats.append(self._count_labels(shard_y))

    @staticmethod
    def _count_labels(labels: torch.Tensor) -> dict[int, int]:
        values, counts = torch.unique(labels, return_counts=True)
        return {int(label): int(count) for label, count in zip(values.tolist(), counts.tolist())}

    def get_client_label_stats(self) -> list[dict[int, int]]:
        return self.client_label_stats

    def _create_clients(self) -> None:
        # 检查是否使用CIFAR-10数据集
        dataset = getattr(self.cfg, 'dataset', 'synthetic')
        # 检查使用哪种模型
        model = getattr(self.cfg, 'model', 'cnn')

        # 使用顺序for循环而非列表推导，错开Actor创建时机
        # 结合 lazy CUDA init，避免50个Actor同时初始化CUDA导致cuBLAS handle耗尽
        self.clients = []
        actor_gpu = (self.cfg.num_gpus
                     if self.cfg.use_gpu and torch.cuda.is_available()
                     else 0.0)
        for client_id, (shard_x, shard_y) in enumerate(self.train_shards):
            actor = ClientActor.options(num_gpus=actor_gpu).remote(
                client_id=client_id,
                features=shard_x,
                labels=shard_y,
                input_dim=self.cfg.input_dim,
                num_classes=self.cfg.num_classes,
                batch_size=self.cfg.batch_size,
                learning_rate=self.cfg.learning_rate,
                local_epochs=self.cfg.local_epochs,
                step_delay_ms=self.client_step_delays_ms[client_id],
                dataset=dataset,
                model=model,
                seed=self.cfg.seed + client_id,
            )
            self.clients.append(actor)

    def _warmup_cuda_sequential(self):
        """顺序预热每个客户端的CUDA上下文，避免50个Actor同时初始化cuBLAS导致句柄耗尽"""
        import time as _time
        t0 = _time.perf_counter()
        for i, client in enumerate(self.clients):
            ray.get(client._warmup_cuda.remote())
            if (i + 1) % 10 == 0:
                print(f"  CUDA warmup: {i+1}/{len(self.clients)}")
        elapsed = _time.perf_counter() - t0
        print(f"CUDA warmup complete ({len(self.clients)} clients in {elapsed:.1f}s)")

    def init_model_params(self):
        # 根据数据集类型和模型选择参数选择模型
        if hasattr(self.cfg, 'dataset') and self.cfg.dataset in ('cifar10', 'neu', 'cifar100'):
            if hasattr(self.cfg, 'model') and self.cfg.model == 'resnet18':
                from .actors import ResNet18
                model = ResNet18(self.cfg.num_classes)
            else:
                from .actors import CNN
                model = CNN(self.cfg.num_classes)
        elif hasattr(self.cfg, 'dataset') and self.cfg.dataset == 'har':
            from .actors import HARMLP
            model = HARMLP(self.cfg.input_dim, self.cfg.num_classes)
        else:
            from .actors import MLP
            model = MLP(self.cfg.input_dim, self.cfg.num_classes)
        
        # 检查是否使用CUDA
        if torch.cuda.is_available():
            model = model.cuda()
        
        return get_model_params(model)

    def _resolve_link_profile(self, kind: str) -> dict[str, float]:
        if self.cfg.link_profile_mode != "role_aware":
            return {
                "bandwidth_mbps": self.cfg.link_bandwidth_mbps,
                "latency_ms": self.cfg.link_latency_ms,
                "drop_rate": self.cfg.link_drop_rate,
            }

        if kind in ("broadcast", "upload"):
            return {
                "bandwidth_mbps": self.cfg.server_link_bandwidth_mbps or self.cfg.link_bandwidth_mbps,
                "latency_ms": self.cfg.server_link_latency_ms or self.cfg.link_latency_ms,
                "drop_rate": self.cfg.server_link_drop_rate
                if self.cfg.server_link_drop_rate is not None
                else self.cfg.link_drop_rate,
            }

        return {
            "bandwidth_mbps": self.cfg.peer_link_bandwidth_mbps or self.cfg.link_bandwidth_mbps,
            "latency_ms": self.cfg.peer_link_latency_ms or self.cfg.link_latency_ms,
            "drop_rate": self.cfg.peer_link_drop_rate
            if self.cfg.peer_link_drop_rate is not None
            else self.cfg.link_drop_rate,
        }

    @staticmethod
    def _extract_client_id(endpoint: str) -> int | None:
        if endpoint.startswith("client_"):
            try:
                return int(endpoint.split("_", 1)[1])
            except ValueError:
                return None
        return None

    def _apply_client_network_adjustments(self, profile: dict[str, float], src: str, dst: str) -> dict[str, float]:
        if self.cfg.client_network_mode != "two_group":
            return profile

        src_id = self._extract_client_id(src)
        dst_id = self._extract_client_id(dst)
        affected = False
        if src_id is not None and src_id in self.slow_network_client_ids:
            affected = True
        if dst_id is not None and dst_id in self.slow_network_client_ids:
            affected = True
        if not affected:
            return profile

        return {
            "bandwidth_mbps": max(profile["bandwidth_mbps"] * self.cfg.slow_network_bandwidth_multiplier, 1e-6),
            "latency_ms": max(profile["latency_ms"] * self.cfg.slow_network_latency_multiplier, 0.0),
            "drop_rate": min(max(profile["drop_rate"] + self.cfg.slow_network_drop_addition, 0.0), 1.0),
        }

    def _apply_edge_overrides(self, profile: dict[str, float], src: str, dst: str) -> dict[str, float]:
        if not self.edge_overrides:
            return profile

        # Priority: exact edge > src wildcard > dst wildcard
        candidates = [f"{src}->{dst}", f"{src}->*", f"*->{dst}"]
        merged = dict(profile)
        for key in candidates:
            override = self.edge_overrides.get(key)
            if not override:
                continue
            merged.update(override)
            break
        return merged

    def _estimate_comm_latency_ms(self, payload_bytes: int, bandwidth_mbps: float, latency_ms: float) -> float:
        # tx_ms = payload_bits / (mbps * 1e6) * 1e3
        safe_bandwidth_mbps = max(bandwidth_mbps, 1e-6)
        tx_ms = (payload_bytes * 8.0) / (safe_bandwidth_mbps * 1_000_000.0) * 1000.0
        return float(latency_ms + tx_ms)

    def summarize_round_comm(self, round_id: int) -> dict[str, int]:
        events = [event for event in self.monitor.comm_events if event["round_id"] == round_id]
        success_count = sum(1 for event in events if event["delivered"])
        failure_count = len(events) - success_count
        timeout_count = sum(1 for event in events if event["timeout_occurred"])
        kind_summary: dict[str, int] = {}
        for kind in ("broadcast", "upload", "upload_storage_blocked", "gossip"):
            kind_events = [event for event in events if event["kind"] == kind]
            kind_summary[f"{kind}_success_count"] = sum(1 for event in kind_events if event["delivered"])
            kind_summary[f"{kind}_failure_count"] = len(kind_events) - kind_summary[f"{kind}_success_count"]
        return {
            "comm_success_count": success_count,
            "comm_failure_count": failure_count,
            "comm_timeout_count": timeout_count,
            "upload_storage_blocked_count": kind_summary["upload_storage_blocked_failure_count"],
            **kind_summary,
        }

    def _record_upload_storage_blocked(self, round_id: int, client_id: int, payload_bytes: int) -> None:
        src = f"client_{client_id}"
        dst = "server"
        base_profile = self._resolve_link_profile("upload")
        link_profile = self._apply_client_network_adjustments(base_profile, src, dst)
        self.monitor.record_comm(
            round_id=round_id,
            src=src,
            dst=dst,
            payload_bytes=payload_bytes,
            kind="upload_storage_blocked",
            simulated_latency_ms=0.0,
            bandwidth_mbps=link_profile["bandwidth_mbps"],
            drop_rate=link_profile["drop_rate"],
            delivered=False,
            retries_used=0,
            timeout_occurred=False,
            total_backoff_ms=0.0,
        )

    def _record_comm(self, round_id: int, src: str, dst: str, payload_bytes: int, kind: str) -> bool:
        base_profile = self._resolve_link_profile(kind)
        network_adjusted = self._apply_client_network_adjustments(base_profile, src, dst)
        link_profile = self._apply_edge_overrides(network_adjusted, src, dst)
        attempt_latency_ms = self._estimate_comm_latency_ms(
            payload_bytes,
            link_profile["bandwidth_mbps"],
            link_profile["latency_ms"],
        )
        max_attempts = max(1, self.cfg.max_comm_retries + 1)
        delivered = False
        retries_used = 0
        total_latency_ms = 0.0
        timeout_occurred = False
        total_backoff_ms = 0.0

        for attempt in range(max_attempts):
            total_latency_ms += attempt_latency_ms
            attempt_timed_out = self.cfg.comm_timeout_ms > 0 and attempt_latency_ms > self.cfg.comm_timeout_ms
            if self.cfg.enforce_comm_delay:
                time.sleep(attempt_latency_ms / 1000.0)

            if attempt_timed_out:
                timeout_occurred = True
            elif random.random() >= link_profile["drop_rate"]:
                delivered = True
                retries_used = attempt
                break

            retries_used = attempt
            if attempt < max_attempts - 1 and self.cfg.backoff_base_ms > 0:
                backoff_ms = self.cfg.backoff_base_ms * (self.cfg.backoff_multiplier**attempt)
                total_backoff_ms += backoff_ms
                total_latency_ms += backoff_ms
                if self.cfg.enforce_comm_delay:
                    time.sleep(backoff_ms / 1000.0)

        if not delivered:
            retries_used = max_attempts - 1

        self.monitor.record_comm(
            round_id=round_id,
            src=src,
            dst=dst,
            payload_bytes=payload_bytes,
            kind=kind,
            simulated_latency_ms=total_latency_ms,
            bandwidth_mbps=link_profile["bandwidth_mbps"],
            drop_rate=link_profile["drop_rate"],
            delivered=delivered,
            retries_used=retries_used,
            timeout_occurred=timeout_occurred,
            total_backoff_ms=total_backoff_ms,
        )
        return delivered

    def broadcast_model(self, params, round_id: int) -> None:
        payload_bytes = estimate_state_dict_bytes(params)
        for client_id in range(self.cfg.num_clients):
            if self._record_comm(round_id, "server", f"client_{client_id}", payload_bytes, "broadcast"):
                ray.get(self.clients[client_id].receive_model.remote(params))

    def select_clients(self):
        sample_count = max(1, int(self.cfg.num_clients * self.cfg.client_sample_ratio))
        all_clients = list(range(self.cfg.num_clients))
        random.shuffle(all_clients)
        return all_clients[:sample_count]

    def inject_dropouts(self, selected_client_ids):
        active = []
        dropped = []
        for client_id in selected_client_ids:
            if random.random() < self.cfg.client_drop_rate:
                dropped.append(client_id)
            else:
                active.append(client_id)
        # Keep at least one active client so aggregation can proceed.
        if not active and dropped:
            recovered = dropped.pop(0)
            active.append(recovered)
        return active, dropped

    def train_selected_clients(self, selected_client_ids, round_id: int):
        # 在训练前更新学习率（动态学习率调度）
        self.adjust_learning_rate(round_id)
        
        refs = [self.clients[client_id].local_train.remote(round_id) for client_id in selected_client_ids]
        return ray.get(refs)
    
    def adjust_learning_rate(self, round_id: int):
        """
        Cosine Annealing with Warmup 学习率调度
        
        - Warmup (0 ~ warmup_rounds): 线性从 0 增长到 initial_lr
        - Cosine (warmup_rounds ~ max_rounds): 余弦曲线从 initial_lr 降到 min_lr
        
        Args:
            round_id: 当前训练轮次（从0开始）
        """
        import math
        
        initial_lr = float(self.cfg.learning_rate)
        min_lr = float(self.cfg.learning_rate_min)
        warmup_rounds = int(self.cfg.warmup_rounds)
        max_rounds = int(self.cfg.rounds)
        
        if self.cfg.cosine_annealing:
            # Warmup 阶段
            if round_id < warmup_rounds:
                # 线性增长：lr = initial_lr * (round_id / warmup_rounds)
                new_lr = initial_lr * ((round_id + 1) / max(1, warmup_rounds))
            else:
                # Cosine Annealing 阶段
                # 从 warmup_rounds 开始的进度：[0, max_rounds - warmup_rounds]
                progress = round_id - warmup_rounds
                total_cosine_steps = max_rounds - warmup_rounds
                # 余弦公式：lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(π * progress / total))
                new_lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * progress / total_cosine_steps))
        else:
            # 简单的指数衰减（备用）
            decay = 0.95
            new_lr = initial_lr * (decay ** round_id)
            new_lr = max(new_lr, min_lr)
        
        # 更新所有客户端的学习率
        refs = [self.clients[cid].set_learning_rate.remote(new_lr) for cid in range(self.cfg.num_clients)]
        ray.get(refs)
    
    def clear_all_client_caches(self):
        """
        强制所有客户端清理 GPU 显存缓存
        
        在每轮训练和 Gossip 交换后调用，释放不再使用的临时张量占用的显存
        """
        refs = [self.clients[cid].clear_gpu_cache.remote() for cid in range(self.cfg.num_clients)]
        ray.get(refs)

    def collect_client_models(self, selected_client_ids, round_id: int):
        refs = [self.clients[client_id].upload_model.remote() for client_id in selected_client_ids]
        payloads = ray.get(refs)
        delivered_payloads = []
        for item in payloads:
            payload_bytes = estimate_state_dict_bytes(item["params"])
            if (
                item["client_id"] in self.low_storage_client_ids
                and self.cfg.low_storage_max_upload_bytes > 0
                and payload_bytes > self.cfg.low_storage_max_upload_bytes
            ):
                self._record_upload_storage_blocked(round_id, item["client_id"], payload_bytes)
                continue
            delivered = self._record_comm(
                round_id,
                f"client_{item['client_id']}",
                "server",
                payload_bytes,
                "upload",
            )
            if delivered:
                delivered_payloads.append(item)
        return delivered_payloads

    def run_gossip_exchange(self, round_id: int, selected_client_ids=None) -> None:
        if self.cfg.topology not in ("ring", "mesh"):
            return

        # For mesh random-per-round mode, refresh neighbors every round.
        if self.cfg.topology == "mesh" and self.cfg.neighbor_strategy == "random_per_round":
            self.neighbors = build_neighbors(
                self.cfg.num_clients,
                self.cfg.topology,
                self.cfg.neighbor_strategy,
                self.cfg.neighbor_k,
            )

        if selected_client_ids is None:
            selected_client_ids = list(range(self.cfg.num_clients))
        selected_set = set(selected_client_ids)

        # 1) Pull all models once.
        all_models = ray.get([self.clients[cid].upload_model.remote() for cid in selected_client_ids])
        params_map = {item["client_id"]: item["params"] for item in all_models}

        # 2) Push each client's neighbor params and ask it to merge locally.
        refs = []
        for client_id in selected_client_ids:
            neighbor_ids = [nid for nid in self.neighbors[client_id] if nid in selected_set]
            if not neighbor_ids:
                continue
            delivered_neighbor_params = []
            delivered_neighbor_count = 0
            for neighbor_id in neighbor_ids:
                payload_bytes = estimate_state_dict_bytes(params_map[neighbor_id])
                delivered = self._record_comm(
                    round_id,
                    f"client_{neighbor_id}",
                    f"client_{client_id}",
                    payload_bytes,
                    "gossip",
                )
                if delivered:
                    delivered_neighbor_params.append(params_map[neighbor_id])
                    delivered_neighbor_count += 1

            if delivered_neighbor_count == 0:
                continue

            neighbor_params = delivered_neighbor_params
            refs.append(self.clients[client_id].gossip_exchange.remote(neighbor_params))
        if refs:
            ray.get(refs)

    def push_gossip_async(self, round_id: int, selected_client_ids: list) -> dict:
        """
        Asynchronous Push-based Gossip protocol.
        Each client pushes its model to random neighbors.
        Non-blocking: doesn't wait for all gossip to complete.
        
        Fix: Collect all senders for each receiver and call receive_gossip_push once
             with all senders' models, instead of calling it multiple times.
        """
        # Refresh neighbors for random-per-round strategy
        if self.cfg.neighbor_strategy == "random_per_round":
            self.neighbors = build_neighbors(
                self.cfg.num_clients,
                self.cfg.topology,
                self.cfg.neighbor_strategy,
                self.cfg.neighbor_k,
            )

        selected_set = set(selected_client_ids)
        
        # 获取所有选中客户端的模型
        all_models_data = ray.get([self.clients[cid].upload_model.remote() for cid in selected_client_ids])
        params_map = {item["client_id"]: item["params"] for item in all_models_data}
        
        # 记录通信统计
        gossip_success = 0
        gossip_failure = 0
        
        # 收集每个接收者应该收到的所有发送者及其模型
        # receiver_id -> {'senders': [sender_ids], 'params_list': [params_dicts]}
        receivers_data = {}
        
        for client_id in selected_client_ids:
            # 获取该客户端的邻居
            neighbor_ids = [nid for nid in self.neighbors[client_id] if nid in selected_set]
            
            if not neighbor_ids:
                continue
            
            # Push自己的模型给所有邻居
            for neighbor_id in neighbor_ids:
                payload_bytes = estimate_state_dict_bytes(params_map[client_id])
                delivered = self._record_comm(
                    round_id,
                    f"client_{client_id}",
                    f"client_{neighbor_id}",
                    payload_bytes,
                    "gossip_push",
                )
                if delivered:
                    gossip_success += 1
                    # 收集发送者信息到接收者的数据结构中
                    if neighbor_id not in receivers_data:
                        receivers_data[neighbor_id] = {'senders': [], 'params_list': []}
                    receivers_data[neighbor_id]['senders'].append(client_id)
                    receivers_data[neighbor_id]['params_list'].append(params_map[client_id])
                else:
                    gossip_failure += 1
        
        # 对每个接收者，一次性调用 receive_gossip_push，传入所有发送者的模型
        refs = []
        for receiver_id, data in receivers_data.items():
            refs.append(
                self.clients[receiver_id].receive_gossip_push.remote(
                    data['params_list'],  # 所有发送者的模型列表
                    data['senders']      # 所有发送者ID列表
                )
            )
        
        # 返回统计信息和refs以便调用者等待完成
        return {
            "gossip_push_success_count": gossip_success,
            "gossip_push_failure_count": gossip_failure,
            "pending_refs": refs  # 现在返回refs列表
        }

    def _compute_label_distributions(self) -> list[torch.Tensor]:
        """Compute normalized label distribution vectors from client_label_stats."""
        dists = []
        num_classes = self.cfg.num_classes
        for stats in self.client_label_stats:
            total = sum(stats.values())
            dist = torch.zeros(num_classes)
            if total > 0:
                for cls_id, count in stats.items():
                    dist[cls_id] = count / total
            dists.append(dist)
        return dists

    def _distribute_label_dists(self, label_dists: list[torch.Tensor]):
        """Send label distributions to all client actors."""
        refs = [self.clients[cid].set_label_distribution.remote(label_dists[cid])
                for cid in range(self.cfg.num_clients)]
        ray.get(refs)

    def push_dtd_async(self, round_id: int, selected_client_ids: list,
                       label_dists: list[torch.Tensor]) -> dict:
        """
        DTD: Distribution-aware Decentralized Training — push-based protocol.

        Same communication pattern as push_gossip_async, but receivers use
        JSD-weighted averaging (receive_dtd_push) instead of equal-weight averaging.

        Args:
            round_id: Current round
            selected_client_ids: Active client IDs
            label_dists: Label distribution tensors for ALL clients

        Returns:
            dict with success/failure counts and pending refs
        """
        if self.cfg.neighbor_strategy == "random_per_round":
            self.neighbors = build_neighbors(
                self.cfg.num_clients,
                self.cfg.topology,
                self.cfg.neighbor_strategy,
                self.cfg.neighbor_k,
            )

        selected_set = set(selected_client_ids)

        # Get all selected client models
        all_models_data = ray.get([self.clients[cid].upload_model.remote()
                                    for cid in selected_client_ids])
        params_map = {item["client_id"]: item["params"] for item in all_models_data}

        dtd_success = 0
        dtd_failure = 0
        receivers_data = {}

        for client_id in selected_client_ids:
            neighbor_ids = [nid for nid in self.neighbors[client_id]
                           if nid in selected_set]
            if not neighbor_ids:
                continue

            for neighbor_id in neighbor_ids:
                payload_bytes = estimate_state_dict_bytes(params_map[client_id])
                delivered = self._record_comm(
                    round_id,
                    f"client_{client_id}",
                    f"client_{neighbor_id}",
                    payload_bytes,
                    "gossip_push",  # reuse same comm kind for fair comparison
                )
                if delivered:
                    dtd_success += 1
                    if neighbor_id not in receivers_data:
                        receivers_data[neighbor_id] = {
                            'senders': [],
                            'params_list': [],
                            'label_dists': []
                        }
                    receivers_data[neighbor_id]['senders'].append(client_id)
                    receivers_data[neighbor_id]['params_list'].append(params_map[client_id])
                    receivers_data[neighbor_id]['label_dists'].append(label_dists[client_id])
                else:
                    dtd_failure += 1

        refs = []
        for receiver_id, data in receivers_data.items():
            refs.append(
                self.clients[receiver_id].receive_dtd_push.remote(
                    data['params_list'],
                    data['senders'],
                    data['label_dists']
                )
            )

        return {
            "gossip_push_success_count": dtd_success,
            "gossip_push_failure_count": dtd_failure,
            "pending_refs": refs,
        }

    def get_client_models(self, client_ids):
        refs = [self.clients[client_id].upload_model.remote() for client_id in client_ids]
        return ray.get(refs)

    def evaluate_global(self, params=None):
        """Evaluate global model.
        
        Args:
            params: Optional model parameters to evaluate. If None, aggregate models
                    from all clients and evaluate the aggregated model.
        """
        if not self.clients:
            return 0.0, 0.0
        
        # If params is provided, use it directly
        if params is not None:
            eval_params = params
        else:
            # Get models from all clients and aggregate
            model_refs = [client.upload_model.remote() for client in self.clients]
            models = ray.get(model_refs)
            
            if not models:
                return 0.0, 0.0
            
            # Aggregate models (equal-weight averaging)
            aggregated_params = None
            for model_data in models:
                client_params = model_data["params"]
                if aggregated_params is None:
                    aggregated_params = {k: v.clone() for k, v in client_params.items()}
                else:
                    for k in aggregated_params:
                        aggregated_params[k] += client_params[k]
            
            # Average the aggregated parameters
            for k in aggregated_params:
                aggregated_params[k] = aggregated_params[k].float() / len(models)
            
            eval_params = aggregated_params
        
        # Use first client for evaluation
        ray.get(self.clients[0].receive_model.remote(eval_params))
        eval_loss, eval_acc = ray.get(self.clients[0].evaluate.remote(self.val_x, self.val_y))
        return eval_loss, eval_acc

    def save_final_model(self, params) -> str:
        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.run_name}_final_model.pt"
        torch.save(params, path)
        return str(path)

    def _generate_topology_graph(self) -> None:
        """Generate and save topology graph."""
        if not VISUALIZATION_AVAILABLE:
            print("Warning: networkx or matplotlib not installed, skipping topology visualization")
            return

        try:
            G = nx.Graph()
            
            has_server = self.cfg.topology == "star"
            
            if has_server:
                G.add_node("server", node_type="server")
            
            for client_id in range(self.cfg.num_clients):
                G.add_node(f"C{client_id}", node_type="client")
            
            if self.cfg.topology == "star":
                for client_id in range(self.cfg.num_clients):
                    G.add_edge("server", f"C{client_id}")
            elif self.cfg.topology == "ring":
                for client_id in range(self.cfg.num_clients):
                    next_client = (client_id + 1) % self.cfg.num_clients
                    G.add_edge(f"C{client_id}", f"C{next_client}")
            elif self.cfg.topology == "mesh":
                for client_id, neighbor_ids in self.neighbors.items():
                    for neighbor_id in neighbor_ids:
                        G.add_edge(f"C{client_id}", f"C{neighbor_id}")
            
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(12, 10))
            
            if self.cfg.topology == "star":
                pos = {"server": (0, 0)}
                for i in range(self.cfg.num_clients):
                    angle = 2 * math.pi * i / self.cfg.num_clients
                    x = 1.2 * math.cos(angle)
                    y = 1.2 * math.sin(angle)
                    pos[f"C{i}"] = (x, y)
            elif self.cfg.topology == "ring":
                pos = {}
                for i in range(self.cfg.num_clients):
                    angle = 2 * math.pi * i / self.cfg.num_clients
                    x = 1.2 * math.cos(angle)
                    y = 1.2 * math.sin(angle)
                    pos[f"C{i}"] = (x, y)
            elif self.cfg.topology == "mesh":
                pos = nx.kamada_kawai_layout(G)
            else:
                pos = nx.spring_layout(G, k=2, iterations=50)
            
            client_nodes = [f"C{i}" for i in range(self.cfg.num_clients)]
            
            client_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(client_nodes)))
            
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=client_nodes,
                node_color=client_colors,
                node_size=400,
                alpha=0.9,
                ax=ax
            )
            
            if has_server:
                nx.draw_networkx_nodes(
                    G, pos,
                    nodelist=["server"],
                    node_color="#FF6B6B",
                    node_size=800,
                    node_shape="s",
                    alpha=0.9,
                    ax=ax
                )
            
            if self.cfg.topology == "mesh":
                nx.draw_networkx_edges(
                    G, pos,
                    width=0.8,
                    alpha=0.4,
                    edge_color="#888888",
                    style="solid",
                    ax=ax
                )
            elif has_server:
                nx.draw_networkx_edges(
                    G, pos,
                    width=1.2,
                    alpha=0.6,
                    edge_color="#4ECDC4",
                    style="solid",
                    ax=ax
                )
            else:
                nx.draw_networkx_edges(
                    G, pos,
                    width=1.5,
                    alpha=0.7,
                    edge_color="#95E1D3",
                    style="solid",
                    ax=ax
                )
            
            labels = {f"C{i}": f"C{i}" for i in range(self.cfg.num_clients)}
            if has_server:
                labels["server"] = "Server"
            
            nx.draw_networkx_labels(
                G, pos,
                labels=labels,
                font_size=9,
                font_weight="bold",
                font_color="white",
                ax=ax
            )
            
            if has_server:
                legend_elements = [
                    plt.scatter([], [], c="#FF6B6B", s=200, marker="s", label="Server (Central Node)", edgecolors="white", linewidths=1.5),
                    plt.scatter([], [], c=plt.cm.Blues(0.7), s=120, label=f"Clients (n={self.cfg.num_clients})", edgecolors="white", linewidths=1.5)
                ]
            else:
                legend_elements = [
                    plt.scatter([], [], c=plt.cm.Blues(0.7), s=150, label=f"Clients (n={self.cfg.num_clients})", edgecolors="white", linewidths=1.5)
                ]
            
            ax.legend(handles=legend_elements, loc="upper right", fontsize=10, framealpha=0.9)
            
            topology_name = {
                "star": "Star Topology",
                "ring": "Ring Topology", 
                "mesh": "Mesh Topology (Random Per-Round)"
            }.get(self.cfg.topology, f"{self.cfg.topology.capitalize()} Topology")
            
            ax.set_title(
                f"{topology_name}\n"
                f"Clients: {self.cfg.num_clients} | "
                f"Neighbors: k={self.cfg.neighbor_k} | "
                f"Gossip Mode: Push (Async)",
                fontsize=14,
                fontweight="bold",
                pad=20
            )
            
            ax.axis("off")
            ax.set_aspect("equal")
            ax.autoscale()
            
            out_dir = Path(self.cfg.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            graph_path = out_dir / f"{self.run_name}_topology.png"
            plt.tight_layout()
            plt.savefig(graph_path, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
            plt.close()
            
            print(f"Topology graph saved to: {graph_path}")

        except Exception as e:
            print(f"Error generating topology graph: {e}")
            import traceback
            traceback.print_exc()
