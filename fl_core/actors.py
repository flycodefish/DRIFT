from __future__ import annotations

import copy
import os
import time
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"The pynvml package is deprecated\\..*",
    category=FutureWarning,
)

import ray
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as T


class MLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


from .data_loader import CNNModel as CNN
from .data_loader import HARMLP, MergeSingletonBatchSampler


class ResNet18(nn.Module):
    """ResNet-18 model for CIFAR-10 dataset."""
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        import torchvision.models as models
        self.model = models.resnet18(pretrained=False)
        # 修改第一个卷积层，适应CIFAR-10的32x32图像
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # 移除第一个池化层，因为32x32图像不需要
        self.model.maxpool = nn.Identity()
        # 修改最后一层全连接层，适应CIFAR-10的类别数
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def get_model_params(model: nn.Module):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def set_model_params(model: nn.Module, params) -> None:
    model.load_state_dict(params, strict=True)


def estimate_state_dict_bytes(params: dict[str, torch.Tensor]) -> int:
    total = 0
    for tensor in params.values():
        total += tensor.numel() * tensor.element_size()
    return total


def _apply_deterministic_settings(enabled: bool) -> None:
    """Apply the same deterministic backend policy in every Ray worker."""
    if not enabled:
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@ray.remote(num_cpus=0.05, max_concurrency=1, num_gpus=0.066)
class ClientActor:
    """Low-level client actor: training/evaluation/model upload APIs."""

    def __init__(
        self,
        client_id: int,
        features: torch.Tensor,
        labels: torch.Tensor,
        input_dim: int,
        num_classes: int,
        batch_size: int,
        learning_rate: float,
        local_epochs: int,
        step_delay_ms: float,
        dataset: str = 'synthetic',
        model: str = 'cnn',
        seed: int = 42,
        deterministic_training: bool = True,
        weight_decay: float = 1e-4,
        merge_singleton_batch: bool = False,
    ) -> None:
        _apply_deterministic_settings(deterministic_training)
        self.client_id = client_id
        self.seed = seed
        
        # 根据数据集类型和模型选择参数选择模型
        if dataset in ('cifar10', 'neu', 'cifar100'):
            if model == 'resnet18':
                self.model = ResNet18(num_classes)
            else:
                self.model = CNN(num_classes)
        elif dataset == 'har':
            self.model = HARMLP(input_dim, num_classes)
        else:
            self.model = MLP(input_dim, num_classes)
        
        # 模型先保持在CPU，延迟到首次 local_train 再迁上GPU
        # 避免50个Actor同时初始化CUDA导致cuBLAS handle耗尽
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model_on_gpu = False
        # 数据保留在CPU，训练时按batch搬上GPU，避免显存OOM
        self.dataset = TensorDataset(features, labels)
        loader_generator = torch.Generator().manual_seed(seed)
        if merge_singleton_batch:
            self.loader = DataLoader(
                self.dataset,
                batch_sampler=MergeSingletonBatchSampler(
                    self.dataset, batch_size, generator=loader_generator
                ),
            )
        else:
            self.loader = DataLoader(
                self.dataset, batch_size=batch_size, shuffle=True,
                generator=loader_generator,
            )
        self.learning_rate = learning_rate
        self.current_lr = learning_rate  # 保存当前学习率，用于动态调整
        self.local_epochs = local_epochs
        self.step_delay_ms = step_delay_ms
        self.weight_decay = weight_decay

        # FedProx: proximal term
        self.proximal_mu = 0.0
        self.global_params = None

        # 数据增强（仅用于图像数据，HAR等向量数据跳过）
        if dataset in ('cifar10', 'neu', 'cifar100'):
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

    def _offload_from_cuda(self):
        """Return the persistent model to CPU after one bounded GPU task."""
        if self._model_on_gpu:
            self.model = self.model.cpu()
            self._model_on_gpu = False
            torch.cuda.empty_cache()

    def _warmup_cuda(self):
        """预热CUDA：移动模型到GPU并执行dummy前向/反向传播以初始化cuBLAS句柄"""
        self._ensure_cuda()
        dummy_input = torch.randn(1, 3, 32, 32, device=self.device)
        self.model.train()
        output = self.model(dummy_input)
        loss = output.sum()
        loss.backward()
        self.model.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self.client_id

    def set_learning_rate(self, new_lr: float):
        """设置学习率（用于动态调整）"""
        self.current_lr = new_lr

    def set_proximal_mu(self, mu: float):
        """设置FedProx proximal term强度"""
        self.proximal_mu = mu

    def set_global_params(self, params):
        """设置全局模型参数（用于FedProx proximal term）"""
        self._ensure_cuda()
        device = next(self.model.parameters()).device
        self.global_params = {k: v.to(device) for k, v in params.items()}

    def local_train(self, round_id: int):
        self._ensure_cuda()
        torch.manual_seed(self.seed + round_id)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed + round_id)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.current_lr,
            weight_decay=self.weight_decay,
        )

        self.model.train()
        start_time = time.perf_counter()
        running_loss = 0.0
        trained_samples = 0
        for _ in range(self.local_epochs):
            for xb, yb in self.loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                if self.train_transform is not None:
                    xb = self.train_transform(xb)  # 数据增强（仅图像）
                output = self.model(xb)
                loss = criterion(output, yb)
                # FedProx proximal term
                if self.proximal_mu > 0 and self.global_params is not None:
                    prox = 0.0
                    for name, param in self.model.named_parameters():
                        prox += ((param - self.global_params[name]) ** 2).sum()
                    loss = loss + (self.proximal_mu / 2.0) * prox
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.size(0)
                trained_samples += xb.size(0)
                
                # [新增] 立即销毁计算图引用，释放显存
                del output, loss
                
                if self.step_delay_ms > 0:
                    time.sleep(self.step_delay_ms / 1000.0)

        # 训练结束后，显式清理梯度缓冲区
        optimizer.zero_grad(set_to_none=True)
        del optimizer
        self._offload_from_cuda()
        
        # 返回前清理缓存
        avg_loss = running_loss / max(1, trained_samples)
        return {
            "client_id": self.client_id,
            "train_loss": avg_loss,
            "num_examples": len(self.dataset),
            "train_time_sec": time.perf_counter() - start_time,
        }

    def evaluate(self, features: torch.Tensor, labels: torch.Tensor):
        self._ensure_cuda()
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        # 检查模型是否在GPU上，如果是，将输入数据也移到GPU上
        device = next(self.model.parameters()).device
        features = features.to(device)
        labels = labels.to(device)
        
        loader = DataLoader(TensorDataset(features, labels), batch_size=256, shuffle=False)

        loss_sum = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in loader:
                logits = self.model(xb)
                loss_sum += criterion(logits, yb).item() * yb.size(0)
                correct += (logits.argmax(dim=1) == yb).sum().item()
                total += yb.numel()
        result = (loss_sum / max(1, total), correct / max(1, total))
        self._offload_from_cuda()
        return result

    def upload_model(self):
        params = get_model_params(self.model)
        return {
            "client_id": self.client_id,
            "params": params,
            "num_examples": len(self.dataset),
        }

    def receive_model(self, params):
        set_model_params(self.model, params)
        return self.client_id
    
    def clear_gpu_cache(self):
        """清理 GPU 显存缓存，释放不再使用的张量占用的显存"""
        if torch.cuda.is_available() and self._model_on_gpu:
            torch.cuda.empty_cache()
            device = next(self.model.parameters()).device
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
        return self.client_id
    
    def gossip_exchange(self, neighbor_params: list[dict[str, torch.Tensor]]):
        current = get_model_params(self.model)
        merged = copy.deepcopy(current)
        all_params = [current] + neighbor_params
        for key in merged:
            merged[key].zero_()
            for params in all_params:
                merged[key] += params[key]
            merged[key] /= len(all_params)
        set_model_params(self.model, merged)
        return self.client_id
    
    def receive_gossip_push(self, sender_params_list: list[dict[str, torch.Tensor]], sender_ids: list[int]):
        """
        Receive models from neighbors (Push method) and average with local model.
        Uses equal-weight averaging: new_model = (local_model + sum(received_models)) / (1 + num_received)

        Args:
            sender_params_list: List of model parameters from all senders
            sender_ids: List of sender client IDs
        """
        current = get_model_params(self.model)
        merged = copy.deepcopy(current)

        # Add local model
        all_params = [current]

        # Add received models from neighbors
        if sender_params_list:
            all_params.extend(sender_params_list)

        # Equal-weight averaging
        for key in merged:
            # Convert to float to avoid type mismatch
            merged[key] = merged[key].float()
            merged[key].zero_()
            for params in all_params:
                merged[key] += params[key].float()
            merged[key] /= len(all_params)

        set_model_params(self.model, merged)
        return self.client_id

    def receive_weighted_push(
        self,
        sender_params_list: list[dict[str, torch.Tensor]],
        sender_ids: list[int],
        weights: list[float],
    ):
        """Apply one synchronous Metropolis-style D-PSGD mixing step."""
        current = get_model_params(self.model)
        all_params = [current] + list(sender_params_list)
        all_weights = [float(weights[0])] + [float(value) for value in weights[1:]]
        if len(all_params) != len(all_weights) or not all_params:
            raise ValueError("weighted push parameters and weights must align")
        total = sum(all_weights)
        if total <= 0:
            raise ValueError("weighted push must have positive total weight")
        merged = copy.deepcopy(current)
        for key in merged:
            value = torch.zeros_like(merged[key], dtype=torch.float32)
            for weight, params in zip(all_weights, all_params):
                value += params[key].float() * (weight / total)
            merged[key] = value.to(merged[key].dtype)
        set_model_params(self.model, merged)
        return self.client_id

    def set_jsd_temperature(self, temperature: float):
        """Set JSD temperature for DTD mixing weights."""
        self.jsd_temperature = temperature
        return self.client_id

    def set_label_distribution(self, label_dist: torch.Tensor):
        """Set the label distribution for this client (used by DTD)."""
        self.label_distribution = label_dist if label_dist is not None else None
        self._label_dist_on_gpu = False
        return self.client_id

    def _ensure_label_dist_on_gpu(self):
        if not self._label_dist_on_gpu and self.label_distribution is not None:
            self.label_distribution = self.label_distribution.to(self.device)
            self._label_dist_on_gpu = True

    def receive_dtd_push(self, sender_params_list: list[dict[str, torch.Tensor]],
                         sender_ids: list[int],
                         sender_label_dists: list[torch.Tensor]):
        """
        DTD: Distribution-aware weighted averaging.

        Weight for sender j: w_j = exp(-JSD(P_self, P_j) / temperature)
        Self weight is always 1.0.

        Args:
            sender_params_list: List of model state_dicts from senders
            sender_ids: List of sender client IDs
            sender_label_dists: List of sender label distribution tensors
        """
        current = get_model_params(self.model)
        merged = copy.deepcopy(current)
        self._ensure_label_dist_on_gpu()
        self_dist = self.label_distribution

        temperature = getattr(self, 'jsd_temperature', 1.0)
        eps = 1e-12

        all_params = [current]
        weights = [1.0]  # self always weighted 1.0

        for i in range(len(sender_params_list)):
            sender_dist = sender_label_dists[i].to(self.device)
            # JSD(P, Q) = 0.5*KL(P||M) + 0.5*KL(Q||M),  M = (P+Q)/2
            M = 0.5 * (self_dist + sender_dist)
            kl_self = (self_dist * ((self_dist + eps) / (M + eps)).log()).sum()
            kl_sender = (sender_dist * ((sender_dist + eps) / (M + eps)).log()).sum()
            jsd = 0.5 * kl_self + 0.5 * kl_sender
            weight = float(torch.exp(-jsd / temperature))
            weights.append(weight)
            all_params.append(sender_params_list[i])

        total_weight = sum(weights)
        for key in merged:
            merged[key] = merged[key].float()
            merged[key].zero_()
            for w, params in zip(weights, all_params):
                merged[key] += w * params[key].float()
            merged[key] /= total_weight

        set_model_params(self.model, merged)
        return self.client_id
