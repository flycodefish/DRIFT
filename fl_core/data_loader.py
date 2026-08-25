"""
Shared data loading utilities for CIFAR-10, CIFAR-100, NEU-DET, and HAR datasets.
Includes model definitions (CNNModel, HARMLP) used across DE-ASFL and baselines.
"""
import numpy as np
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ── Model definitions ───────────────────────────────────────

class SimpleModel(nn.Module):
    """Simple neural network for smoke testing."""
    def __init__(self, input_dim=10, hidden_dim=64, num_classes=10):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNNModel(nn.Module):
    """CNN for CIFAR-10/NEU image classification (~0.6M params)."""
    def __init__(self, num_classes=10):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class HARMLP(nn.Module):
    """MLP for HAR human activity recognition (561-dim → 6 classes)."""
    def __init__(self, input_dim=561, num_classes=6):
        super(HARMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x


# ── Internal helpers ────────────────────────────────────────

def _compute_label_distributions(train_data, num_clients, num_classes=10):
    """Compute per-client label distributions (probability vectors)."""
    distributions = []
    for client_id in range(num_clients):
        labels = train_data[client_id]['labels']
        counts = torch.bincount(labels, minlength=num_classes).float()
        probs = counts / counts.sum()
        distributions.append(probs.numpy())
    return distributions


def _build_iid_shards_simple(train_x, train_y, val_x, val_y, num_clients):
    """IID partition: evenly split data across clients."""
    indices = torch.randperm(train_x.shape[0])
    shard_size = train_x.shape[0] // num_clients

    train_data = []
    val_data = []
    indices_list = []

    for client_id in range(num_clients):
        start = client_id * shard_size
        end = train_x.shape[0] if client_id == num_clients - 1 else start + shard_size
        shard_idx = indices[start:end]

        train_data.append({
            'images': train_x[shard_idx],
            'labels': train_y[shard_idx]
        })

        val_start = client_id * (len(val_x) // num_clients)
        val_end = (client_id + 1) * (len(val_x) // num_clients) if client_id < num_clients - 1 else len(val_x)
        val_data.append({
            'images': val_x[val_start:val_end],
            'labels': val_y[val_start:val_end]
        })

        indices_list.append(shard_idx.tolist())

    return train_data, val_data, indices_list


def _build_dirichlet_shards_simple(train_x, train_y, val_x, val_y, num_clients, alpha, num_classes=10):
    """Dirichlet non-IID partition."""
    alpha = max(alpha, 1e-6)
    client_indices = [[] for _ in range(num_clients)]

    for class_id in range(num_classes):
        class_mask = train_y == class_id
        class_indices = torch.nonzero(class_mask, as_tuple=False).flatten().tolist()
        if not class_indices:
            continue
        random.shuffle(class_indices)

        concentration = torch.full((num_clients,), alpha)
        proportions = torch.distributions.Dirichlet(concentration).sample()
        raw_counts = proportions * len(class_indices)
        counts = torch.floor(raw_counts).to(torch.int64)
        remain = len(class_indices) - int(counts.sum().item())

        if remain > 0:
            frac = raw_counts - counts.float()
            topk = torch.topk(frac, k=remain).indices.tolist()
            for idx in topk:
                counts[idx] += 1

        current_pos = 0
        for client_id in range(num_clients):
            client_indices[client_id].extend(
                class_indices[current_pos:current_pos + counts[client_id].item()]
            )
            current_pos += counts[client_id].item()

    # Ensure every client has at least batch_size samples
    min_samples = 64
    for client_id in range(num_clients):
        deficit = min_samples - len(client_indices[client_id])
        if deficit > 0:
            donors = sorted(
                [(idx, len(client_indices[idx])) for idx in range(num_clients) if idx != client_id],
                key=lambda x: -x[1]
            )
            for donor_id, _ in donors:
                if deficit <= 0:
                    break
                take = min(deficit, len(client_indices[donor_id]) - min_samples)
                if take > 0:
                    taken = client_indices[donor_id][-take:]
                    client_indices[client_id].extend(taken)
                    client_indices[donor_id] = client_indices[donor_id][:-take]
                    deficit -= take

    train_data = []
    indices_list = []

    for client_id in range(num_clients):
        shard_idx = torch.tensor(client_indices[client_id], dtype=torch.long)
        train_data.append({
            'images': train_x[shard_idx],
            'labels': train_y[shard_idx]
        })
        indices_list.append(client_indices[client_id])

    val_data = []
    val_shard_size = len(val_x) // num_clients
    for client_id in range(num_clients):
        val_start = client_id * val_shard_size
        val_end = len(val_x) if client_id == num_clients - 1 else (client_id + 1) * val_shard_size
        val_data.append({
            'images': val_x[val_start:val_end],
            'labels': val_y[val_start:val_end]
        })

    return train_data, val_data, indices_list


# ── Public data loading API ─────────────────────────────────

def load_cifar10_data(data_dir='./data', num_clients=30, partition_mode='dirichlet', alpha=0.1,
                     global_test_ratio=0.1):
    """Load CIFAR-10 and partition across clients.

    Returns:
        train_data, val_data, indices, global_test_data, label_distributions
    """
    import torchvision

    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=None)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=None)

    # 官方训练集只用于本地训练；官方测试集只用于验证和全局评估。
    # 不能先合并再随机切分，否则会把测试样本泄漏到本地训练中。
    train_x = torch.tensor(trainset.data).permute(0, 3, 1, 2).float() / 255.0
    train_y = torch.tensor(trainset.targets)
    val_x = torch.tensor(testset.data).permute(0, 3, 1, 2).float() / 255.0
    val_y = torch.tensor(testset.targets)
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std

    global_test_data = {'images': val_x, 'labels': val_y}

    if partition_mode == 'iid':
        train_data, val_data, indices = _build_iid_shards_simple(train_x, train_y, val_x, val_y, num_clients)
    elif partition_mode == 'dirichlet':
        train_data, val_data, indices = _build_dirichlet_shards_simple(train_x, train_y, val_x, val_y, num_clients, alpha, num_classes=10)
    else:
        raise ValueError(f"Unsupported partition mode: {partition_mode}")

    label_distributions = _compute_label_distributions(train_data, num_clients)

    return train_data, val_data, indices, global_test_data, label_distributions


def load_cifar100_data(data_dir='./data', num_clients=50, partition_mode='dirichlet', alpha=0.1,
                       global_test_ratio=0.1):
    """Load CIFAR-100 and partition across clients.

    Returns:
        train_data, val_data, indices, global_test_data, label_distributions
    """
    import torchvision

    trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=None)
    testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=None)

    all_data = np.concatenate([trainset.data, testset.data], axis=0)
    all_labels = np.concatenate([np.array(trainset.targets), np.array(testset.targets)], axis=0)

    all_data = torch.tensor(all_data).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(1, 3, 1, 1)
    std = torch.tensor([0.2675, 0.2565, 0.2761]).view(1, 3, 1, 1)
    all_data = (all_data - mean) / std
    all_labels = torch.tensor(all_labels)

    split = int(0.8 * len(all_data))
    indices = torch.randperm(len(all_data))
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_x = all_data[train_indices]
    train_y = all_labels[train_indices]
    val_x = all_data[val_indices]
    val_y = all_labels[val_indices]

    global_test_data = None

    if partition_mode == 'iid':
        train_data, val_data, indices = _build_iid_shards_simple(train_x, train_y, val_x, val_y, num_clients)
    elif partition_mode == 'dirichlet':
        train_data, val_data, indices = _build_dirichlet_shards_simple(train_x, train_y, val_x, val_y, num_clients, alpha, num_classes=100)
    else:
        raise ValueError(f"Unsupported partition mode: {partition_mode}")

    label_distributions = _compute_label_distributions(train_data, num_clients, num_classes=100)

    return train_data, val_data, indices, global_test_data, label_distributions


def load_neu_data(data_dir='./data', num_clients=12, partition_mode='dirichlet', alpha=0.1,
                  global_test_ratio=0.1):
    """Load NEU-DET steel surface defect dataset and partition across clients.

    Uses kagglehub for automatic download.  Grayscale images are resized to 32×32
    and replicated to 3 channels to match CNNModel/ResNet18 input.

    Returns:
        train_data, val_data, indices, global_test_data, label_distributions
    """
    import kagglehub
    from PIL import Image

    try:
        dataset_path = kagglehub.dataset_download("kaustubhdikshit/neu-surface-defect-database")
        base = os.path.join(dataset_path, "NEU-DET")
    except Exception:
        base = data_dir

    class_names = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

    images = []
    labels = []

    for class_idx, class_name in enumerate(class_names):
        for split in ['train', 'validation']:
            cls_dir = os.path.join(base, split, 'images', class_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                img_path = os.path.join(cls_dir, fname)
                img = Image.open(img_path).convert('L')
                img = img.resize((32, 32), Image.BILINEAR)
                img_tensor = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
                images.append(img_tensor)
                labels.append(class_idx)

    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    # Replicate grayscale to 3 channels
    images = images.unsqueeze(1).repeat(1, 3, 1, 1)

    mean = images.mean(dim=[0, 2, 3], keepdim=True)
    std = images.std(dim=[0, 2, 3], keepdim=True)
    images = (images - mean) / (std + 1e-7)

    print(f"NEU surface defect dataset: {len(images)} images, {len(class_names)} classes")

    split = int(0.8 * len(images))
    perm = torch.randperm(len(images))
    train_indices = perm[:split]
    val_indices = perm[split:]

    train_x = images[train_indices]
    train_y = labels[train_indices]
    val_x = images[val_indices]
    val_y = labels[val_indices]

    global_test_data = None

    if partition_mode == 'iid':
        train_data, val_data, indices = _build_iid_shards_simple(train_x, train_y, val_x, val_y, num_clients)
    elif partition_mode == 'dirichlet':
        train_data, val_data, indices = _build_dirichlet_shards_simple(train_x, train_y, val_x, val_y, num_clients, alpha, num_classes=6)
    else:
        raise ValueError(f"Unsupported partition mode: {partition_mode}")

    # 6-class distribution helper
    def _compute_distributions(data, n_clients, n_classes=6):
        dists = []
        for cid in range(n_clients):
            labs = data[cid]['labels']
            counts = torch.bincount(labs, minlength=n_classes).float()
            dists.append((counts / counts.sum()).numpy())
        return dists

    label_distributions = _compute_distributions(train_data, num_clients)

    return train_data, val_data, indices, global_test_data, label_distributions


def load_har_data(data_dir='./data', num_clients=12, partition_mode='dirichlet', alpha=0.1,
                  global_test_ratio=0.2):
    """Load UCI HAR human activity recognition dataset and partition across clients.

    Returns:
        train_data, val_data, indices, global_test_data, label_distributions
    """
    har_path = os.path.join(data_dir, 'har', 'UCI HAR Dataset')
    if not os.path.exists(har_path):
        raise FileNotFoundError(f"HAR dataset not found at {har_path}. "
                                f"Download from: https://archive.ics.uci.edu/ml/machine-learning-databases/00240/")

    # Load raw data
    train_x = np.loadtxt(os.path.join(har_path, 'train', 'X_train.txt'))
    train_y = np.loadtxt(os.path.join(har_path, 'train', 'y_train.txt')).astype(np.int64) - 1
    test_x = np.loadtxt(os.path.join(har_path, 'test', 'X_test.txt'))
    test_y = np.loadtxt(os.path.join(har_path, 'test', 'y_test.txt')).astype(np.int64) - 1

    all_x = np.vstack([train_x, test_x])
    all_y = np.concatenate([train_y, test_y])

    from sklearn.model_selection import train_test_split
    train_x, global_test_x, train_y, global_test_y = train_test_split(
        all_x, all_y, test_size=global_test_ratio, random_state=42, stratify=all_y
    )
    train_x, val_x, train_y, val_y = train_test_split(
        train_x, train_y, test_size=0.1, random_state=42, stratify=train_y
    )

    train_x = torch.tensor(train_x, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.long)
    val_x = torch.tensor(val_x, dtype=torch.float32)
    val_y = torch.tensor(val_y, dtype=torch.long)
    global_test_x = torch.tensor(global_test_x, dtype=torch.float32)
    global_test_y = torch.tensor(global_test_y, dtype=torch.long)

    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True)
    std[std < 1e-8] = 1.0
    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std
    global_test_x = (global_test_x - mean) / std

    if partition_mode == 'dirichlet':
        train_data, val_data, indices = _build_dirichlet_shards_simple(
            train_x, train_y, val_x, val_y, num_clients, alpha, num_classes=6
        )
    else:
        train_data, val_data, indices = _build_iid_shards_simple(
            train_x, train_y, val_x, val_y, num_clients
        )

    global_test_data = {'images': global_test_x, 'labels': global_test_y}

    total_samples = len(all_y)
    print(f"HAR dataset: {total_samples} samples, {6} classes → {num_clients} clients (α={alpha})")
    for i in range(num_clients):
        client_labels = train_data[i]['labels']
        unique_labels = len(torch.unique(client_labels))
        print(f"  Client {i}: {len(client_labels)} samples, {unique_labels} classes")

    label_distributions = _compute_label_distributions(train_data, num_clients, num_classes=6)

    return train_data, val_data, indices, global_test_data, label_distributions
