from __future__ import annotations

import random


def build_neighbors(
    num_clients: int,
    topology: str,
    neighbor_strategy: str = "fixed",
    neighbor_k: int = 2,
) -> dict[int, list[int]]:
    """Return neighbor ids per client for the given topology."""
    if topology == "star":
        # In star mode, clients communicate via the server only.
        return {client_id: [] for client_id in range(num_clients)}

    if topology == "ring":
        neighbors: dict[int, list[int]] = {}
        for client_id in range(num_clients):
            left = (client_id - 1) % num_clients
            right = (client_id + 1) % num_clients
            neighbors[client_id] = [left, right]
        return neighbors

    if topology == "mesh":
        neighbors: dict[int, list[int]] = {}
        all_ids = list(range(num_clients))

        if neighbor_strategy == "fixed":
            for client_id in range(num_clients):
                # 为每个客户端选择固定的k个邻居
                candidates = [other for other in all_ids if other != client_id]
                k = max(1, min(neighbor_k, len(candidates)))
                # 选择最近的k个邻居（循环排列）
                neighbors[client_id] = []
                for i in range(1, k+1):
                    neighbor = (client_id + i) % num_clients
                    neighbors[client_id].append(neighbor)
                for i in range(1, k+1):
                    neighbor = (client_id - i) % num_clients
                    if neighbor not in neighbors[client_id]:
                        neighbors[client_id].append(neighbor)
                # 确保邻居数量不超过k
                neighbors[client_id] = neighbors[client_id][:k]
            return neighbors

        if neighbor_strategy == "random_per_round":
            for client_id in range(num_clients):
                candidates = [other for other in all_ids if other != client_id]
                k = max(1, min(neighbor_k, len(candidates)))
                neighbors[client_id] = random.sample(candidates, k=k)
            return neighbors

        raise ValueError(f"Unsupported neighbor strategy for mesh: {neighbor_strategy}")

    raise ValueError(f"Unsupported topology: {topology}")
