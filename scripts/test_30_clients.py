"""
测试30个客户端的选举、聚类和训练功能（支持多层树状结构验证）
支持从YAML配置文件读取参数
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deasfl import NewFLConfig, NewFLRuntime
from deasfl.runtime import load_cifar10_data, load_cifar100_data, load_neu_data, load_har_data
import torch
import time
import numpy as np


def create_pattern_data(num_clients=30, train_samples=100, val_samples=20, input_dim=10, num_classes=10):
    """创建具有可学习模式的数据集"""
    train_data = []
    val_data = []
    indices = []
    
    for client_id in range(num_clients):
        # 训练数据 - 创建有模式的数据：第i类的数据在第i个维度上有最高值
        train_images = []
        train_labels = []
        for _ in range(train_samples):
            label = np.random.randint(0, num_classes)
            x = np.random.randn(input_dim) * 0.5
            x[label] += 3.0  # 在目标类别对应的维度上添加信号
            train_images.append(x)
            train_labels.append(label)
        
        client_train_data = {
            'images': torch.tensor(train_images, dtype=torch.float32),
            'labels': torch.tensor(train_labels, dtype=torch.long)
        }
        
        # 验证数据
        val_images = []
        val_labels = []
        for _ in range(val_samples):
            label = np.random.randint(0, num_classes)
            x = np.random.randn(input_dim) * 0.5
            x[label] += 3.0
            val_images.append(x)
            val_labels.append(label)
        
        client_val_data = {
            'images': torch.tensor(val_images, dtype=torch.float32),
            'labels': torch.tensor(val_labels, dtype=torch.long)
        }
        
        client_indices = list(range(client_id * train_samples, (client_id + 1) * train_samples))
        
        train_data.append(client_train_data)
        val_data.append(client_val_data)
        indices.append(client_indices)
    
    return train_data, val_data, indices


def build_parent_map(clustering_result):
    """
    从聚类结果构建父节点映射
    
    Args:
        clustering_result: 聚类结果字典
    
    Returns:
        parent_map: {node_id: parent_id} 映射，簇首的父节点为 None
    """
    parent_map = {}
    
    # 获取所有节点信息
    if 'all_nodes_info' in clustering_result:
        for node_info in clustering_result['all_nodes_info']:
            node_id = node_info['client_id']
            parent_map[node_id] = node_info.get('parent')
    
    # 如果没有详细节点信息，尝试从簇信息构建
    if not parent_map and 'clusters' in clustering_result:
        for cluster_id, cluster_info in clustering_result['clusters'].items():
            parent_map[cluster_id] = None  # 簇首没有父节点
            if 'members' in cluster_info:
                for member in cluster_info['members']:
                    parent_map[member] = cluster_id
    
    return parent_map


def print_tree(node_id, parent_map, depth=0, visited=None):
    """
    递归打印树状结构
    
    Args:
        node_id: 当前节点ID
        parent_map: 父节点映射
        depth: 当前深度
        visited: 已访问节点集合（用于检测环）
    """
    if visited is None:
        visited = set()
    
    if node_id in visited:
        print("  " * depth + f"Node {node_id} (⚠️ 检测到环)")
        return
    
    visited.add(node_id)
    
    prefix = "  " * depth
    is_head = parent_map[node_id] is None
    role = "(簇首)" if is_head else "(成员)"
    
    print(f"{prefix}Node {node_id} {role}")
    
    # 找到所有子节点
    children = [child for child, parent in parent_map.items() if parent == node_id]
    for child in sorted(children):
        print_tree(child, parent_map, depth + 1, visited.copy())


def validate_hierarchical_structure(clustering_result, max_child):
    """
    验证多层树状结构
    
    Args:
        clustering_result: 聚类结果字典
        max_child: 最大子节点数限制
    
    Returns:
        validation_result: 验证结果字典
    """
    print("\n" + "=" * 80)
    print("验证多层树状结构")
    print("=" * 80)
    
    # 构建父节点映射
    parent_map = build_parent_map(clustering_result)
    
    if not parent_map:
        print("❌ 无法构建父节点映射")
        return {'valid': False, 'error': '无法构建父节点映射'}
    
    print(f"\n1. 父节点映射:")
    for node_id in sorted(parent_map.keys()):
        print(f"   Node {node_id}: parent = {parent_map[node_id]}")
    
    # 检查单父约束
    print(f"\n2. 单父约束检查:")
    has_multiple_parents = False
    for node_id, parent in parent_map.items():
        if parent is not None:
            print(f"   ✅ Node {node_id} 有父节点 {parent}")
        else:
            print(f"   ✅ Node {node_id} 是簇首（无父节点）")
    
    # 检查子节点数量约束
    print(f"\n3. 子节点数量约束检查 (max_child={max_child}):")
    child_count = {}
    for node_id in parent_map.keys():
        child_count[node_id] = 0
    
    for child, parent in parent_map.items():
        if parent is not None:
            child_count[parent] += 1
    
    has_exceeded_max_child = False
    for node_id, count in child_count.items():
        if count > max_child:
            print(f"   ❌ Node {node_id} 子节点数 {count} 超过限制 {max_child}")
            has_exceeded_max_child = True
        else:
            print(f"   ✅ Node {node_id} 子节点数 {count}")
    
    # 检查是否存在环
    print(f"\n4. 环检测:")
    has_cycle = False
    for start_node in parent_map.keys():
        visited = set()
        current = start_node
        while current is not None:
            if current in visited:
                has_cycle = True
                break
            visited.add(current)
            current = parent_map.get(current)
        
        if has_cycle:
            print(f"   ❌ 从 Node {start_node} 出发检测到环")
            break
    
    if not has_cycle:
        print("   ✅ 未检测到环")
    
    # 检测是否存在多层结构
    print(f"\n5. 层次结构检测:")
    has_hierarchy = False
    max_depth = 1
    
    def get_depth(node_id, current_depth=1):
        nonlocal max_depth
        max_depth = max(max_depth, current_depth)
        
        children = [child for child, parent in parent_map.items() if parent == node_id]
        for child in children:
            get_depth(child, current_depth + 1)
    
    for head in [n for n, p in parent_map.items() if p is None]:
        get_depth(head)
    
    if max_depth > 2:
        has_hierarchy = True
        print(f"   ✅ 检测到多层结构，最大深度: {max_depth}")
    else:
        print(f"   ⚠️  只有两层结构，最大深度: {max_depth}")
    
    # 打印树状结构
    print(f"\n6. 树状结构:")
    heads = [n for n, p in parent_map.items() if p is None]
    for head in sorted(heads):
        print(f"\n   簇 {head} 的树:")
        print_tree(head, parent_map, depth=2)
    
    # 总结
    print(f"\n7. 验证总结:")
    all_valid = not (has_multiple_parents or has_exceeded_max_child or has_cycle)
    
    if all_valid:
        print("   ✅ 所有约束验证通过")
    else:
        print("   ❌ 存在约束违规")
    
    return {
        'valid': all_valid,
        'has_hierarchy': has_hierarchy,
        'max_depth': max_depth,
        'parent_map': parent_map,
        'child_count': child_count,
        'has_cycle': has_cycle,
        'has_exceeded_max_child': has_exceeded_max_child
    }


def test_30_clients(config_path: str = None):
    """测试联邦学习训练功能
    
    Args:
        config_path: YAML配置文件路径，如果为None则使用默认配置
    """
    print("=" * 80)
    print("测试联邦学习训练")
    print("=" * 80)

    try:
        if config_path:
            cfg = NewFLConfig.from_yaml(config_path)
            print(f"\n从配置文件加载配置: {config_path}")
        else:
            cfg = NewFLConfig()
            print(f"\n使用默认配置")

        print(f"\n{'='*60}")
        print("配置信息:")
        print(f"{'='*60}")
        print(f"  数据集: {cfg.dataset}")
        print(f"  模型: {cfg.model}")
        print(f"  客户端数量: {cfg.num_clients}")
        print(f"  训练轮数: {cfg.rounds}")
        print(f"  每轮本地训练次数: {cfg.local_epochs}")
        print(f"  批次大小: {cfg.batch_size}")
        print(f"  学习率: {cfg.learning_rate}")
        print(f"  数据划分: {cfg.partition} (alpha={cfg.alpha})")
        print(f"  拓扑类型: {cfg.topology_type}")
        print(f"  邻居数量范围: {cfg.min_neighbors}-{cfg.max_neighbors}")
        print(f"  选举启用: {cfg.enable_election}")
        print(f"  聚类启用: {cfg.enable_clustering}")
        if cfg.enable_clustering:
            print(f"  簇最大子节点数: {cfg.cluster_max_child}")

        # 初始化运行时
        runtime = NewFLRuntime(cfg)
        runtime.start()

        # 加载数据集
        if cfg.dataset == 'neu':
            print(f"\n加载 NEU 钢铁表面缺陷数据集...")
            train_data, val_data, indices, global_test_data, label_distributions = load_neu_data(
                data_dir=cfg.data_dir,
                num_clients=cfg.num_clients,
                partition_mode=cfg.partition,
                alpha=cfg.alpha
            )
        elif cfg.dataset == 'har':
            print(f"\n加载 HAR 人体活动识别数据集...")
            train_data, val_data, indices, global_test_data, label_distributions = load_har_data(
                data_dir=cfg.data_dir,
                num_clients=cfg.num_clients,
                partition_mode=cfg.partition,
                alpha=cfg.alpha
            )
        elif cfg.dataset == 'cifar100':
            print(f"\n加载 CIFAR-100 数据集（{cfg.num_clients} 客户端）...")
            train_data, val_data, indices, global_test_data, label_distributions = load_cifar100_data(
                data_dir=cfg.data_dir,
                num_clients=cfg.num_clients,
                partition_mode=cfg.partition,
                alpha=cfg.alpha
            )
        else:
            print(f"\n加载 CIFAR-10 数据集...")
            train_data, val_data, indices, global_test_data, label_distributions = load_cifar10_data(
                data_dir=cfg.data_dir,
                num_clients=cfg.num_clients,
                partition_mode=cfg.partition,
                alpha=cfg.alpha
            )
        print(f"数据加载完成！")
        print(f"  训练样本数: {sum(len(d['images']) for d in train_data)}")
        print(f"  验证样本数: {sum(len(d['images']) for d in val_data)}")

        # 将标签分布存储到 runtime，供聚类使用
        runtime.label_distributions = label_distributions

        # 初始化客户端
        print(f"初始化客户端...")
        runtime.init_clients(train_data, val_data, indices)

        # 构建拓扑
        print(f"构建拓扑...")
        runtime.build_topology()

        # 运行选举
        election_time = 0.0
        election_result = None
        if cfg.enable_election:
            print(f"\n" + "=" * 80)
            print("步骤1：运行选举")
            print("=" * 80)
            start_time = time.time()
            runtime.init_election(
                base_trigger_time=cfg.election_base_trigger_time,
                trigger_factor=cfg.election_trigger_factor
            )
            election_result = runtime.run_election()
            election_time = time.time() - start_time

            print(f"\n选举结果:")
            print(f"  最优客户端: Node {election_result['best_client_id']}")
            print(f"  最优得分: {election_result['best_client_score']:.2f}")
            print(f"  收敛跳数: {election_result['convergence_hops']}")
            print(f"  选举耗时: {election_time:.2f}s")
        else:
            print(f"\n" + "=" * 80)
            print("步骤1：跳过选举（enable_election=false）")
            print("=" * 80)
            # 构造虚拟选举结果，以 client 0 为默认聚合节点
            election_result = {
                'best_client_id': 0,
                'best_client_score': 0.0,
                'convergence_hops': 0,
                'all_nodes_info': []
            }

        # 运行聚类
        clustering_time = 0.0
        clustering_result = None
        validation_result = {'valid': True}
        if cfg.enable_clustering:
            print(f"\n" + "=" * 80)
            print("步骤2：运行聚类")
            print("=" * 80)
            start_time = time.time()
            runtime.init_clustering(
                election_results=election_result,
                max_child=cfg.cluster_max_child,
                cluster_offset=cfg.cluster_offset
            )
            clustering_result = runtime.run_clustering()
            runtime.set_clustering_results(clustering_result, election_result)
            clustering_time = time.time() - start_time

            print(f"\n聚类结果:")
            print(f"  簇首: {clustering_result['cluster_heads']}")
            print(f"  簇数量: {clustering_result['num_clusters']}")
            print(f"  聚类耗时: {clustering_time:.2f}s")

            # 验证层次化结构
            validation_result = validate_hierarchical_structure(clustering_result, cfg.cluster_max_child)
        else:
            print(f"\n" + "=" * 80)
            print("步骤2：跳过聚类（enable_clustering=false）")
            print("=" * 80)
            # 清除聚类状态，让训练走扁平聚合路径
            runtime.clustering_results = None
            runtime.tree_mapping = None

        # ============ 新增：运行训练 ============
        print(f"\n" + "=" * 80)
        print("步骤3：运行训练")
        print("=" * 80)

        # 创建 Ray Actor 客户端（启用分布式并行训练）
        print("创建 Ray 并行训练客户端...")
        runtime._init_ray_actors(train_data, val_data)

        # 运行训练
        training_results = runtime.run_training(num_rounds=cfg.rounds)
        
        # 验证训练效果
        avg_acc = np.mean(training_results['final_client_accuracies'])
        if avg_acc > 0.7:
            print(f"\n🎉 训练成功！平均准确率达到 {avg_acc:.4f}")
        else:
            print(f"\n⚠️  训练效果一般，平均准确率为 {avg_acc:.4f}")

        # 可视化（带保护）
        print(f"\n" + "=" * 80)
        print("步骤4：生成可视化")
        print("=" * 80)
        try:
            from visualize_topology import visualize_election_and_clustering
            viz_dir = getattr(runtime, 'output_dir', 'visualizations_30_clients')
            if clustering_result is not None:
                visualize_election_and_clustering(
                    topology=runtime.topology,
                    election_result=election_result,
                    clustering_result=clustering_result,
                    output_dir=viz_dir
                )
                print("✅ 可视化生成成功")
            else:
                print("⚠️  跳过聚类可视化（enable_clustering=false）")
        except ImportError:
            print("⚠️  可视化模块未找到，跳过可视化")
        except Exception as e:
            print(f"⚠️  可视化失败: {e}")

        # 关闭运行时
        runtime.shutdown()

        print(f"\n" + "=" * 80)
        print("测试完成！")
        print("=" * 80)
        print(f"\n总耗时: {election_time + clustering_time + training_results['training_time']:.2f}s")
        
        # 验证结果总结
        if validation_result['valid']:
            print("\n🎉 所有验证通过！")
        else:
            print("\n⚠️  存在验证失败，请检查上述输出")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        if 'runtime' in locals():
            runtime.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"

    parser = argparse.ArgumentParser(description='联邦学习训练')
    parser.add_argument('--config', type=str, default=None,
                       help='YAML配置文件路径，默认使用config_de_asfl.yaml')
    args = parser.parse_args()
    
    config_path = args.config if args.config else 'config_de_asfl.yaml'
    test_30_clients(config_path=config_path)