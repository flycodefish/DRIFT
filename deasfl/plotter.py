"""
可视化模块：生成论文级别的训练图表和拓扑可视化

该模块负责：
1. 绘制训练损失和验证损失曲线
2. 绘制验证准确率曲线
3. 生成组合图（损失+准确率）
4. 保存训练指标到CSV文件
5. 生成实验报告
6. 可视化拓扑结构（支持节点大小和颜色根据资源得分变化）

使用的库：
- matplotlib: 基础绘图库
- seaborn: 美化图表样式
- pandas: 数据处理和CSV读写
- networkx: 图结构处理
- numpy: 数值计算

图表风格：
- 使用seaborn的whitegrid风格
- 设置高质量（dpi=300）输出
- 支持论文级别的美观度
"""

import os
import numpy as np
import matplotlib
# 使用非交互式后端，避免显示窗口
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx

class NewPlotter:
    """
    图表生成器类
    
    参数：
        output_dir: 图表保存目录
    """
    
    def __init__(self, output_dir: str):
        """
        初始化图表生成器
        
        参数：
            output_dir: 输出目录路径
        """
        self.output_dir = output_dir
        # 设置seaborn样式（白色网格背景）
        sns.set_style("whitegrid")
        # 获取颜色调色板（10种颜色）
        self.colors = sns.color_palette("Set1", 10)
    
    def plot_training_curves(self, metrics: dict, filename: str = "training_curves.png"):
        """
        绘制训练曲线（损失和准确率分开显示）
        
        参数：
            metrics: 训练指标字典，包含rounds, train_loss, val_loss, val_acc
            filename: 输出文件名
            
        生成的图表：
            左图：训练损失和验证损失曲线
            右图：验证准确率曲线
        """
        rounds = metrics['rounds']
        train_loss = metrics['train_loss']
        val_loss = metrics['val_loss']
        val_acc = metrics['val_acc']
        
        # 创建1行2列的子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左图：损失曲线
        ax1.plot(rounds, train_loss, label='Train Loss', color=self.colors[0], linewidth=2)
        ax1.plot(rounds, val_loss, label='Val Loss', color=self.colors[1], linewidth=2)
        ax1.set_xlabel('Round', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14)
        ax1.legend(fontsize=10)
        
        # 右图：准确率曲线
        ax2.plot(rounds, val_acc, label='Val Accuracy', color=self.colors[2], linewidth=2)
        ax2.set_xlabel('Round', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Validation Accuracy', fontsize=14)
        ax2.set_ylim(0, 1)  # 准确率范围0-1
        ax2.legend(fontsize=10)
        
        # 自动调整布局
        plt.tight_layout()
        # 保存图片（高分辨率）
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        # 关闭图表，释放内存
        plt.close()
    
    def plot_loss_acc(self, metrics: dict, filename: str = "loss_acc_combined.png"):
        """
        绘制组合图（损失和准确率在同一图中）
        
        参数：
            metrics: 训练指标字典
            filename: 输出文件名
            
        特点：
            使用双Y轴，左侧显示损失，右侧显示准确率
            便于对比损失下降和准确率上升的关系
        """
        rounds = metrics['rounds']
        train_loss = metrics['train_loss']
        val_loss = metrics['val_loss']
        val_acc = metrics['val_acc']
        
        # 创建主图
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # 颜色配置
        color1 = self.colors[0]  # 训练损失颜色
        color2 = self.colors[1]  # 验证损失颜色
        color3 = self.colors[2]  # 准确率颜色
        
        # 左侧Y轴：损失
        ax1.set_xlabel('Round', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        l1, = ax1.plot(rounds, train_loss, label='Train Loss', color=color1, linewidth=2)
        l2, = ax1.plot(rounds, val_loss, label='Val Loss', color=color2, linewidth=2)
        
        # 右侧Y轴：准确率
        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy', fontsize=12)
        l3, = ax2.plot(rounds, val_acc, label='Val Accuracy', color=color3, linewidth=2)
        
        # 设置标题
        ax1.set_title('Training Progress', fontsize=14)
        
        # 合并图例
        lines = [l1, l2, l3]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='lower left', fontsize=10)
        
        # 保存图片
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_metrics(self, metrics: dict, filename: str = "metrics.csv"):
        """
        保存训练指标到CSV文件
        
        参数：
            metrics: 训练指标字典
            filename: 输出文件名
            
        CSV内容：
            round: 轮数
            train_loss: 训练损失
            val_loss: 验证损失
            val_acc: 验证准确率
        """
        # 创建DataFrame
        df = pd.DataFrame({
            'round': metrics['rounds'],
            'train_loss': metrics['train_loss'],
            'val_loss': metrics['val_loss'],
            'val_acc': metrics['val_acc']
        })
        # 保存到CSV
        df.to_csv(os.path.join(self.output_dir, filename), index=False)
    
    def generate_report(self, cfg, metrics: dict, runtime: float):
        """
        生成实验报告
        
        参数：
            cfg: 配置对象
            metrics: 训练指标字典
            runtime: 总运行时间（秒）
            
        报告内容：
            1. 实验配置信息
            2. 训练结果（最终指标、最佳指标）
            3. 指标汇总（平均值）
        """
        report = f"""New FL Algorithm Experiment Report
============================================

Experiment Configuration:
-------------------------
Algorithm: {cfg.algorithm}
Aggregation Strategy: {cfg.aggregation_strategy}
Number of Clients: {cfg.num_clients}
Number of Rounds: {cfg.rounds}
Local Epochs: {cfg.local_epochs}
Batch Size: {cfg.batch_size}
Learning Rate: {cfg.learning_rate}
Dataset: {cfg.dataset}
Model: {cfg.model}
Data Partition: {cfg.partition} (alpha={cfg.alpha})

Training Results:
-----------------
Total Runtime: {runtime:.2f} seconds
Final Training Loss: {metrics['train_loss'][-1]:.4f}
Final Validation Loss: {metrics['val_loss'][-1]:.4f}
Final Validation Accuracy: {metrics['val_acc'][-1]:.4f}
Best Validation Accuracy: {max(metrics['val_acc']):.4f}

Metrics Summary:
----------------
Average Training Loss: {sum(metrics['train_loss'])/len(metrics['train_loss']):.4f}
Average Validation Loss: {sum(metrics['val_loss'])/len(metrics['val_loss']):.4f}
Average Validation Accuracy: {sum(metrics['val_acc'])/len(metrics['val_acc']):.4f}

============================================
"""
        # 保存报告到文件
        with open(os.path.join(self.output_dir, "report.txt"), 'w') as f:
            f.write(report)
        
        print("Report generated successfully")
    
    def plot_topology(self, topology: list, resource_scores: list = None, 
                      filename: str = "topology.png", title: str = "Network Topology"):
        """
        绘制网络拓扑图
        
        参数：
            topology: 拓扑结构列表，每个元素是一个节点的邻居列表
            resource_scores: 每个节点的资源得分列表（可选），用于节点大小和颜色
            filename: 输出文件名
            title: 图表标题
            
        特点：
            - 使用 spring_layout 布局，避免节点围成一圈
            - 节点大小根据资源得分变化（得分越高节点越大）
            - 节点颜色根据资源得分变化（得分越高颜色越深）
            - 无边框背景，突出节点和连接
        """
        n_nodes = len(topology)
        
        # 创建无向图
        G = nx.Graph()
        
        # 添加节点
        G.add_nodes_from(range(n_nodes))
        
        # 添加边（只添加一次，避免重复）
        for i in range(n_nodes):
            for j in topology[i]:
                if j > i:  # 避免重复添加边
                    G.add_edge(i, j)
        
        # 使用 spring_layout 布局（力导向布局，避免形成圆圈）
        pos = nx.spring_layout(G, seed=42, k=0.15, iterations=100)
        
        # 计算节点大小（根据资源得分）
        if resource_scores is not None:
            min_score = min(resource_scores)
            max_score = max(resource_scores)
            # 节点大小范围：200-1500
            node_sizes = [200 + (score - min_score) / (max_score - min_score) * 1300 
                         if max_score != min_score else 500 
                         for score in resource_scores]
            # 节点颜色（使用蓝色渐变，得分越高颜色越深）
            node_colors = [self._get_color(score, min_score, max_score) 
                          for score in resource_scores]
        else:
            node_sizes = 500
            node_colors = '#1f78b4'  # 默认蓝色
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, 
                               alpha=0.8, edgecolors='white', linewidths=2, ax=ax)
        
        # 绘制边（浅色，避免遮挡节点）
        nx.draw_networkx_edges(G, pos, edge_color='#d3d3d3', alpha=0.6, ax=ax)
        
        # 绘制节点标签
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', 
                                font_color='white', ax=ax)
        
        # 设置标题
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # 隐藏坐标轴
        ax.axis('off')
        
        # 添加颜色条（如果有资源得分）
        if resource_scores is not None:
            sm = plt.cm.ScalarMappable(cmap='Blues', 
                                       norm=plt.Normalize(vmin=min_score, vmax=max_score))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.02, pad=0.02)
            cbar.set_label('Resource Score', fontsize=12)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Topology visualization saved to: {os.path.join(self.output_dir, filename)}")
    
    def _get_color(self, score: float, min_score: float, max_score: float) -> str:
        """
        根据资源得分获取节点颜色（蓝色渐变）
        
        参数：
            score: 当前节点的资源得分
            min_score: 最小得分
            max_score: 最大得分
            
        返回：
            颜色字符串（十六进制）
        """
        if max_score == min_score:
            return '#6baed6'  # 中等蓝色
        
        # 归一化到 0-1 范围
        normalized = (score - min_score) / (max_score - min_score)
        
        # 蓝色渐变：从浅蓝到深蓝
        # RGB值：浅蓝(107, 174, 214) -> 深蓝(0, 51, 102)
        r = int(107 - normalized * 107)
        g = int(174 - normalized * 123)
        b = int(214 - normalized * 112)
        
        return f'#{r:02x}{g:02x}{b:02x}'
