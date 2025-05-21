import numpy as np
from typing import Tuple, Optional
import torch

class AnomalyInjector:
    """异常注入器，实现时间异常和空间异常的注入方法"""
    
    def __init__(
        self,
        n_nodes: int,
        n_timesteps: int,
        time_anomaly_ratio: float = 0.1,
        space_anomaly_ratio: float = 0.1,
        time_threshold: float = 0.7,
        k_neighbors: Optional[int] = None,
        seed: Optional[int] = None
    ):
        """
        初始化异常注入器
        
        参数:
            n_nodes: 节点数量
            n_timesteps: 时间步数量
            time_anomaly_ratio: 时间异常比例
            space_anomaly_ratio: 空间异常比例
            time_threshold: 时间异常阈值μ
            k_neighbors: 空间异常采样邻居数，默认为节点数的10%
            seed: 随机种子
        """
        self.n_nodes = n_nodes
        self.n_timesteps = n_timesteps
        self.time_anomaly_ratio = time_anomaly_ratio
        self.space_anomaly_ratio = space_anomaly_ratio
        self.time_threshold = time_threshold
        self.k_neighbors = k_neighbors or int(n_nodes * 0.1)
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
    
    def inject_time_anomaly(
        self,
        data: np.ndarray,
        return_mask: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        注入时间异常
        
        参数:
            data: 输入数据，形状为 [n_nodes, n_timesteps, n_features]
            return_mask: 是否返回异常掩码
            
        返回:
            注入异常后的数据和可选的异常掩码
        """
        # 复制数据以避免修改原始数据
        injected_data = data.copy()
        
        # 计算每个节点的历史最大值
        max_flows = np.max(data, axis=1, keepdims=True)  # [n_nodes, 1, n_features]
        
        # 随机选择要注入异常的节点
        n_anomaly_nodes = int(self.n_nodes * self.time_anomaly_ratio)
        anomaly_nodes = np.random.choice(self.n_nodes, n_anomaly_nodes, replace=False)
        
        # 创建异常掩码
        anomaly_mask = np.zeros_like(data, dtype=bool)
        
        # 对选中的节点注入时间异常
        for node in anomaly_nodes:
            # 应用阈值限制
            threshold = self.time_threshold * max_flows[node]
            injected_data[node] = np.minimum(data[node], threshold)
            anomaly_mask[node] = injected_data[node] < data[node]
        
        if return_mask:
            return injected_data, anomaly_mask
        return injected_data
    
    def inject_space_anomaly(self, data, return_mask=False):
        """
        注入空间异常
        Args:
            data: 形状为 (n_nodes, n_patches, patch_len, n_features) 的数据
            return_mask: 是否返回异常掩码
        """
        n_nodes, n_patches, patch_len, n_features = data.shape
        injected_data = data.copy()
        
        # 计算每个patch的平均值，用于空间异常检测
        patch_means = np.mean(data, axis=(1, 2))  # [n_nodes, n_features]
        
        # 选择要注入异常的节点
        n_anomaly_nodes = int(n_nodes * self.space_anomaly_ratio)
        anomaly_nodes = np.random.choice(n_nodes, n_anomaly_nodes, replace=False)
        
        # 为每个异常节点找到最不同的邻居
        for node in anomaly_nodes:
            # 计算当前节点与所有其他节点的差异
            node_mean = patch_means[node]  # [n_features]
            other_means = patch_means[np.arange(n_nodes) != node]  # [n_nodes-1, n_features]
            
            # 计算L2距离
            distances = np.linalg.norm(other_means - node_mean, axis=1)  # [n_nodes-1]
            
            # 选择k个最不同的邻居
            k = min(self.k_neighbors, len(distances))
            top_k_idx = np.argpartition(distances, -k)[-k:]
            max_diff_idx = top_k_idx[np.argmax(distances[top_k_idx])]
            
            # 获取最不同邻居的索引（需要调整，因为other_means排除了当前节点）
            other_nodes = np.arange(n_nodes) != node
            max_diff_neighbor = np.where(other_nodes)[0][max_diff_idx]
            
            # 用邻居的数据替换当前节点的数据
            injected_data[node] = data[max_diff_neighbor]
        
        if return_mask:
            # 创建异常掩码
            anomaly_mask = np.zeros((n_nodes, n_patches), dtype=bool)
            anomaly_mask[anomaly_nodes] = True
            return injected_data, anomaly_mask
        
        return injected_data
    
    def inject_anomalies(
        self,
        data: np.ndarray,
        inject_time: bool = True,
        inject_space: bool = True,
        return_mask: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        同时注入时间异常和空间异常
        
        参数:
            data: 输入数据，形状为 [n_nodes, n_patches, patch_len, n_features]
            inject_time: 是否注入时间异常
            inject_space: 是否注入空间异常
            return_mask: 是否返回异常掩码
            
        返回:
            注入异常后的数据和可选的异常掩码
        """
        injected_data = data.copy()
        time_mask = None
        space_mask = None
        
        if inject_time:
            injected_data, time_mask = self.inject_time_anomaly(injected_data, return_mask=True)
        
        if inject_space:
            injected_data, space_mask = self.inject_space_anomaly(injected_data, return_mask=True)
            # 将空间异常掩码扩展到与时间异常掩码相同的维度
            if space_mask is not None and time_mask is not None:
                space_mask = np.expand_dims(space_mask, axis=(2, 3))  # 扩展为 [n_nodes, n_patches, 1, 1]
                space_mask = np.tile(space_mask, (1, 1, time_mask.shape[2], time_mask.shape[3]))  # 扩展到与time_mask相同的维度
        
        if return_mask:
            # 合并两种异常的掩码
            if time_mask is not None and space_mask is not None:
                anomaly_mask = np.logical_or(time_mask, space_mask)
            else:
                anomaly_mask = time_mask if time_mask is not None else space_mask
            return injected_data, anomaly_mask
        
        return injected_data 