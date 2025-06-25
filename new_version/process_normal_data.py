import torch
import numpy as np
import torch.nn.functional as F
import math
from .temporal_attention import process_temporal_masked_data
import torch.nn as nn
import os

class StaticDependencyMatrix:
    def __init__(self, n_nodes):
        self.n_nodes = n_nodes
        # 初始化三种静态依赖矩阵
        self.distance_matrix = torch.zeros(n_nodes, n_nodes)
        self.correlation_matrix = torch.zeros(n_nodes, n_nodes)
        self.adjacency_matrix = torch.zeros(n_nodes, n_nodes)
        
    def compute_distance_matrix(self, data):
        """计算欧氏距离矩阵"""
        self.distance_matrix = torch.cdist(data, data)
        return self.distance_matrix
    
    def compute_correlation_matrix(self, data):
        """计算相关系数矩阵"""
        # 计算相关系数
        data_centered = data - data.mean(dim=0, keepdim=True)
        std = data.std(dim=0, keepdim=True)
        corr = torch.mm(data_centered.t(), data_centered) / (data.shape[0] - 1)
        corr = corr / (std.t() * std)
        self.correlation_matrix = torch.abs(corr)  # 取绝对值
        return self.correlation_matrix
    
    def compute_adjacency_matrix(self, threshold=0.5):
        """基于相关系数计算邻接矩阵"""
        self.adjacency_matrix = (self.correlation_matrix > threshold).float()
        # 将对角线设为0
        self.adjacency_matrix.fill_diagonal_(0)
        return self.adjacency_matrix
    
    def get_static_matrices(self):
        """获取所有静态依赖矩阵"""
        return torch.stack([
            self.distance_matrix,
            self.correlation_matrix,
            self.adjacency_matrix
        ])

class SingleGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dist_mat, n_layers=1, activation=F.relu, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dist_mat = dist_mat
        self.n_layers = n_layers
        self.activation = activation

        d = self.dist_mat.shape[0]
        self.sigma = torch.nn.Linear(1, d)
        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(in_channels, out_channels, bias=bias)] + [
                torch.nn.Linear(out_channels, out_channels, bias=bias)
                for _ in range(n_layers)
            ])
        self.mat_2 = dist_mat.square()
        self.mask = self.generate_mask()

    def generate_mask(self):
        # 距离为0的位置设为1，对角线设为0，与DSCL-master保持一致
        matrix = (self.dist_mat.cpu().detach().numpy() == 0.).astype(int)
        # 对角线设为0
        matrix[range(matrix.shape[0]), range(matrix.shape[0])] = 0
        return torch.Tensor(matrix) == 1

    def forward(self, x):
        # 计算收缩矩阵, x : (N, T, D) 或 (B, N, D)
        sigma = torch.sigmoid(self.sigma.weight * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        exp = torch.exp(-self.mat_2 / (2 * sigma**2))
        prior = exp / (math.sqrt(2 * math.pi) * sigma)
        prior = prior.masked_fill(self.mask.to(exp.device), 0)  # 使用masked_fill而不是乘法
        prior /= prior.sum(0)  # 添加归一化，与DSCL-master保持一致

        for i in range(self.n_layers):
            wx = prior @ x
            x = self.activation(self.linears[i](wx))
        
        return x, prior

class MultipleGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, matrices, n_layers=1, activation=F.relu, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.matrices = matrices
        self.n_layers = n_layers
        self.activation = activation
        self.n_graph = matrices.shape[0]

        d = self.matrices.shape[-1]
        self.sigma = torch.nn.Linear(self.n_graph, d)
        self.alpha = torch.nn.parameter.Parameter(torch.ones(self.n_graph) / self.n_graph)
        # 修复线性层维度，应该是in_channels而不是1
        self.linears = nn.ModuleList(
            [nn.Linear(in_channels, out_channels, bias=bias)] +
            [nn.Linear(out_channels, out_channels, bias=bias) for _ in range(n_layers-1)]
        )
        self.mat_2 = matrices.square()
        self.mask = self.generate_mask()

    def generate_mask(self):
        # 距离为0的位置设为1，对角线设为0，与DSCL-master保持一致
        matrix = (self.matrices.cpu().detach().numpy() == 0.).astype(int)
        # 对角线设为0
        matrix[:, range(matrix.shape[1]), range(matrix.shape[1])] = 0
        return torch.Tensor(matrix) == 1

    @property
    def STS(self):
        sigma = self.sigma.weight.reshape(self.n_graph, 1, -1)
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        exp = torch.exp(-self.matrices / (2 * sigma**2))
        prior = exp / (math.sqrt(2 * math.pi) * sigma)
        prior = prior.masked_fill(self.mask.to(exp.device), 0)
        prior /= prior.sum(1, keepdims=True) + 1e-8
        prior = prior.permute(1, 2, 0) @ torch.softmax(self.alpha, 0)
        return prior

    def forward(self, x):
        # 直接对整个批次处理，与DSCL-master保持一致
        prior = self.STS  # [N, N]
        
        for i in range(self.n_layers):
            wx = prior @ x  # [B, N, C]
            x = self.activation(self.linears[i](wx))
        
        return x, prior

def process_normal_data(batch_size=20):
    print("=== 开始处理正常数据 ===\n")
    
    # 1. 加载数据
    print("1. 加载数据...")
    # X_normal = np.load('data/temporal_masked_train.npy')  # [14, 144, 263, 2]
    X_normal = np.load('data/processed/train_data_with_anomalies_3d.npy')
    print(f"原始数据形状: {X_normal.shape}")
    # 如果是3维，自动扩展为4维
    if X_normal.ndim == 3:
        X_normal = np.expand_dims(X_normal, axis=0)
        print(f"扩展为4维后形状: {X_normal.shape}")
    
    # 加载三种邻接矩阵
    print("\n加载邻接矩阵...")
    adj_distance = np.load('data/processed/dist.npy')
    adj_correlation = np.load('data/processed/adj.npy')
    adj_connectivity = np.load('data/processed/poi_sim.npy')
    adj_matrices = torch.stack([
        torch.FloatTensor(adj_distance),
        torch.FloatTensor(adj_correlation),
        torch.FloatTensor(adj_connectivity)
    ])
    print(f"邻接矩阵形状: {adj_matrices.shape}")
    
    # 2. 时间注意力处理，先判断是否有缓存
    temporal_save_path = 'data/processed/temporal_attention_processed.npy'
    if os.path.exists(temporal_save_path):
        print(f"\n检测到已存在时间注意力处理结果，直接加载: {temporal_save_path}")
        all_processed_X = np.load(temporal_save_path)
    else:
        print("\n未检测到缓存，逐节点做时间自注意力处理...")
        num_days, num_freq, num_nodes, num_features = X_normal.shape
        all_processed_X = []
        for i in range(num_nodes):
            #print(f"\n处理节点 {i} ...")
            # 获取当前节点的所有时间步
            node_data = X_normal[:, :, i:i+1, :]  # [1, 2016, 1, 2]
            node_tensor = torch.FloatTensor(node_data)
            processed_X, _ = process_temporal_masked_data(
                node_tensor,
                d_model=256,
                dim_k=32,
                dim_v=32,
                n_heads=8,
                dim_fc=64,
                device='cpu'
            )  # [1, 2016, 256]
            all_processed_X.append(processed_X.detach().cpu().numpy())
            #print(f"已处理节点 {i}")
        all_processed_X = np.concatenate(all_processed_X, axis=0)  # [263, 2016, 256]
        print("\n时间注意力处理完成，保存结果...")
        np.save(temporal_save_path, all_processed_X)
        print(f"已保存到 {temporal_save_path}")
    print("\n时间注意力处理完成，开始空间GCN处理...")
    
    # 3. 整体做空间GCN
    all_processed_X_tensor = torch.FloatTensor(all_processed_X)  # [263, 2016, 256]
    all_processed_X_tensor = all_processed_X_tensor.permute(1, 0, 2)  # [2016, 263, 256]
    all_processed_X_tensor = all_processed_X_tensor.unsqueeze(0)      # [1, 2016, 263, 256]
    
    multiple_gcn = MultipleGCN(
        in_channels=all_processed_X_tensor.shape[-1],
        out_channels=128,
        matrices=adj_matrices,
        n_layers=2
    )
    
    spatial_features, static_scores = multiple_gcn(all_processed_X_tensor)  # [1, 2016, 263, 128], [263, 263]
    print(f"空间特征形状: {spatial_features.shape}")
    print(f"静态流注意力分数形状: {static_scores.shape}")
    print("\n=== 处理完成 ===")
    
    # 保存静态流注意力分数，用于双流对比损失
    np.save('data/processed/static_scores.npy', static_scores.detach().cpu().numpy())
    print("静态流注意力分数已保存到 data/processed/static_scores.npy")

if __name__ == "__main__":
    process_normal_data() 