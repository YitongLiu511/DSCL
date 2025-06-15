import torch
import numpy as np
import torch.nn.functional as F
import math
from temporal_attention import process_temporal_masked_data
import torch.nn as nn

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
                for _ in range(n_layers - 1)
            ])
        self.mat_2 = dist_mat.square()
        self.mask = self.generate_mask()

    def generate_mask(self):
        matrix = (self.dist_mat.cpu().detach().numpy() == 0.).astype(int)
        matrix = torch.tensor(matrix, device=self.dist_mat.device)
        return matrix

    def forward(self, x):
        # 调整输入维度以适应GCN处理
        B, T, D = x.shape
        x = x.reshape(B * T, D)  # 将时间维度合并到批次维度
        
        sigma = torch.sigmoid(self.sigma.weight * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        exp = torch.exp(-self.mat_2 / (2 * sigma**2))
        prior = exp / (math.sqrt(2 * math.pi) * sigma)
        prior = prior * self.mask

        for i in range(self.n_layers):
            wx = prior @ x
            x = self.activation(self.linears[i](wx))
        
        # 恢复原始维度
        x = x.reshape(B, T, self.out_channels)
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
        self.linears = nn.ModuleList(
            [nn.Linear(1, out_channels, bias=bias)] +
            [nn.Linear(out_channels, out_channels, bias=bias) for _ in range(n_layers-1)]
        )
        self.mat_2 = matrices.square()
        self.mask = self.generate_mask()

    def generate_mask(self):
        matrix = (self.matrices.cpu().detach().numpy() == 0.).astype(bool)
        matrix = torch.tensor(matrix, device=self.matrices.device)
        return matrix

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
        # x: [B, T, N] -> [B, T, N, 1]
        if x.dim() == 3:
            x = x.unsqueeze(-1)
        B, T, N, F = x.shape
        prior = self.STS  # [N, N]
        for i in range(self.n_layers):
            # 空间卷积
            x = torch.matmul(prior, x)  # [B, T, N, F]
            # 特征变换
            x = self.activation(self.linears[i](x))  # [B, T, N, out_channels]
        return x, prior

def process_normal_data(batch_size=20):
    print("=== 开始处理正常数据 ===\n")
    
    # 1. 加载数据
    print("1. 加载数据...")
    data_path = 'data/processed'
    X_normal = np.load(f'{data_path}/X_normal.npy')  # [14, 144, 263, 2]
    print(f"正常训练集形状: {X_normal.shape}")
    
    # 加载三种邻接矩阵
    print("\n加载邻接矩阵...")
    adj_distance = np.load(f'{data_path}/dist.npy')
    adj_correlation = np.load(f'{data_path}/adj.npy')
    adj_connectivity = np.load(f'{data_path}/poi_sim.npy')
    adj_matrices = torch.stack([
        torch.FloatTensor(adj_distance),
        torch.FloatTensor(adj_correlation),
        torch.FloatTensor(adj_connectivity)
    ])
    print(f"邻接矩阵形状: {adj_matrices.shape}")
    
    # 2. 分批处理节点，只做时间注意力
    num_days, num_freq, num_nodes, num_features = X_normal.shape
    all_processed_X = []
    for start in range(0, num_nodes, batch_size):
        end = min(start + batch_size, num_nodes)
        print(f"\n处理节点 {start} 到 {end-1} ...")
        # 取出当前批次的节点数据 [14, 144, batch, 2]
        batch_data = X_normal[:, :, start:end, :]
        batch_tensor = torch.FloatTensor(batch_data)
        # 时间注意力处理
        processed_X, _ = process_temporal_masked_data(
            batch_tensor,
            d_model=256,
            dim_k=32,
            dim_v=32,
            n_heads=8,
            dim_fc=64,
            device='cpu'
        )  # [batch, 2016, 256]
        all_processed_X.append(processed_X.detach().cpu().numpy())
        print(f"已处理节点 {start} 到 {end-1}")
    # 合并所有批次
    all_processed_X = np.concatenate(all_processed_X, axis=0)  # [263, 2016, 256]
    print("\n时间注意力处理完成，开始空间GCN处理...")
    # 3. 整体做空间GCN
    all_processed_X_tensor = torch.FloatTensor(all_processed_X)
    multiple_gcn = MultipleGCN(
        in_channels=all_processed_X_tensor.shape[-1],
        out_channels=128,
        matrices=adj_matrices,
        n_layers=2
    )
    spatial_features, _ = multiple_gcn(all_processed_X_tensor)  # [263, 2016, 128]
    # 保存最终结果
    save_path = 'data/processed'
    np.save(f'{save_path}/normal_processed_X.npy', all_processed_X)
    np.save(f'{save_path}/normal_spatial_features.npy', spatial_features.detach().cpu().numpy())
    print(f"\n所有数据已保存到 {save_path} 目录")
    print("保存的数据包括：")
    print("1. 处理后的正常数据：normal_processed_X.npy")
    print("2. 空间特征：normal_spatial_features.npy")
    print("\n=== 处理完成 ===")

if __name__ == "__main__":
    process_normal_data() 