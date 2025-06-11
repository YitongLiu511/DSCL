import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, SAGEConv, GATConv
from torch_geometric.utils import dense_to_sparse
import math

class MultipleGCN(nn.Module):
    def __init__(self, in_channels, out_channels, n_views=2, dropout=0.0, gcn_type='cheb', K=2, bias=True):
        super(MultipleGCN, self).__init__()
        self.n_views = n_views
        self.gcn_type = gcn_type
        
        # 为每个视图创建独立的GCN层
        self.gcns = nn.ModuleList()
        for i in range(n_views):
            if gcn_type == 'cheb':
                self.gcns.append(ChebConv(in_channels, out_channels, K, bias=bias))
            elif gcn_type == 'gcn':
                self.gcns.append(GCNConv(in_channels, out_channels, bias=bias))
            elif gcn_type == 'sage':
                self.gcns.append(SAGEConv(in_channels, out_channels, bias=bias))
            elif gcn_type == 'gat':
                self.gcns.append(GATConv(in_channels, out_channels, heads=1, bias=bias))
            else:
                raise ValueError(f"Unsupported GCN type: {gcn_type}")
        
        # 特征投影层，用于后续处理
        self.projection = nn.Linear(out_channels * n_views, out_channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, adj_list):
        """
        x: [B, N, C]，B为批次大小，N为节点数，C为特征维度
        adj_list: 图邻接矩阵列表，长度为n_views
        """
        batch_size, num_nodes, _ = x.size()
        outputs = []
        
        # 对每个视图应用GCN
        for i in range(self.n_views):
            # 重塑为[B*N, C]以批量处理
            x_flat = x.reshape(-1, x.size(-1))
            
            # 将numpy数组转换为PyTorch张量
            adj = torch.from_numpy(adj_list[i]).float()
            
            # 将邻接矩阵转换为边索引格式
            edge_index, edge_weight = dense_to_sparse(adj)
            
            # 应用GCN
            x_gcn = self.gcns[i](x_flat, edge_index, edge_weight)
            
            # 恢复维度
            x_gcn = x_gcn.reshape(batch_size, num_nodes, -1)
            outputs.append(x_gcn)
        
        # 拼接所有视图的输出
        out = torch.cat(outputs, dim=-1)
        
        # 应用投影层
        out = self.projection(out)
        out = self.dropout(out)
        
        return out

class SpatialAttention(nn.Module):
    def __init__(self, in_channels, out_channels, n_heads=8, dropout=0.0):
        super(SpatialAttention, self).__init__()
        self.n_heads = n_heads
        self.head_dim = out_channels // n_heads
        
        # 线性投影层
        self.q_linear = nn.Linear(in_channels, out_channels)
        self.k_linear = nn.Linear(in_channels, out_channels)
        self.v_linear = nn.Linear(in_channels, out_channels)
        self.out_proj = nn.Linear(out_channels, out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, adj=None):
        """
        x: [B, N, C]，B为批次大小，N为节点数，C为特征维度
        adj: 可选的邻接矩阵，用于掩码注意力
        """
        batch_size, num_nodes, _ = x.size()
        
        # 线性投影并分割成多头
        q = self.q_linear(x).view(batch_size, num_nodes, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, num_nodes, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, num_nodes, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力得分
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用邻接矩阵作为注意力掩码（如果提供）
        if adj is not None:
            # 将邻接矩阵扩展到多头维度
            adj_expanded = adj.unsqueeze(0).unsqueeze(1).expand(batch_size, self.n_heads, -1, -1)
            # 将0替换为负无穷大，使softmax后的值接近0
            scores = scores.masked_fill(adj_expanded == 0, -1e9)
        
        # 应用softmax获取注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重到值向量
        attn_output = torch.matmul(attn_weights, v)
        
        # 合并多头输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_nodes, -1)
        
        # 应用输出投影
        output = self.out_proj(attn_output)
        
        return output, attn_weights

class SpatialAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_views=2, n_heads=8, dropout=0.0, gcn_type='cheb', K=2):
        super(SpatialAttentionBlock, self).__init__()
        
        # 多视图GCN层
        self.gcn = MultipleGCN(
            in_channels=in_channels,
            out_channels=out_channels,
            n_views=n_views,
            dropout=dropout,
            gcn_type=gcn_type,
            K=K
        )
        
        # 空间自注意力层
        self.attention = SpatialAttention(
            in_channels=out_channels,
            out_channels=out_channels,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(out_channels, out_channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels * 4, out_channels),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, adj_list):
        """
        x: [B, N, C]，B为批次大小，N为节点数，C为特征维度
        adj_list: 图邻接矩阵列表，长度为n_views
        """
        # 1. 多视图GCN处理
        gcn_out = self.gcn(x, adj_list)
        
        # 残差连接和层归一化
        x = x + self.dropout(gcn_out)
        x = self.norm1(x)
        
        # 2. 空间自注意力处理
        attn_out, _ = self.attention(x)
        
        # 残差连接和层归一化
        x = x + self.dropout(attn_out)
        x = self.norm2(x)
        
        # 3. 前馈网络
        ff_out = self.feed_forward(x)
        
        # 残差连接
        x = x + self.dropout(ff_out)
        
        return x

def process_spatial_attention(temporal_output, adj_list, d_model=512, num_graphs=2, device='cpu'):
    """
    处理空间注意力的便捷函数
    
    Args:
        temporal_output: 时间注意力处理后的输出 [B, T, N]
        adj_list: 图邻接矩阵列表
        d_model: 模型维度
        num_graphs: 图的数量
        device: 设备类型
        
    Returns:
        processed_data: 处理后的数据 [B, T, N]
    """
    # 确保输入在CPU上
    temporal_output = temporal_output.cpu()
    
    # 创建处理器
    processor = SpatialAttentionBlock(
        in_channels=1,  # 输入维度为1，因为每个节点只有一个特征
        out_channels=d_model,
        n_views=num_graphs,
        n_heads=8,
        dropout=0.1,
        gcn_type='gcn'
    ).to(device)
    
    # 处理数据
    with torch.no_grad():
        # 对每个时间步分别处理
        batch_size, time_steps, num_nodes = temporal_output.size()
        spatial_outputs = []
        
        for t in range(time_steps):
            x_t = temporal_output[:, t, :]  # [B, N]
            x_t = x_t.unsqueeze(-1)  # [B, N, 1]
            x_s = processor(x_t, adj_list)  # [B, N, D]
            x_s = x_s.mean(dim=-1)  # [B, N]
            spatial_outputs.append(x_s)
        
        # 堆叠所有时间步的输出
        processed_data = torch.stack(spatial_outputs, dim=1)  # [B, T, N]
    
    return processed_data 