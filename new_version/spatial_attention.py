import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, SAGEConv, GATConv
from torch_geometric.utils import dense_to_sparse
import math
import numpy as np

class MultiheadAttention(nn.Module):
    '''For the shape (B, L, D)'''

    def __init__(
        self,
        d_model: int,
        dim_k: int,
        dim_v: int,
        n_heads: int,
    ) -> None:
        super().__init__()
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.n_heads = n_heads

        self.q = nn.Linear(d_model, dim_k * n_heads)
        self.k = nn.Linear(d_model, dim_k * n_heads)
        self.v = nn.Linear(d_model, dim_v * n_heads)
        self.o = nn.Linear(dim_v * n_heads, d_model)
        self.norm_fact = 1 / math.sqrt(d_model)

    def attention(self, Q, K, V):
        B, L = Q.shape[:2]
        scores = torch.einsum("blhe,bshe->bhls", Q, K) * self.norm_fact
        scores = scores.softmax(dim=-1)
        output = torch.einsum("bhls,bshd->blhd", scores, V).reshape(B, L, -1)
        return output, scores

    def forward(self, x, y):
        '''x : (B, L, D)'''
        B, L, _ = x.shape
        Q = self.q(x).reshape(B, L, self.n_heads, -1)  # (N, B, L, K)
        K = self.k(x).reshape(B, L, self.n_heads, -1)  # (N, B, L, K)
        V = self.v(y).reshape(B, L, self.n_heads, -1)  # (N, B, L, K)
        output, scores = self.attention(Q, K, V)
        return self.o(output), scores

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
            
            # 确保邻接矩阵是 PyTorch 张量
            if not isinstance(adj_list[i], torch.Tensor):
                adj_list[i] = torch.from_numpy(adj_list[i]).float()
            
            # 将邻接矩阵转换为边索引格式
            edge_index, edge_weight = dense_to_sparse(adj_list[i])
            
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
        assert out_channels % n_heads == 0, "out_channels 必须能被 n_heads 整除"
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
        batch_size, num_nodes, in_channels = x.size()
        
        # 线性投影
        q = self.q_linear(x)  # [B, N, out_channels]
        k = self.k_linear(x)  # [B, N, out_channels]
        v = self.v_linear(x)  # [B, N, out_channels]
        
        # 重塑为多头形式
        q = q.view(batch_size, num_nodes, self.n_heads, self.head_dim).transpose(1, 2)  # [B, n_heads, N, head_dim]
        k = k.view(batch_size, num_nodes, self.n_heads, self.head_dim).transpose(1, 2)  # [B, n_heads, N, head_dim]
        v = v.view(batch_size, num_nodes, self.n_heads, self.head_dim).transpose(1, 2)  # [B, n_heads, N, head_dim]
        
        # 计算注意力得分
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, n_heads, N, N]
        
        # 应用邻接矩阵作为注意力掩码（如果提供）
        if adj is not None:
            # 将邻接矩阵扩展到多头维度
            adj_expanded = adj.unsqueeze(0).unsqueeze(1).expand(batch_size, self.n_heads, -1, -1)
            # 将0替换为负无穷大，使softmax后的值接近0
            scores = scores.masked_fill(adj_expanded == 0, -1e9)
        
        # 应用softmax获取注意力权重
        attn_weights = F.softmax(scores, dim=-1)  # [B, n_heads, N, N]
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重到值向量
        attn_output = torch.matmul(attn_weights, v)  # [B, n_heads, N, head_dim]
        
        # 合并多头输出
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, N, n_heads, head_dim]
        attn_output = attn_output.view(batch_size, num_nodes, -1)  # [B, N, out_channels]
        
        # 应用输出投影
        output = self.out_proj(attn_output)  # [B, N, out_channels]
        
        return output, scores

class SpatialAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels=2, n_views=2, n_heads=8, dropout=0.0, gcn_type='cheb', K=2):
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
        Returns:
            output: 处理后的特征 [B, N, out_channels]
            scores: 空间注意力分数 [B, n_heads, N, N]
        """
        # 1. 多视图GCN处理
        gcn_out = self.gcn(x, adj_list)
        
        # 残差连接和层归一化
        x = x + self.dropout(gcn_out)
        x = self.norm1(x)
        
        # 2. 空间自注意力处理 - 获取注意力分数
        attn_out, attn_scores = self.attention(x)
        
        # 残差连接和层归一化
        x = x + self.dropout(attn_out)
        x = self.norm2(x)
        
        # 3. 前馈网络
        ff_out = self.feed_forward(x)
        
        # 残差连接
        output = x + self.dropout(ff_out)
        
        # 返回输出和注意力分数，就像DSCL-master中的MultiheadAttention一样
        return output, attn_scores

def process_spatial_attention(temporal_output, adj_list, d_model=256, num_graphs=2, device='cpu'):
    """
    处理空间注意力的便捷函数
    
    Args:
        temporal_output: 时间注意力处理后的输出 [B, T, N]，其中N是节点数（263）
        adj_list: 图邻接矩阵列表
        d_model: 模型维度，默认为256（可以被8整除）
        num_graphs: 图的数量
        device: 设备类型
        
    Returns:
        processed_data: 处理后的数据 [B, T, N]
        attention_scores: 注意力分数 [B, T, n_heads, N, N]
    """
    print("\n=== 开始空间注意力处理 ===")
    print(f"输入形状: {temporal_output.shape}")
    print(f"参数设置: d_model={d_model}, num_graphs={num_graphs}, device={device}")
    
    # 确保输入在CPU上
    temporal_output = temporal_output.cpu()
    
    # 获取输入维度
    batch_size, time_steps, num_nodes = temporal_output.size()
    print(f"批次大小: {batch_size}, 时间步数: {time_steps}, 节点数: {num_nodes}")
    
    # 创建处理器
    print("\n初始化空间注意力处理器...")
    processor = SpatialAttentionBlock(
        in_channels=1,  # 每个节点的特征维度是1
        out_channels=2,  # 修改为2，与频域解码器输入维度一致
        n_views=num_graphs,
        n_heads=8,
        dropout=0.1,
        gcn_type='gcn'
    ).to(device)
    
    # 处理数据
    print("\n开始处理每个时间步...")
    with torch.no_grad():
        # 对每个时间步分别处理
        spatial_outputs = []
        attention_scores_list = []
        
        for t in range(time_steps):
            #print(f"\n处理第 {t+1}/{time_steps} 个时间步:")
            x_t = temporal_output[:, t, :]  # [B, N]
            #print(f"  当前时间步输入形状: {x_t.shape}")
            
            x_t = x_t.unsqueeze(-1)  # [B, N, 1]
            #print(f"  重塑后形状: {x_t.shape}")
            
            x_s, scores = processor(x_t, adj_list)  # [B, N, D], [B, n_heads, N, N]
            #print(f"  空间注意力处理输出形状: {x_s.shape}")
            
            x_s = x_s.mean(dim=-1)  # [B, N]
            #print(f"  平均池化后形状: {x_s.shape}")
            
            spatial_outputs.append(x_s)
            attention_scores_list.append(scores)
            #print(f"  添加到输出列表，当前列表长度: {len(spatial_outputs)}")
        
        # 堆叠所有时间步的输出
        processed_data = torch.stack(spatial_outputs, dim=1)  # [B, T, N]
        attention_scores = torch.stack(attention_scores_list, dim=1)  # [B, T, n_heads, N, N]
        print(f"\n堆叠后最终输出形状: {processed_data.shape}")
        print(f"注意力分数形状: {attention_scores.shape}")
    
    print("\n=== 空间注意力处理完成 ===")
    return processed_data, attention_scores

def preprocess_temporal_data(data):
    """
    将时间注意力处理后的数据转换为空间注意力所需的格式
    
    Args:
        data: numpy array, shape [263, 2016, 2]
        
    Returns:
        torch.Tensor, shape [2016, 263, 2]
    """
    # 转置数据，将时间步作为批次
    data = np.transpose(data, (1, 0, 2))
    return torch.FloatTensor(data)

class SpatialAttentionProcessor:
    def __init__(self, d_in=2, d_model=2, dim_k=32, dim_v=32, n_heads=4):
        self.device = torch.device('cpu')
        self.input_proj = nn.Linear(d_in, d_model).to(self.device)
        self.model = MultiheadAttention(d_model, dim_k, dim_v, n_heads).to(self.device)

    def forward(self, x):
        # x: numpy array, shape [263, 2016, 2]
        x = np.transpose(x, (1, 0, 2))  # [2016, 263, 2]
        x = torch.FloatTensor(x).to(self.device)
        x = self.input_proj(x)  # [2016, 263, d_model]
        output, scores = self.model(x, x)
        output = output.detach().cpu().numpy()
        output = np.transpose(output, (1, 0, 2))  # [263, 2016, d_model]
        return output

def process_spatial_attention():
    # 加载时间注意力处理后的数据
    data = np.load('data/temporal_attention_processed_train.npy')
    print(f"加载数据形状: {data.shape}")
    
    # 创建处理器
    processor = SpatialAttentionProcessor()
    
    # 处理数据
    output = processor.forward(data)
    print(f"输出数据形状: {output.shape}")
    
    # 保存结果
    np.save('data/spatial_attention_processed_train.npy', output)
    print("处理完成，结果已保存到 data/spatial_attention_processed_train.npy")

if __name__ == "__main__":
    process_spatial_attention() 