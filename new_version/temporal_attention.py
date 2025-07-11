import torch
from torch import nn
import math
import torch.nn.functional as F
import numpy as np

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, idx):
        return self.pe[idx]

# 1. 读取掩码数据和掩码索引
masked_data = np.load('data/datanew/temporal_masked.npy')
mask_indices = np.load('data/datanew/temporal_mask_indices.npy')  # [num_nodes, 掩码数]

if masked_data.ndim == 3:
    # [num_nodes, num_time, num_features]
    num_nodes, num_time, num_features = masked_data.shape
    all_time = num_time
elif masked_data.ndim == 4:
    # [天, 槽, 节点, 特征]
    num_days, num_time, num_nodes, num_features = masked_data.shape
    masked_data = masked_data.transpose(2, 0, 1, 3).reshape(num_nodes, num_days * num_time, num_features)
    all_time = num_days * num_time
else:
    raise ValueError("masked_data shape 不支持")

unmasked_mask = np.ones((num_nodes, all_time), dtype=bool)
for node in range(num_nodes):
    unmasked_mask[node, mask_indices[node]] = False

# 3. 定义位置编码函数

def get_positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model), dtype=np.float32)
    position = np.arange(0, seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe

# 4. 给未掩码和掩码部分分别添加位置编码
unmasked_data = []
for node in range(num_nodes):
    node_unmask_idx = np.where(unmasked_mask[node])[0]
    node_unmask = masked_data[node, node_unmask_idx, :]  # [未掩码数, 特征数]
    pe = get_positional_encoding(len(node_unmask_idx), node_unmask.shape[1])
    node_unmask_pe = node_unmask + pe
    unmasked_data.append(node_unmask_pe)
# unmasked_data: list，每个元素是该节点未掩码部分加了位置编码的数据

masked_data_list = []
for node in range(num_nodes):
    node_mask_idx = mask_indices[node]
    node_mask = masked_data[node, node_mask_idx, :]  # [掩码数, 特征数]
    pe = get_positional_encoding(len(node_mask_idx), node_mask.shape[1])
    node_mask_pe = node_mask + pe
    masked_data_list.append(node_mask_pe)
# masked_data_list: list，每个元素是该节点掩码部分加了位置编码的数据

# 5. 后续可将unmasked_data送入attention编码器

class MultiheadAttention(nn.Module):
    '''For the shape (B, L, D)'''

    def __init__(
        self,
        d_model: int,
        dim_k: int,
        dim_v: int,
        n_heads: int,
        device: str = 'cpu',
    ) -> None:
        super().__init__()
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.n_heads = n_heads
        self.device = device

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
        # 确保所有模块都在输入设备上
        self.q = self.q.to(x.device)
        self.k = self.k.to(x.device)
        self.v = self.v.to(x.device)
        self.o = self.o.to(x.device)
        
        # print("[TemporalAttn] input x mean/std:", x.mean().item(), x.std().item())
        Q = self.q(x).reshape(B, L, self.n_heads, -1)
        K = self.k(x).reshape(B, L, self.n_heads, -1)
        V = self.v(y).reshape(B, L, self.n_heads, -1)
        # 放大Q/K
        Q = Q * 5
        K = K * 5
        scores = torch.einsum("blhe,bshe->bhls", Q, K) * self.norm_fact
        # print("[TemporalAttn] scores mean/std/max/min:", scores.mean().item(), scores.std().item(), scores.max().item(), scores.min().item())
        attn_weights = scores.softmax(dim=-1)
        # print("[TemporalAttn] attn_weights mean/std/max/min:", attn_weights.mean().item(), attn_weights.std().item(), attn_weights.max().item(), attn_weights.min().item())
        output = torch.einsum("bhls,bshd->blhd", attn_weights, V).reshape(B, L, -1)
        return self.o(output), attn_weights

class TemporalTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        dim_k: int,
        dim_v: int,
        n_heads: int,
        dim_fc: int = 128,
        dropout: float = 0.1,
        half: bool = False,
        return_attn: bool = False,
    ) -> None:
        super().__init__()
        self.half_ = half
        self.d_model = d_model
        self.attn = MultiheadAttention(d_model, dim_k, dim_v, n_heads)
        self.conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=dim_fc,
            kernel_size=1,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        if not half:
            self.conv2 = nn.Conv1d(
                in_channels=dim_fc,
                out_channels=d_model,
                kernel_size=1,
            )
        self.norm2 = nn.LayerNorm(d_model)
        self.return_attn = return_attn

    def forward(self, x):
        x_, attn = self.attn(x, x)
        y = x = self.norm1(self.dropout(x + x_))
        y = self.dropout(torch.relu(self.conv1(y.transpose(-1, 1))))
        if self.half_:
            if self.return_attn:
                return y, attn
            return y
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        output = self.norm2(x + y)
        if self.return_attn:
            return output, attn
        return output

class TemporalAttentionProcessor(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        dim_k: int = 32,
        dim_v: int = 32,
        n_heads: int = 4,
        dim_fc: int = 64,
        device: str = 'cpu',
        input_dim: int = 2,  # 新增参数，默认为2
    ):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        
        # 输入投影层，将输入维度转换为d_model
        self.input_projection = nn.Linear(input_dim, d_model).to(device)
        
        # 位置编码层
        self.positional_embedding = PositionalEmbedding(d_model).to(device)
        
        # 多层时间注意力块
        self.attention_layers = nn.ModuleList([
            TemporalAttentionLayer(d_model, dim_k, dim_v, n_heads, dim_fc, device)
            for _ in range(3)  # 3层注意力
        ])
        
        # 输出投影层
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, input_dim)  # 输出维度与输入维度一致
        ).to(device)
        
    def forward(self, x):
        """
        处理经过时间掩码的数据
        
        Args:
            x: 输入张量 [263, 2016, 2]
            # 263: 节点数（作为批次大小）
            # 2016: 时间步长
            # 2: 特征数
            
        Returns:
            output: 处理后的张量 [263, 2016, 2]
            attention_weights: 所有层的注意力权重列表
        """
        # 确保输入在正确的设备上
        x = x.to(self.device)
        
        # 确保所有模块都在输入设备上
        self.input_projection = self.input_projection.to(x.device)
        self.positional_embedding = self.positional_embedding.to(x.device)
        self.output_projection = self.output_projection.to(x.device)
        
        # 输入投影
        x = self.input_projection(x)  # [263, 2016, d_model]
        
        # 添加位置编码
        T = x.shape[1]  # 时间步长
        pos_emb = self.positional_embedding(torch.arange(T, device=x.device))  # [T, d_model]
        x = x + pos_emb.unsqueeze(0)  # [263, 2016, d_model]
        
        # 存储所有层的注意力权重
        attention_weights = []
        
        # 通过多层时间注意力块
        for layer in self.attention_layers:
            layer = layer.to(x.device)  # 确保层在正确设备上
            x, attn = layer(x)
            attention_weights.append(attn)
        
        # 输出投影
        output = self.output_projection(x)  # [263, 2016, 2]
        
        return output, attention_weights

class TemporalAttentionLayer(nn.Module):
    def __init__(self, d_model=256, dim_k=32, dim_v=32, nhead=4, dim_fc=64, device='cpu'):
        super().__init__()
        self.attn = MultiheadAttention(d_model, dim_k, dim_v, nhead, device)
        self.fc1 = nn.Linear(d_model, dim_fc)
        self.fc2 = nn.Linear(dim_fc, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # 确保所有模块都在输入设备上
        self.attn = self.attn.to(x.device)
        self.fc1 = self.fc1.to(x.device)
        self.fc2 = self.fc2.to(x.device)
        self.norm1 = self.norm1.to(x.device)
        self.norm2 = self.norm2.to(x.device)
        
        x_, attn = self.attn(x, x)
        x = self.norm1(x + self.dropout(x_))
        x_ = self.fc2(F.relu(self.fc1(x)))
        x = self.norm2(x + self.dropout(x_))
        return x, attn

def preprocess_masked_data(masked_data):
    """
    预处理掩码后的数据，将其转换为时间注意力模型所需的格式
    
    Args:
        masked_data: 掩码后的数据 [14, 144, 263, 2]
        
    Returns:
        processed_data: 处理后的数据 [263, 2016, 2]
        # 263: 节点数（作为批次大小）
        # 2016: 时间步长 (14 * 144)
        # 2: 特征数
    """
    num_days, num_freq, num_nodes, num_features = masked_data.shape
    
    # 合并天数和时间槽维度，并调整维度顺序
    x = masked_data.reshape(-1, num_nodes, num_features)  # [2016, 263, 2]
    x = x.permute(1, 0, 2)  # [263, 2016, 2]
    
    return x

def process_temporal_masked_data(data, d_model=256, dim_k=32, dim_v=32, n_heads=4, dim_fc=64, device='cpu', input_dim=2):
    """
    处理时间掩码后的数据
    Args:
        data: 输入数据 [B, T, N, D]
        d_model: 模型维度
        dim_k: 键维度
        dim_v: 值维度
        n_heads: 注意力头数
        dim_fc: 前馈网络维度
        device: 设备
        input_dim: 输入特征维度
    Returns:
        processed_data: 处理后的数据
        attention_weights: 注意力权重
    """
    # 预处理数据
    processed_data = preprocess_masked_data(data)
    
    # 初始化处理器
    processor = TemporalAttentionProcessor(
        d_model=d_model,
        dim_k=dim_k,
        dim_v=dim_v,
        n_heads=n_heads,
        dim_fc=dim_fc,
        device=device,
        input_dim=input_dim
    )
    
    # 处理数据
    output, attention_weights = processor(processed_data)
    
    return output, attention_weights 