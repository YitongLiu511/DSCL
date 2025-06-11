import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        # 确保d_model能被nhead整除
        assert d_model % nhead == 0, "d_model必须能被nhead整除"
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        # 线性投影层，拆分Q、K、V
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        # 获取输入维度
        bsz, tgt_len, _ = query.size()
        
        # 线性投影
        q = self.q_linear(query)  # [bsz, tgt_len, d_model]
        k = self.k_linear(key)    # [bsz, tgt_len, d_model]
        v = self.v_linear(value)  # [bsz, tgt_len, d_model]
        
        # 重塑为多头形式
        q = q.view(bsz, tgt_len, self.nhead, self.head_dim).transpose(1, 2)  # [bsz, nhead, tgt_len, head_dim]
        k = k.view(bsz, tgt_len, self.nhead, self.head_dim).transpose(1, 2)  # [bsz, nhead, tgt_len, head_dim]
        v = v.view(bsz, tgt_len, self.nhead, self.head_dim).transpose(1, 2)  # [bsz, nhead, tgt_len, head_dim]
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [bsz, nhead, tgt_len, tgt_len]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, v)  # [bsz, nhead, tgt_len, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()  # [bsz, tgt_len, nhead, head_dim]
        attn_output = attn_output.view(bsz, tgt_len, self.d_model)  # [bsz, tgt_len, d_model]
        
        # 输出投影
        output = self.out_proj(attn_output)
        
        return output, attn_weights

class TemporalTransformer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, device='cpu'):
        super().__init__()
        self.device = device
        self.self_attn = MultiheadAttention(d_model, nhead, dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output, attn_weights = self.self_attn(x, x, x, mask=mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        ff_output = self.feedforward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        
        return x, attn_weights

class TemporalAttentionProcessor(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_feedforward=256, dropout=0.1, n_layers=3, device='cpu'):
        super().__init__()
        self.device = device
        # 确保d_model能被nhead整除
        self.d_model = (d_model // nhead) * nhead  # 调整d_model使其能被nhead整除
        self.nhead = nhead
        
        # 输入投影层，将输入维度转换为d_model
        self.input_projection = nn.Linear(263, self.d_model).to(device)  # 263是输入维度
        
        # 多层时间注意力块
        self.layers = nn.ModuleList([
            TemporalTransformer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                device=device
            ).to(device) for _ in range(n_layers)
        ])
        
        # 输出投影层
        self.output_projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 263)  # 输出维度改回263
        ).to(device)
        
    def forward(self, x):
        """
        处理经过时间掩码的数据
        
        Args:
            x: 输入张量 [B, T, D]，其中B是批次大小，T是时间步长，D是特征维度
            
        Returns:
            output: 处理后的张量 [B, T, D]
            attention_weights: 所有层的注意力权重列表
        """
        # 确保输入在正确的设备上
        x = x.to(self.device)
        
        # 输入投影
        x = self.input_projection(x)  # [B, T, d_model]
        
        # 存储所有层的注意力权重
        attention_weights = []
        
        # 通过多层时间注意力块
        for layer in self.layers:
            x, attn_weights = layer(x)
            attention_weights.append(attn_weights)
        
        # 输出投影
        output = self.output_projection(x)  # [B, T, 263]
        
        return output, attention_weights

def process_temporal_masked_data(masked_data, d_model=512, nhead=8, device='cpu'):
    """
    处理经过时间掩码的数据
    
    Args:
        masked_data: 经过时间掩码的数据 [B, T, D]
        d_model: 模型维度
        nhead: 注意力头数
        device: 设备类型
        
    Returns:
        processed_data: 处理后的数据 [B, T, D]
        attention_weights: 注意力权重列表
    """
    # 确保输入数据在CPU上
    masked_data = masked_data.cpu()
    
    # 创建处理器
    processor = TemporalAttentionProcessor(
        d_model=d_model,
        nhead=nhead,
        device=device
    ).to(device)
    
    # 处理数据
    with torch.no_grad():
        processed_data, attention_weights = processor(masked_data)
    
    return processed_data, attention_weights 