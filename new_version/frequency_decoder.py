import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

class AttentionLayer(nn.Module):
    """
    标准自注意力层的正确实现，替代了原有的错误版本。
    这个版本手动实现了scaled dot-product attention，以确保维度处理的正确性。
    """
    def __init__(self, d_model, n_heads=8, d_keys=None, d_values=None, dropout=0.1):
        super(AttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.d_keys = d_keys or (d_model // n_heads)
        self.d_values = d_values or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, self.d_keys * self.n_heads)
        self.key_projection = nn.Linear(d_model, self.d_keys * self.n_heads)
        self.value_projection = nn.Linear(d_model, self.d_values * self.n_heads)
        self.out_projection = nn.Linear(self.d_values * self.n_heads, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        
        residual = queries

        # 1. Project and split heads
        q = self.query_projection(queries).view(B, L, H, self.d_keys)
        k = self.key_projection(keys).view(B, S, H, self.d_keys)
        v = self.value_projection(values).view(B, S, H, self.d_values)

        # 2. Transpose for attention calculation
        q = q.transpose(1, 2)  # (B, H, L, d_keys)
        k = k.transpose(1, 2)  # (B, H, S, d_keys)
        v = v.transpose(1, 2)  # (B, H, S, d_values)

        # 3. Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_keys)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, v)

        # 4. Concatenate heads and project back
        context = context.transpose(1, 2).contiguous().view(B, L, -1)
        output = self.out_projection(context)
        
        # 5. Residual connection and layer norm
        return self.norm(residual + self.dropout(output)), attn_weights

class Encoder(nn.Module):
    """
    通用编码器模块，由多个注意力层堆叠而成。
    现在会同时返回每一层的特征输出和注意力权重。
    """
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x):
        # x [B, T, D]
        outlist = []
        attlist = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, x, x)
            outlist.append(x)
            attlist.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        # 返回最终输出，以及每一层的特征输出列表和注意力权重列表
        return x, outlist, attlist

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
    def forward(self, data=None, idx=None):
        if data is not None:
            p = self.pe[:data].unsqueeze(0)
        else:
            p = self.pe.unsqueeze(0).repeat(idx.shape[0],1,1)[torch.arange(idx.shape[0])[:,None],idx,:]
        return p

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        d_ff = d_ff or d_model * 4
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

class FrequencyEncoder(nn.Module):
    def __init__(
        self,
        c_in: int = 2,
        d_model: int = 64,
        n_heads: int = 8,
        e_layers: int = 3,
        dropout: float = 0.1
    ):
        super(FrequencyEncoder, self).__init__()
        self.input_projection = nn.Linear(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model*4, dropout) for _ in range(e_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        # print('FrequencyEncoder 输入 x 的 shape:', x.shape)
        x_emb = self.input_projection(x)
        x_emb = x_emb + self.position_embedding(data=x_emb.shape[1])
        for layer in self.layers:
            x_emb = layer(x_emb)
        x_emb = self.norm(x_emb)
        return x_emb  # [B, T, D]

def main():
    # 设置参数
    d_model = 64
    nhead = 8
    num_decoder_layers = 6
    dim_feedforward = 2048
    dropout = 0.1
    
    # 创建模型
    model = FrequencyEncoder(
        c_in=2,
        d_model=d_model,
        n_heads=nhead,
        e_layers=num_decoder_layers,
        dropout=dropout
    )
    
    # 加载频域掩码后的数据
    masked_data = load_frequency_masked_data('path_to_your_masked_data.pt')
    original_data = load_frequency_masked_data('path_to_your_original_data.pt')
    
    # 前向传播
    feature_outputs, attention_weights = model(original_data)
    
    # 保存输出


if __name__ == '__main__':
    main() 