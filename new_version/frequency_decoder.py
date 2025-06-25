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

class FrequencyEncoder(nn.Module):
    """
    频率编码器，模仿 FreEnc 的设计。
    现在 forward 方法会解包并返回特征列表和注意力列表。
    """
    def __init__(
        self,
        c_in: int = 2,
        d_model: int = 256,
        n_heads: int = 8,
        e_layers: int = 3,
        dropout: float = 0.1
    ):
        super(FrequencyEncoder, self).__init__()
        
        # 输入投影层
        self.input_projection = nn.Linear(c_in, d_model)
        
        # 核心编码器
        self.encoder = Encoder(
            [
                AttentionLayer(d_model, n_heads, dropout=dropout) for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )
        
        # 输出投影层，用于最终的注意力计算
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        处理经过频率掩码和逆变换后的数据
        
        Args:
            x: 输入数据 [B, T, C] (例如 B, T, 2)
            
        Returns:
            Tuple[list, list]:
                - feature_outputs (list): 包含每一层特征输出的列表
                - attention_weights (list): 包含每一层注意力权重的列表
        """
        # 1. 输入投影: [B, T, C] -> [B, T, D]
        x_emb = self.input_projection(x)
        
        # 2. 编码: 返回最终输出、层级特征列表、层级注意力列表
        _, feature_outputs, attention_weights = self.encoder(x_emb)
        
        # 注意：原代码的 final_projection 在这里可能不再需要，
        # 因为我们现在关心的是编码器内部的特征。
        # 为了保持接口清晰，我们暂时只返回编码器的直接输出。
        
        return feature_outputs, attention_weights

def main():
    # 设置参数
    d_model = 512
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
    torch.save(feature_outputs, 'feature_outputs.pt')
    torch.save(attention_weights, 'attention_weights.pt')

if __name__ == '__main__':
    main() 