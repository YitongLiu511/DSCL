import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

class Decoder(nn.Module):
    def __init__(self, d_model: int, nhead: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        # memory: [B, T, D]
        
        # 自注意力
        x2 = self.norm1(x)
        x2 = x2.transpose(0, 1)  # [T, B, D]
        x2, _ = self.self_attn(x2, x2, x2)
        x2 = x2.transpose(0, 1)  # [B, T, D]
        x = x + self.dropout1(x2)

        # 交叉注意力
        x2 = self.norm2(x)
        x2 = x2.transpose(0, 1)  # [T, B, D]
        memory = memory.transpose(0, 1)  # [T, B, D]
        x2, _ = self.multihead_attn(x2, memory, memory)
        x2 = x2.transpose(0, 1)  # [B, T, D]
        x = x + self.dropout2(x2)

        # 前馈网络
        x2 = self.norm3(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x2))))
        x = x + self.dropout3(x2)

        return x

class FrequencyDecoder(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        dim_fc: int = 128,
        dropout: float = 0.1,
        n_layers: int = 3,
        device: str = 'cpu'
    ):
        super(FrequencyDecoder, self).__init__()
        self.device = device
        self.d_model = d_model
        
        # 修改输入投影层维度为2，因为输入特征是2维的
        self.input_projection = nn.Linear(2, d_model).to(device)
        
        # 多层解码器
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                dim_fc=dim_fc,
                dropout=dropout
            ).to(device) for _ in range(n_layers)
        ])
        
        # 修改输出投影层维度为2，因为输出特征也是2维的
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2)
        ).to(device)

    def forward(self, x, frequency_masked_x):
        """
        处理频域掩码数据
        
        Args:
            x: 原始输入数据 [B, T, D]
            frequency_masked_x: 频域掩码后的数据 [B, T, D]
            
        Returns:
            output: 解码后的数据 [B, T, D]
            attention_weights: 注意力权重列表
        """
        # 确保输入在正确的设备上
        x = x.to(self.device)
        frequency_masked_x = frequency_masked_x.to(self.device)
        
        # 输入投影
        x = self.input_projection(x)  # [B, T, d_model]
        frequency_masked_x = self.input_projection(frequency_masked_x)  # [B, T, d_model]
        
        # 存储所有层的注意力权重
        attention_weights = []
        
        # 通过多层解码器
        for layer in self.layers:
            x, attn_weights = layer(x, frequency_masked_x)
            attention_weights.append(attn_weights)
        
        # 输出投影
        output = self.output_projection(x)  # [B, T, 2]
        
        return output, attention_weights

class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_fc: int,
        dropout: float
    ):
        super(DecoderLayer, self).__init__()
        
        # 自注意力层
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 交叉注意力层
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_fc),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_fc, d_model)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, frequency_masked_x):
        """
        解码器层的前向传播
        
        Args:
            x: 原始输入 [B, T, D]
            frequency_masked_x: 频域掩码输入 [B, T, D]
            
        Returns:
            output: 处理后的数据 [B, T, D]
            attention_weights: 注意力权重
        """
        # 自注意力
        residual = x
        x = self.norm1(x)
        x, self_attn_weights = self.self_attention(x, x, x)
        x = self.dropout(x)
        x = residual + x
        
        # 交叉注意力
        residual = x
        x = self.norm2(x)
        x, cross_attn_weights = self.cross_attention(x, frequency_masked_x, frequency_masked_x)
        x = self.dropout(x)
        x = residual + x
        
        # 前馈网络
        residual = x
        x = self.norm3(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = residual + x
        
        return x, (self_attn_weights, cross_attn_weights)

def process_frequency_masked_data(masked_data, original_data, d_model=256, n_heads=8, dim_fc=128, device='cpu'):
    """
    处理频域掩码数据
    
    Args:
        masked_data: 频域掩码后的数据 [B, T, D]
        original_data: 原始数据 [B, T, D]
        d_model: 模型维度
        n_heads: 注意力头数
        dim_fc: 前馈网络维度
        device: 设备类型
        
    Returns:
        processed_data: 处理后的数据 [B, T, D]
        attention_weights: 注意力权重列表
    """
    print("\n=== 开始频域解码处理 ===")
    print(f"输入形状: {masked_data.shape}")
    print(f"参数设置: d_model={d_model}, n_heads={n_heads}, dim_fc={dim_fc}, device={device}")
    
    # 确保输入数据在CPU上
    masked_data = masked_data.cpu()
    original_data = original_data.cpu()
    
    # 创建解码器
    print("\n初始化频域解码器...")
    decoder = FrequencyDecoder(
        d_model=d_model,
        n_heads=n_heads,
        dim_fc=dim_fc,
        device=device
    ).to(device)
    
    # 处理数据
    print("\n开始处理数据...")
    with torch.no_grad():
        processed_data, attention_weights = decoder(original_data, masked_data)
        print(f"解码器输出形状: {processed_data.shape}")
        print(f"注意力权重数量: {len(attention_weights)}")
        for i, (self_attn, cross_attn) in enumerate(attention_weights):
            print(f"  第 {i+1} 层自注意力权重形状: {self_attn.shape}")
            print(f"  第 {i+1} 层交叉注意力权重形状: {cross_attn.shape}")
    
    print("\n=== 频域解码处理完成 ===")
    return processed_data, attention_weights

def load_frequency_masked_data(file_path: str) -> torch.Tensor:
    """
    加载频域掩码后的数据
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        加载的张量
    """
    return torch.load(file_path)

def main():
    # 设置参数
    d_model = 512
    nhead = 8
    num_decoder_layers = 6
    dim_feedforward = 2048
    dropout = 0.1
    
    # 创建模型
    model = FrequencyDecoder(
        d_model=d_model,
        n_heads=nhead,
        dim_fc=dim_feedforward,
        dropout=dropout
    )
    
    # 加载频域掩码后的数据
    masked_data = load_frequency_masked_data('path_to_your_masked_data.pt')
    original_data = load_frequency_masked_data('path_to_your_original_data.pt')
    
    # 前向传播
    output, attention_weights = model(original_data, masked_data)
    
    # 保存输出
    torch.save(output, 'decoder_output.pt')
    torch.save(attention_weights, 'attention_weights.pt')

if __name__ == '__main__':
    main() 