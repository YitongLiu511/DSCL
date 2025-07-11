import torch
import torch.nn as nn
import math

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

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        d_ff = d_ff or d_model * 4
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x, memory):
        # x: [B, T, D], memory: [B, S, D]
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        cross_out, _ = self.cross_attn(x, memory, memory)
        x = self.norm2(x + cross_out)
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, d_model=256, n_heads=4, e_layers=3, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_model*4, dropout) for _ in range(e_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x, memory):
        # x: [B, T, D], memory: [B, S, D]
        for layer in self.layers:
            x = layer(x, memory)
        x = self.norm(x)
        return x

class TemporalDecoder(nn.Module):
    def __init__(self, input_dim=2, d_model=256, n_heads=4, e_layers=3, dropout=0.1, mask_token_init=None):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        
        # 输入投影层，将input_dim投影到d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        self.pos_emb = PositionalEmbedding(d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model) if mask_token_init is None else mask_token_init)
        self.decoder = TransformerDecoder(d_model, n_heads, e_layers, dropout)
        self.pro = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, input_dim),  # 输出投影回原始维度
            nn.Sigmoid()
        )
    def forward(self, normal_tokens, mask_indices, total_len):
        # normal_tokens: [B, N_unmask, input_dim] - 未掩码的观测数据
        # mask_indices: [B, N_mask] - 掩码位置索引
        # total_len: int - 完整序列长度
        B, N_unmask, input_dim = normal_tokens.shape
        N_mask = mask_indices.shape[1]
        device = normal_tokens.device
        
        # 输入投影
        normal_tokens_projected = self.input_projection(normal_tokens)  # [B, N_unmask, d_model]
        
        # 构造完整序列，所有位置初始化为掩码表示
        tokens = torch.zeros(B, total_len, self.d_model, device=device)
        
        # 为每个batch处理
        for b in range(B):
            # 获取掩码位置和未掩码位置
            mask_positions = mask_indices[b]  # [N_mask]
            mask_set = set(mask_positions.tolist())
            unmask_positions = [i for i in range(total_len) if i not in mask_set]
            
            # 1. 在掩码位置插入掩码表示（mask_token + position_embedding）
            if len(mask_positions) > 0:
                mask_tokens = self.mask_token.repeat(len(mask_positions), 1) + self.pos_emb(mask_positions)
                tokens[b, mask_positions, :] = mask_tokens
            
            # 2. 在未掩码位置插入真实观测数据
            for i, pos in enumerate(unmask_positions):
                if i < normal_tokens_projected[b].shape[0]:  # 确保不越界
                    tokens[b, pos, :] = normal_tokens_projected[b, i, :]
        
        # 解码（带交叉注意力）
        dec_out = self.decoder(tokens, normal_tokens_projected)  # memory=编码器输出
        rec = self.pro(dec_out)  # [B, T, input_dim]
        return rec 