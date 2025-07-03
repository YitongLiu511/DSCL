import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SpatialSelfAttentionBlock(nn.Module):
    """
    空间自注意力模块，用于计算节点间的依赖关系。
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(SpatialSelfAttentionBlock, self).__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [B, N, D_model]
        B, N, _ = x.shape
        # print("[SpatialSelfAttentionBlock] input x mean/std:", x.mean().item(), x.std().item())
        q = self.q_linear(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_linear(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_linear(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        # 放大q/k
        q = q * 5
        k = k * 5
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        # print("[SpatialSelfAttentionBlock] scores mean/std/max/min:", scores.mean().item(), scores.std().item(), scores.max().item(), scores.min().item())
        attn_weights = F.softmax(scores, dim=-1)
        # print("[SpatialSelfAttentionBlock] attn_weights mean/std/max/min:", attn_weights.mean().item(), attn_weights.std().item(), attn_weights.max().item(), attn_weights.min().item())
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, N, -1)
        output = self.out_proj(context)
        return output, attn_weights 