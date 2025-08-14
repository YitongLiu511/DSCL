import torch
from torch import nn
import math


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
        # 注意力缩放应使用每个head的键维度 dim_k，而非整体 d_model
        # 使用 d_model 会造成缩放过强，softmax 退化为接近均匀分布
        self.norm_fact = 1 / math.sqrt(self.dim_k)

    def attention(self, Q, K, V, chunk_size=8, k_block_size=64):
        B, L = Q.shape[:2]
        H = Q.shape[2]
        outputs = []
        scores_blocks_cpu = []
        for q_start in range(0, L, chunk_size):
            q_end = min(q_start + chunk_size, L)
            Q_chunk = Q[:, q_start:q_end]  # (B, q_chunk, H, E)

            # 第一遍：跨K块计算行最大，做稳定softmax
            row_max = None  # (B, H, q_chunk)
            for k_start in range(0, L, k_block_size):
                k_end = min(k_start + k_block_size, L)
                K_blk = K[:, k_start:k_end]           # (B, k_blk, H, E)
                scores_blk = torch.einsum("blhe,bshe->bhls", Q_chunk, K_blk) * self.norm_fact  # (B,H,q,k_blk)
                blk_max = scores_blk.max(dim=-1).values  # (B,H,q)
                row_max = blk_max if row_max is None else torch.maximum(row_max, blk_max)

            # 第二遍：累加exp与加权V
            row_sumexp = torch.zeros_like(row_max)                   # (B,H,q)
            weighted_sum = torch.zeros(B, H, q_end - q_start, V.shape[-1], device=V.device)  # (B,H,q,Dv)
            row_blocks = []  # 可选：为了返回注意力，把每块softmax(score)存CPU
            for k_start in range(0, L, k_block_size):
                k_end = min(k_start + k_block_size, L)
                K_blk = K[:, k_start:k_end]
                V_blk = V[:, k_start:k_end]
                scores_blk = torch.einsum("blhe,bshe->bhls", Q_chunk, K_blk) * self.norm_fact  # (B,H,q,k_blk)
                exp_blk = torch.exp(scores_blk - row_max.unsqueeze(-1))  # (B,H,q,k_blk)
                row_sumexp = row_sumexp + exp_blk.sum(dim=-1)
                # 贡献到输出
                weighted_sum = weighted_sum + torch.einsum("bhls,bshd->bhld", exp_blk, V_blk)
                # 收集块softmax到CPU，避免一次性在GPU保存完整注意力
                row_blocks.append(exp_blk.detach().cpu())

            # 得到归一softmax输出
            output_chunk = (weighted_sum / (row_sumexp.unsqueeze(-1) + 1e-12)).transpose(1, 2).reshape(B, q_end - q_start, -1)
            outputs.append(output_chunk)

            # 将块拼接，构成该Q段的完整注意力（在CPU）
            # 注意：该张量维度为 (B,H,q_chunk,L)
            scores_q_cpu = []
            for i, k_start in enumerate(range(0, L, k_block_size)):
                k_end = min(k_start + k_block_size, L)
                # 归一化：exp / sumexp
                scores_q_cpu.append((row_blocks[i] / (row_sumexp.unsqueeze(-1).detach().cpu() + 1e-12)))
            scores_q_cpu = torch.cat(scores_q_cpu, dim=-1)
            scores_blocks_cpu.append(scores_q_cpu)

        output = torch.cat(outputs, dim=1)
        scores = torch.cat(scores_blocks_cpu, dim=2)  # 拼接q_chunk维
        return output, scores

    def forward(self, x, y):
        '''x : (B, L, D)'''
        B, L, _ = x.shape
        Q = self.q(x).reshape(B, L, self.n_heads, -1)  # (N, B, L, K)
        K = self.k(x).reshape(B, L, self.n_heads, -1)  # (N, B, L, K)
        V = self.v(y).reshape(B, L, self.n_heads, -1)  # (N, B, L, K)
        # 一次性打印缩放与维度信息，便于数值诊断（仅首次）
        if not hasattr(self, "_printed_scale_info"):
            try:
                print(f"🔧 Attention scale: 1/sqrt(dim_k) = {self.norm_fact:.6f}, dim_k={self.dim_k}, n_heads={self.n_heads}, L={L}")
            except Exception:
                pass
            self._printed_scale_info = True
        output, scores = self.attention(Q, K, V, chunk_size=8, k_block_size=64)
        return self.o(output), scores
