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
        self.norm_fact = 1 / math.sqrt(d_model)

    def attention(self, Q, K, V, chunk_size=16):
        B, L = Q.shape[:2]
        outputs = []
        scores_list = []
        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            Q_chunk = Q[:, start:end]  # (B, chunk, H, E)
            # K, V shape: (B, L, H, E)
            # einsum: (B, chunk, H, E) x (B, L, H, E) -> (B, H, chunk, L)
            scores = torch.einsum("blhe,bshe->bhls", Q_chunk, K) * self.norm_fact
            scores = scores.softmax(dim=-1)
            output = torch.einsum("bhls,bshd->blhd", scores, V).reshape(B, end-start, -1)
            outputs.append(output)
            scores_list.append(scores.cpu())  # 先转CPU，降低GPU显存占用
        output = torch.cat(outputs, dim=1)
        scores = torch.cat(scores_list, dim=2)  # dim=2是chunk维
        # 不再转回GPU，scores全程留在CPU，避免显存溢出
        return output, scores

    def forward(self, x, y):
        '''x : (B, L, D)'''
        B, L, _ = x.shape
        Q = self.q(x).reshape(B, L, self.n_heads, -1)  # (N, B, L, K)
        K = self.k(x).reshape(B, L, self.n_heads, -1)  # (N, B, L, K)
        V = self.v(y).reshape(B, L, self.n_heads, -1)  # (N, B, L, K)
        output, scores = self.attention(Q, K, V, chunk_size=16)
        return self.o(output), scores
