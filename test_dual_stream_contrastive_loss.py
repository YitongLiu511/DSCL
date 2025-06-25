import numpy as np
import torch
from new_version.dual_stream_contrastive import DualStreamContrastiveLoss
import torch.nn.functional as F

if __name__ == "__main__":
    print("加载动态流分数（空间注意力分数）...")
    score_dy = np.load('data/spatial_attention_scores_train_batch.npy')  # [B, n_heads, N, N]
    print(f"score_dy shape: {score_dy.shape}")

    print("加载静态流分数...")
    score_st = np.load('data/processed/static_scores.npy')  # [N, N]
    print(f"score_st shape: {score_st.shape}")

    # 转为torch张量
    score_dy = torch.FloatTensor(score_dy)  # [B, n_heads, N, N]
    score_st = torch.FloatTensor(score_st)  # [N, N]

    # 自动归一化，确保为概率分布
    score_dy = F.softmax(score_dy, dim=-1)
    score_st = F.softmax(score_st, dim=-1)

    # 扩展静态分数到batch和head维度（与动态分数对齐）
    B, n_heads, N, _ = score_dy.shape
    score_st_expand = score_st.unsqueeze(0).unsqueeze(0).expand(B, n_heads, N, N)  # [B, n_heads, N, N]

    # 初始化损失
    criterion = DualStreamContrastiveLoss()
    print("计算双流对比损失...")
    loss = criterion(score_dy, score_st_expand)
    loss_scalar = loss.mean()
    print(f"对比损失标量: {loss_scalar.item():.6f}")
    print(f"对比损失 shape: {loss.shape}") 