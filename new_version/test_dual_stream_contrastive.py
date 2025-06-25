import numpy as np
import torch
from dual_stream_contrastive import sym_kl_loss, DualStreamContrastiveLoss, compute_anomaly_score

if __name__ == "__main__":
    print("=== 使用真实数据测试对称KL、双流对比损失、异常分数 ===\n")
    # 使用绝对路径加载真实数据
    try:
        dynamic_scores = np.load('C:/Users/86155/Downloads/DSCLBEW/data/processed/dynamic_scores.npy')
        static_scores = np.load('C:/Users/86155/Downloads/DSCLBEW/data/processed/static_scores.npy')
        print(f"动态流分数形状: {dynamic_scores.shape}")
        print(f"静态流分数形状: {static_scores.shape}")
        # 转为tensor
        score_dy = torch.from_numpy(dynamic_scores).float()
        score_st = torch.from_numpy(static_scores).float()
        # 归一化
        if score_dy.dim() == 4:  # [B, n_heads, N, N]
            score_dy = torch.softmax(score_dy, dim=-1)
        else:  # [N, N]
            score_dy = torch.softmax(score_dy, dim=-1)
        score_st = torch.softmax(score_st, dim=0)
        # 1. 对称KL
        sym_kl = sym_kl_loss(score_dy, score_st)
        print(f"对称KL散度: {sym_kl}")
        # 2. 双流对比损失
        contrastive_loss = DualStreamContrastiveLoss(use_diff_const=False)
        loss = contrastive_loss(score_dy, score_st)
        print(f"双流对比损失: {loss}")
        # 3. 异常分数
        anomaly_score = compute_anomaly_score(score_dy, score_st)
        print(f"异常分数: {anomaly_score}")
        print("\n✓ 真实数据测试完成")
    except FileNotFoundError as e:
        print(f"⚠ 文件不存在: {e}") 