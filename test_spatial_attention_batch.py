import torch
import numpy as np
from new_version.spatial_attention import SpatialAttentionBlock

if __name__ == "__main__":
    print("开始批量空间注意力推理...")
    # 加载时间注意力处理后的数据
    data = np.load('data/temporal_attention_processed_train.npy')  # [263, 2016, 2]
    print(f"原始数据shape: {data.shape}")
    # 转换为 [B, N, C]，即 [2016, 263, 2]
    data = torch.FloatTensor(data).permute(1, 0, 2)  # [2016, 263, 2]
    print(f"转换后数据shape: {data.shape}")

    # 加载邻接矩阵
    adj = np.load('data/processed/adj.npy')
    adj_list = [torch.FloatTensor(adj)]  # 只用一个邻接矩阵
    print(f"邻接矩阵shape: {adj.shape}")

    # 创建空间注意力处理器
    n_heads = 2
    processor = SpatialAttentionBlock(
        in_channels=2,
        out_channels=2,
        n_views=1,
        n_heads=n_heads,
        dropout=0.1,
        gcn_type='gcn'
    )

    # 批量推理
    with torch.no_grad():
        output, scores = processor(data, adj_list)  # output: [2016, 263, 2], scores: [2016, n_heads, 263, 263]
    print(f"空间注意力输出shape: {output.shape}")
    print(f"空间注意力分数shape: {scores.shape}")

    # 保存分数
    np.save('data/spatial_attention_scores_train_batch.npy', scores.detach().numpy())
    print("已保存为 data/spatial_attention_scores_train_batch.npy") 