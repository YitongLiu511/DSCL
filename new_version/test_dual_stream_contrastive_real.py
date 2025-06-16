import torch
import numpy as np
from dual_stream_contrastive import DualStreamContrastiveLoss, compute_anomaly_score
from spatial_attention import SpatialAttentionBlock
from process_normal_data import MultipleGCN
import os
import torch.nn as nn

def test_dual_stream_contrastive_real():
    print("\n" + "="*80)
    print("【使用实际数据测试双流对比损失】")
    print("="*80 + "\n")
    
    # 1. 加载数据
    print("1. 加载数据")
    print("-"*50)
    
    # 加载时间注意力处理后的数据
    temporal_data = np.load('data/processed/temporal_attention_processed.npy')
    print(f"时间注意力数据形状: {temporal_data.shape}")
    
    # 加载邻接矩阵
    adj_distance = np.load('data/processed/dist.npy')
    adj_correlation = np.load('data/processed/adj.npy')
    adj_connectivity = np.load('data/processed/poi_sim.npy')
    
    # 将邻接矩阵转换为张量并堆叠
    adj_matrices = torch.stack([
        torch.FloatTensor(adj_distance),
        torch.FloatTensor(adj_correlation),
        torch.FloatTensor(adj_connectivity)
    ])
    print(f"邻接矩阵形状: {adj_matrices.shape}")
    
    # 2. 处理动态流（空间注意力）
    print("\n2. 处理动态流（空间注意力）")
    print("-"*50)
    
    # 初始化空间注意力处理器
    spatial_processor = SpatialAttentionBlock(
        in_channels=2,  # 输入特征维度是2
        out_channels=4,  # 输出特征维度是4，确保能被n_heads=4整除
        n_views=3,  # 使用3个视图（距离、相关性、POI相似度）
        n_heads=4,  # 4个注意力头
        dropout=0.1,
        gcn_type='gcn'  # 使用GCN
    )
    
    # 处理数据
    x = torch.tensor(temporal_data, dtype=torch.float32).permute(1, 0, 2)  # [2016, 263, 2]
    # 添加一个线性层来调整输入维度
    input_proj = nn.Linear(2, 4).to(x.device)
    x = input_proj(x)  # [2016, 263, 4]
    dynamic_features = spatial_processor(x, [adj_distance, adj_correlation, adj_connectivity])
    print(f"动态流特征形状: {dynamic_features.shape}")
    
    # 3. 处理静态流（多视图GCN）
    print("\n3. 处理静态流（多视图GCN）")
    print("-"*50)
    
    # 初始化GCN处理器
    gcn_processor = MultipleGCN(
        in_channels=4,  # 输入特征维度是4，与动态流保持一致
        out_channels=4,  # 输出特征维度也是4，保持一致性
        n_views=3,  # 使用3个视图
        dropout=0.1,
        gcn_type='gcn'  # 使用GCN
    )
    
    # 处理数据
    x = torch.tensor(temporal_data, dtype=torch.float32).permute(1, 0, 2)  # [2016, 263, 2]
    x = input_proj(x)  # [2016, 263, 4]
    static_features, _ = gcn_processor(x, [adj_distance, adj_correlation, adj_connectivity])
    print(f"静态流特征形状: {static_features.shape}")
    
    # 4. 计算双流对比损失
    print("\n4. 计算双流对比损失")
    print("-"*50)
    
    # 初始化损失模块
    loss_module = DualStreamContrastiveLoss(temperature=0.07)
    
    # 计算损失
    total_loss = loss_module(
        dynamic_features=dynamic_features,
        static_features=static_features
    )
    
    print(f"总损失: {total_loss.item():.4f}")
    
    # 5. 计算异常分数
    print("\n5. 计算异常分数")
    print("-"*50)
    
    anomaly_scores = loss_module.compute_anomaly_score(
        dynamic_features=dynamic_features,
        static_features=static_features
    )
    
    print(f"异常分数形状: {anomaly_scores.shape}")
    print(f"异常分数统计:")
    print(f"  最小值: {anomaly_scores.min().item():.4f}")
    print(f"  最大值: {anomaly_scores.max().item():.4f}")
    print(f"  平均值: {anomaly_scores.mean().item():.4f}")
    print(f"  标准差: {anomaly_scores.std().item():.4f}")
    
    # 6. 保存结果
    print("\n6. 保存结果")
    print("-"*50)
    
    # 创建保存目录
    os.makedirs('data/processed', exist_ok=True)
    
    # 保存特征和异常分数
    torch.save(dynamic_features, 'data/processed/dynamic_features.pt')
    torch.save(static_features, 'data/processed/static_features.pt')
    torch.save(anomaly_scores, 'data/processed/anomaly_scores.pt')
    
    print("结果已保存到 data/processed 目录")

if __name__ == "__main__":
    test_dual_stream_contrastive_real() 