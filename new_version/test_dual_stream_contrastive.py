import torch
import numpy as np
import torch.nn.functional as F
from dual_stream_contrastive import DualStreamContrastiveLoss, compute_anomaly_score

def test_dual_stream_contrastive():
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置参数
    batch_size = 32
    feature_dim = 128
    n_nodes = 263  # 与原始模型相同的节点数
    
    print("\n" + "="*80)
    print("【测试双流对比损失】")
    print("="*80)
    
    # 生成测试数据
    print("\n1. 生成测试数据")
    print("-"*50)
    
    # 生成动态流特征
    dynamic_features = torch.randn(batch_size, n_nodes, feature_dim)
    print(f"动态流特征维度: {dynamic_features.shape}")
    
    # 生成静态流特征
    static_features = torch.randn(batch_size, n_nodes, feature_dim)
    print(f"静态流特征维度: {static_features.shape}")
    
    # 生成注意力分数
    dynamic_attn = torch.randn(batch_size, n_nodes, n_nodes)
    dynamic_attn = F.softmax(dynamic_attn, dim=-1)  # 归一化注意力分数
    print(f"动态流注意力分数维度: {dynamic_attn.shape}")
    
    static_attn = torch.randn(batch_size, n_nodes, n_nodes)
    static_attn = F.softmax(static_attn, dim=-1)  # 归一化注意力分数
    print(f"静态流注意力分数维度: {static_attn.shape}")
    
    # 初始化损失模块
    print("\n2. 初始化损失模块")
    print("-"*50)
    loss_module = DualStreamContrastiveLoss(temperature=0.07)
    print(f"温度参数: {loss_module.temperature}")
    
    # 计算损失
    print("\n3. 计算损失")
    print("-"*50)
    total_loss, feature_loss, attn_loss = loss_module(
        dynamic_features, 
        static_features,
        dynamic_attn,
        static_attn
    )
    
    print(f"总损失: {total_loss.item():.4f}")
    print(f"特征对比损失: {feature_loss.item():.4f}")
    print(f"注意力对比损失: {attn_loss.item():.4f}")
    
    # 计算异常分数
    print("\n4. 计算异常分数")
    print("-"*50)
    anomaly_score = compute_anomaly_score(
        dynamic_features,
        static_features,
        dynamic_attn,
        static_attn
    )
    print(f"异常分数维度: {anomaly_score.shape}")
    print(f"异常分数统计:")
    print(f"- 最小值: {anomaly_score.min().item():.4f}")
    print(f"- 最大值: {anomaly_score.max().item():.4f}")
    print(f"- 平均值: {anomaly_score.mean().item():.4f}")
    print(f"- 标准差: {anomaly_score.std().item():.4f}")
    
    # 测试边界情况
    print("\n5. 测试边界情况")
    print("-"*50)
    
    # 测试1: 完全相同的特征
    print("\n测试1: 完全相同的特征")
    same_features = dynamic_features.clone()
    same_attn = dynamic_attn.clone()
    total_loss, feature_loss, attn_loss = loss_module(
        same_features,
        same_features,
        same_attn,
        same_attn
    )
    print(f"总损失: {total_loss.item():.4f}")
    
    # 测试2: 完全相反的特征
    print("\n测试2: 完全相反的特征")
    opposite_features = -dynamic_features
    opposite_attn = 1 - dynamic_attn
    total_loss, feature_loss, attn_loss = loss_module(
        dynamic_features,
        opposite_features,
        dynamic_attn,
        opposite_attn
    )
    print(f"总损失: {total_loss.item():.4f}")
    
    # 测试3: 包含NaN的特征
    print("\n测试3: 包含NaN的特征")
    nan_features = dynamic_features.clone()
    nan_features[0, 0, 0] = float('nan')
    total_loss, feature_loss, attn_loss = loss_module(
        nan_features,
        static_features,
        dynamic_attn,
        static_attn
    )
    print(f"总损失: {total_loss.item():.4f}")
    
    print("\n" + "="*80)
    print("【测试完成】")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_dual_stream_contrastive() 