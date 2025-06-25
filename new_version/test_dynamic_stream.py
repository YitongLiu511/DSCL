import torch
import numpy as np
from spatial_attention import SpatialAttentionBlock

def test_dynamic_stream():
    print("\n" + "="*80)
    print("【测试动态流（空间注意力）返回(output, scores)】")
    print("="*80 + "\n")
    
    # 1. 准备测试数据
    print("1. 准备测试数据")
    print("-"*50)
    
    # 模拟时间注意力处理后的数据
    batch_size = 2
    num_nodes = 263
    feature_dim = 2
    
    # 创建测试输入
    x = torch.randn(batch_size, num_nodes, feature_dim)
    print(f"输入数据形状: {x.shape}")
    
    # 创建测试邻接矩阵
    adj_distance = torch.randn(num_nodes, num_nodes)
    adj_correlation = torch.randn(num_nodes, num_nodes)
    adj_connectivity = torch.randn(num_nodes, num_nodes)
    
    adj_list = [adj_distance, adj_correlation, adj_connectivity]
    print(f"邻接矩阵数量: {len(adj_list)}")
    
    # 2. 初始化动态流处理器
    print("\n2. 初始化动态流处理器")
    print("-"*50)
    
    dynamic_processor = SpatialAttentionBlock(
        in_channels=feature_dim,  # 输入特征维度是2
        out_channels=4,  # 输出特征维度是4，确保能被n_heads=4整除
        n_views=3,  # 使用3个视图
        n_heads=4,  # 4个注意力头
        dropout=0.1,
        gcn_type='gcn'
    )
    
    print(f"动态流处理器初始化完成")
    print(f"  输入通道数: {feature_dim}")
    print(f"  输出通道数: 4")
    print(f"  注意力头数: 4")
    print(f"  视图数量: 3")
    
    # 3. 测试动态流处理
    print("\n3. 测试动态流处理")
    print("-"*50)
    
    # 处理数据
    output, scores = dynamic_processor(x, adj_list)
    
    print(f"动态流输出形状: {output.shape}")
    print(f"动态流注意力分数形状: {scores.shape}")
    
    # 4. 验证输出格式
    print("\n4. 验证输出格式")
    print("-"*50)
    
    # 检查输出维度
    expected_output_shape = (batch_size, num_nodes, 4)
    expected_scores_shape = (batch_size, 4, num_nodes, num_nodes)
    
    print(f"期望输出形状: {expected_output_shape}")
    print(f"实际输出形状: {output.shape}")
    print(f"输出形状匹配: {output.shape == expected_output_shape}")
    
    print(f"期望注意力分数形状: {expected_scores_shape}")
    print(f"实际注意力分数形状: {scores.shape}")
    print(f"注意力分数形状匹配: {scores.shape == expected_scores_shape}")
    
    # 5. 验证注意力分数性质
    print("\n5. 验证注意力分数性质")
    print("-"*50)
    
    # 检查注意力分数是否在[0,1]范围内
    scores_min = scores.min().item()
    scores_max = scores.max().item()
    scores_sum = scores.sum(dim=-1).mean().item()  # 每行的和应该接近1
    
    print(f"注意力分数最小值: {scores_min:.4f}")
    print(f"注意力分数最大值: {scores_max:.4f}")
    print(f"注意力分数行和平均值: {scores_sum:.4f}")
    print(f"注意力分数是否合理: {0 <= scores_min <= scores_max <= 1}")
    print(f"注意力分数行和是否接近1: {abs(scores_sum - 1) < 0.1}")
    
    # 6. 总结
    print("\n6. 测试总结")
    print("-"*50)
    
    success = (
        output.shape == expected_output_shape and
        scores.shape == expected_scores_shape and
        0 <= scores_min <= scores_max <= 1 and
        abs(scores_sum - 1) < 0.1
    )
    
    if success:
        print("✅ 动态流测试成功！")
        print("   - 正确返回(output, scores)格式")
        print("   - 输出维度符合预期")
        print("   - 注意力分数性质正确")
    else:
        print("❌ 动态流测试失败！")
        print("   - 请检查输出格式和维度")
    
    return output, scores

if __name__ == "__main__":
    test_dynamic_stream() 