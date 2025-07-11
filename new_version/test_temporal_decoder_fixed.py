import torch
import numpy as np
import sys
import os

# 添加路径
sys.path.append('/workspace/DSCL/DSCL-master')
sys.path.append('/workspace/DSCL/DSCL-master/new_version')

from temporal_decoder import TemporalDecoder

def test_temporal_decoder():
    print("测试修改后的时域解码器...")
    
    # 模型参数
    input_dim = 2
    d_model = 256
    n_heads = 8  # 确保能被d_model整除
    e_layers = 3
    dropout = 0.1
    
    # 创建解码器
    decoder = TemporalDecoder(
        input_dim=input_dim,
        d_model=d_model,
        n_heads=n_heads,
        e_layers=e_layers,
        dropout=dropout
    )
    
    # 测试数据
    batch_size = 2
    total_len = 100
    n_unmask = 60  # 未掩码观测数量
    n_mask = total_len - n_unmask  # 掩码位置数量
    
    # 模拟未掩码观测数据
    normal_tokens = torch.randn(batch_size, n_unmask, input_dim)
    
    # 模拟掩码位置索引（每个batch的掩码位置）
    mask_indices = torch.randint(0, total_len, (batch_size, n_mask))
    
    print(f"输入形状:")
    print(f"  normal_tokens: {normal_tokens.shape}")
    print(f"  mask_indices: {mask_indices.shape}")
    print(f"  total_len: {total_len}")
    
    # 前向传播
    try:
        output = decoder(normal_tokens, mask_indices, total_len)
        print(f"输出形状: {output.shape}")
        print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        # 验证输出维度
        expected_shape = (batch_size, total_len, input_dim)
        assert output.shape == expected_shape, f"输出形状错误: 期望 {expected_shape}, 得到 {output.shape}"
        
        print("✅ 时域解码器测试通过!")
        
        # 保存测试结果
        np.save('temporal_decoder_output.npy', output.detach().numpy())
        print("输出已保存到 temporal_decoder_output.npy")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_temporal_decoder() 