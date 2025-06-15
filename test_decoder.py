import torch
import numpy as np
from new_version.frequency_decoder import process_frequency_masked_data

def test_frequency_decoder():
    print("=== 开始测试频域解码器 ===\n")
    
    # 加载数据
    print("1. 加载数据...")
    data_path = 'data/processed'
    
    # 加载原始训练集数据
    X = np.load(f'{data_path}/X_anomaly.npy')  # 训练集（有异常）
    print(f"原始训练集形状: {X.shape}")
    
    # 加载频域掩码后的数据
    frequency_masked_X = np.load(f'{data_path}/frequency_masked_X.npy')
    print(f"频域掩码数据形状: {frequency_masked_X.shape}")
    
    # 转换为PyTorch张量
    X_tensor = torch.FloatTensor(X)
    frequency_masked_X_tensor = torch.FloatTensor(frequency_masked_X)
    
    # 2. 处理数据
    print("\n2. 开始处理数据...")
    processed_data, attention_weights = process_frequency_masked_data(
        masked_data=frequency_masked_X_tensor,
        original_data=X_tensor,
        d_model=256,
        n_heads=8,
        dim_fc=128,
        device='cpu'
    )
    
    # 3. 保存处理后的数据
    print("\n3. 保存处理后的数据...")
    save_path = 'data/processed'
    
    # 保存处理后的数据
    np.save(f'{save_path}/frequency_decoded_X.npy', processed_data.numpy())
    
    # 保存注意力权重
    for i, (self_attn, cross_attn) in enumerate(attention_weights):
        np.save(f'{save_path}/frequency_decoder_self_attn_{i+1}.npy', self_attn.numpy())
        np.save(f'{save_path}/frequency_decoder_cross_attn_{i+1}.npy', cross_attn.numpy())
    
    print(f"\n所有数据已保存到 {save_path} 目录")
    print("保存的数据包括：")
    print("1. 频域解码后的数据：frequency_decoded_X.npy")
    print("2. 自注意力权重：frequency_decoder_self_attn_*.npy")
    print("3. 交叉注意力权重：frequency_decoder_cross_attn_*.npy")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_frequency_decoder()
