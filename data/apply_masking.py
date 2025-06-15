import numpy as np
import torch
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from new_version.temporal_frequency_masking import TemporalFrequencyMasking

def apply_masking():
    print("=== 开始加载数据 ===")
    # 加载数据
    train_data = np.load('data/anomaly_injected_train.npy')
    test_data = np.load('data/normalized_test.npy')
    
    print(f"训练集形状: {train_data.shape}")
    print(f"测试集形状: {test_data.shape}")
    
    # 转换为PyTorch张量
    train_tensor = torch.FloatTensor(train_data)
    test_tensor = torch.FloatTensor(test_data)
    
    print("\n=== 创建掩码模块 ===")
    # 创建掩码模块
    masking_module = TemporalFrequencyMasking(
        window_size=10,
        temporal_mask_ratio=0.1,
        frequency_mask_ratio=0.1,
        d_model=263,
        n_features=2  # 明确指定特征数量
    )
    
    print("\n=== 应用时间掩码 ===")
    # 对训练集应用时间掩码
    temporal_masked_train, temporal_mask_indices = masking_module.temporal_masking(train_tensor)
    print(f"训练集时间掩码位置数量: {temporal_mask_indices.shape[0] * temporal_mask_indices.shape[1]}")
    
    # 对测试集应用时间掩码
    temporal_masked_test, temporal_mask_indices_test = masking_module.temporal_masking(test_tensor)
    print(f"测试集时间掩码位置数量: {temporal_mask_indices_test.shape[0] * temporal_mask_indices_test.shape[1]}")
    
    print("\n=== 应用频率掩码 ===")
    # 对训练集应用频率掩码
    frequency_masked_train, frequency_mask_indices = masking_module.frequency_masking(train_tensor)
    print(f"训练集频率掩码位置数量: {frequency_mask_indices.shape[0] * frequency_mask_indices.shape[1]}")
    
    # 对测试集应用频率掩码
    frequency_masked_test, frequency_mask_indices_test = masking_module.frequency_masking(test_tensor)
    print(f"测试集频率掩码位置数量: {frequency_mask_indices_test.shape[0] * frequency_mask_indices_test.shape[1]}")
    
    print("\n=== 保存处理后的数据 ===")
    # 将处理后的数据转换回numpy数组并保存
    np.save('data/temporal_masked_train.npy', temporal_masked_train.cpu().detach().numpy())
    np.save('data/temporal_masked_test.npy', temporal_masked_test.cpu().detach().numpy())
    np.save('data/frequency_masked_train.npy', frequency_masked_train.cpu().detach().numpy())
    np.save('data/frequency_masked_test.npy', frequency_masked_test.cpu().detach().numpy())
    
    # 保存掩码位置
    np.save('data/temporal_mask_indices.npy', temporal_mask_indices.cpu().detach().numpy())
    np.save('data/temporal_mask_indices_test.npy', temporal_mask_indices_test.cpu().detach().numpy())
    np.save('data/frequency_mask_indices.npy', frequency_mask_indices.cpu().detach().numpy())
    np.save('data/frequency_mask_indices_test.npy', frequency_mask_indices_test.cpu().detach().numpy())
    
    print("=== 处理完成 ===")
    print(f"时间掩码后的训练集形状: {temporal_masked_train.shape}")
    print(f"时间掩码后的测试集形状: {temporal_masked_test.shape}")
    print(f"频率掩码后的训练集形状: {frequency_masked_train.shape}")
    print(f"频率掩码后的测试集形状: {frequency_masked_test.shape}")

if __name__ == "__main__":
    apply_masking() 