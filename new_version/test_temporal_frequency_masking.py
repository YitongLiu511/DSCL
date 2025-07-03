import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
from new_version.temporal_frequency_masking import TemporalFrequencyMasking

if __name__ == "__main__":
    # 直接加载数据，保持 (N, T, F) 格式
    data = np.load('data/datanew/anomaly_injected_train.npy')  # (263, 2016, 2)
    data_tensor = torch.FloatTensor(data)

    # 初始化掩码模块
    masking_module = TemporalFrequencyMasking(
        window_size=10,
        temporal_mask_ratio=0.1,
        frequency_mask_ratio=0.1,
        d_model=2,      # embedding维度
        n_features=2,    # 这里n_features要和输入特征数一致
        device='cpu'     # 强制用CPU，避免显存不足
    )

    # 时域掩码
    temporal_masked, temporal_mask_indices = masking_module.temporal_masking(data_tensor)
    print("时域掩码输出 shape:", temporal_masked.shape)
    print("时域掩码索引 shape:", temporal_mask_indices.shape)

    # 频域掩码
    frequency_masked, frequency_mask = masking_module.frequency_masking(data_tensor)
    print("频域掩码输出 shape:", frequency_masked.shape)
    print("频域掩码mask shape:", frequency_mask.shape)

    # 保存到data/datanew目录
    save_dir = 'data/datanew'
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'temporal_masked.npy'), temporal_masked.detach().cpu().numpy())
    np.save(os.path.join(save_dir, 'temporal_mask_indices.npy'), temporal_mask_indices.detach().cpu().numpy())
    np.save(os.path.join(save_dir, 'frequency_masked.npy'), frequency_masked.detach().cpu().numpy())
    np.save(os.path.join(save_dir, 'frequency_mask.npy'), frequency_mask.detach().cpu().numpy())
    print('已保存所有掩码结果到data/datanew目录') 