import torch
import numpy as np
import argparse
from data.load_nyc import load_dataset
from new_version.temporal_frequency_masking import TemporalFrequencyMasking
from new_version.temporal_attention import process_temporal_masked_data
from new_version.spatial_attention import process_spatial_attention
import os

def test_pipeline():
    print("=== 开始测试流程 ===\n")
    
    # 设置设备为CPU
    device = torch.device('cpu')
    print(f"使用设备: {device}\n")
    
    # 1. 加载数据并注入异常
    print("1. 加载数据并注入异常...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--inject_anomaly', action='store_true')
    parser.add_argument('--anomaly_ratio', type=float, default=0.1)
    parser.add_argument('--normalize', action='store_true')
    args = parser.parse_args()
    
    X, test_X, (adj, dist, poi_sim), y = load_dataset(args)
    print(f"数据加载完成，训练集形状: {X.shape}, 测试集形状: {test_X.shape}\n")
    
    # 2. 时频掩蔽处理
    print("2. 开始时频掩蔽处理...")
    masking_module = TemporalFrequencyMasking(
        window_size=10,
        temporal_mask_ratio=0.1,
        frequency_mask_ratio=0.1,
        d_model=263
    )
    
    # 转换为PyTorch张量
    X_tensor = torch.FloatTensor(X)
    test_X_tensor = torch.FloatTensor(test_X)
    
    # 处理训练集
    print("\n处理训练集...")
    temporal_masked_X, temporal_mask_indices = masking_module.temporal_masking(X_tensor)
    frequency_masked_X, frequency_mask_indices = masking_module.frequency_masking(X_tensor)
    print(f"训练集时间掩蔽位置数量: {temporal_mask_indices.shape[0] * temporal_mask_indices.shape[1]}")
    print(f"训练集频率掩蔽位置数量: {frequency_mask_indices.shape[0] * frequency_mask_indices.shape[1]}")
    
    # 处理测试集
    print("\n处理测试集...")
    temporal_masked_test_X, temporal_mask_indices_test = masking_module.temporal_masking(test_X_tensor)
    frequency_masked_test_X, frequency_mask_indices_test = masking_module.frequency_masking(test_X_tensor)
    print(f"测试集时间掩蔽位置数量: {temporal_mask_indices_test.shape[0] * temporal_mask_indices_test.shape[1]}")
    print(f"测试集频率掩蔽位置数量: {frequency_mask_indices_test.shape[0] * frequency_mask_indices_test.shape[1]}\n")
    
    # 3. 时间注意力处理
    print("3. 开始时间注意力处理...")
    
    # 处理训练集时间掩码数据
    print("\n处理训练集时间掩码数据...")
    processed_X, train_attention_weights = process_temporal_masked_data(
        temporal_masked_X,
        d_model=256,  # 使用能被8整除的维度
        nhead=8,      # 使用8个注意力头
        device='cpu'
    )
    print(f"训练集时间注意力处理完成，输出形状: {processed_X.shape}")
    print(f"注意力权重数量: {len(train_attention_weights)}")
    
    # 4. 空间注意力处理
    print("\n4. 开始空间注意力处理...")
    
    # 准备图结构数据
    adj_list = [adj, dist, poi_sim]  # 使用距离图和POI相似度图
    
    # 处理训练集空间注意力
    print("\n处理训练集空间注意力...")
    processed_X = process_spatial_attention(
        processed_X,  # [14, 144, 263]
        adj_list,
        d_model=263,  # 使用区域数量作为模型维度
        num_graphs=3,  # 使用3个图：邻接矩阵、距离图和POI相似度图
        device='cpu'
    )
    print(f"训练集空间注意力处理完成，输出形状: {processed_X.shape}")
    
    print("\n=== 时频掩蔽处理完成 ===")
    print(f"处理后的训练集形状: {processed_X.shape}")
    
    # 4. 保存处理后的数据
    print("\n4. 保存处理后的数据...")
    save_path = os.path.join('data', 'processed')
    os.makedirs(save_path, exist_ok=True)
    
    # 保存处理后的训练数据
    np.save(os.path.join(save_path, 'processed_X.npy'), processed_X.numpy())
    np.save(os.path.join(save_path, 'y.npy'), y.numpy())
    
    # 保存图结构数据
    np.save(os.path.join(save_path, 'adj.npy'), adj.numpy())
    np.save(os.path.join(save_path, 'dist.npy'), dist.numpy())
    np.save(os.path.join(save_path, 'poi_sim.npy'), poi_sim.numpy())
    
    print(f"数据已保存到: {save_path}")
    
    return processed_X, y, (adj, dist, poi_sim)

if __name__ == "__main__":
    test_pipeline()