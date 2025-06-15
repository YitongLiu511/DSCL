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

    # 加载数据，只对训练集注入异常
    X, test_X, (adj, dist, poi_sim), y = load_dataset(args)
    print(f"数据加载完成，训练集形状: {X.shape}, 测试集形状: {test_X.shape}\n")
    
    # 保存原始数据（训练集注入异常前和测试集）
    print("\n保存原始数据...")
    save_path = os.path.join('data', 'processed')
    os.makedirs(save_path, exist_ok=True)
    
    # 保存训练集注入异常前的数据
    np.save(os.path.join(save_path, 'X_normal.npy'), X)  # 训练集（注入异常前）
    np.save(os.path.join(save_path, 'test_X.npy'), test_X)  # 测试集
    
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
    
    # 处理测试集（不注入异常，只进行掩码）
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
    
    # 准备图结构数据并转换为 PyTorch 张量
    adj_list = [
        torch.FloatTensor(adj),
        torch.FloatTensor(dist),
        torch.FloatTensor(poi_sim)
    ]  # 使用距离图和POI相似度图
    
    # 处理训练集空间注意力
    print("\n处理训练集空间注意力...")
    processed_X = process_spatial_attention(
        processed_X,  # [14, 144, 263]
        adj_list,
        d_model=256,  # 改为256，能被8整除
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
    
    # 保存原始数据（训练集有异常注入，测试集没有）
    print("\n保存原始数据...")
    np.save(os.path.join(save_path, 'X_anomaly.npy'), X)  # 训练集（有异常）
    np.save(os.path.join(save_path, 'test_X.npy'), test_X)  # 测试集（无异常）
    
    # 保存时域掩码数据
    print("\n保存时域掩码数据...")
    np.save(os.path.join(save_path, 'temporal_masked_X.npy'), temporal_masked_X.detach().cpu().numpy())
    np.save(os.path.join(save_path, 'temporal_masked_test_X.npy'), temporal_masked_test_X.detach().cpu().numpy())
    np.save(os.path.join(save_path, 'temporal_mask_indices.npy'), temporal_mask_indices.detach().cpu().numpy())
    np.save(os.path.join(save_path, 'temporal_mask_indices_test.npy'), temporal_mask_indices_test.detach().cpu().numpy())
    
    # 保存频域掩码数据
    print("\n保存频域掩码数据...")
    np.save(os.path.join(save_path, 'frequency_masked_X.npy'), frequency_masked_X.detach().cpu().numpy())
    np.save(os.path.join(save_path, 'frequency_masked_test_X.npy'), frequency_masked_test_X.detach().cpu().numpy())
    np.save(os.path.join(save_path, 'frequency_mask_indices.npy'), frequency_mask_indices.detach().cpu().numpy())
    np.save(os.path.join(save_path, 'frequency_mask_indices_test.npy'), frequency_mask_indices_test.detach().cpu().numpy())
    
    # 保存处理后的训练数据
    np.save(os.path.join(save_path, 'processed_X.npy'), processed_X.detach().cpu().numpy())
    
    # 如果 y 不为 None，则保存标签
    if y is not None:
    np.save(os.path.join(save_path, 'y.npy'), y.numpy())
    
    # 保存图数据
    np.save(os.path.join(save_path, 'adj.npy'), adj)
    np.save(os.path.join(save_path, 'dist.npy'), dist)
    np.save(os.path.join(save_path, 'poi_sim.npy'), poi_sim)
    
    print(f"\n所有数据已保存到 {save_path} 目录")
    print("保存的数据包括：")
    print("1. 原始数据：")
    print("   - X_normal.npy（训练集，注入异常前）")
    print("   - X_anomaly.npy（训练集，注入异常后）")
    print("   - test_X.npy（测试集，无异常）")
    print("2. 时域掩码数据：temporal_masked_X.npy, temporal_masked_test_X.npy")
    print("3. 时域掩码索引：temporal_mask_indices.npy, temporal_mask_indices_test.npy")
    print("4. 频域掩码数据：frequency_masked_X.npy, frequency_masked_test_X.npy")
    print("5. 频域掩码索引：frequency_mask_indices.npy, frequency_mask_indices_test.npy")
    print("6. 处理后的数据：processed_X.npy")
    print("7. 标签数据：y.npy（如果存在）")
    print("8. 图数据：adj.npy, dist.npy, poi_sim.npy")
    
    return processed_X, y, (adj, dist, poi_sim)

if __name__ == "__main__":
    test_pipeline()