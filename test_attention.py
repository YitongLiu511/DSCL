import numpy as np
import torch
from new_version.temporal_attention import process_temporal_masked_data
from new_version.spatial_attention import process_spatial_attention

def test_attention():
    print("=== 开始测试注意力模块 ===\n")
    
    # 1. 加载预处理数据
    print("1. 加载预处理数据...")
    processed_X = np.load('data/processed/processed_X.npy')
    processed_X_tensor = torch.FloatTensor(processed_X)
    print(f"加载的数据形状: {processed_X_tensor.shape}\n")
    
    # 2. 加载图数据
    print("2. 加载图数据...")
    adj = np.load('data/processed/adj.npy')
    dist = np.load('data/processed/dist.npy')
    poi_sim = np.load('data/processed/poi_sim.npy')
    adj_list = [torch.FloatTensor(adj), torch.FloatTensor(dist), torch.FloatTensor(poi_sim)]
    print(f"邻接矩阵形状: {adj.shape}")
    print(f"距离矩阵形状: {dist.shape}")
    print(f"POI相似度矩阵形状: {poi_sim.shape}\n")
    
    # 3. 测试时间注意力
    print("3. 测试时间注意力...")
    temporal_output, temporal_weights = process_temporal_masked_data(
        processed_X_tensor,
        d_model=256,    # 模型维度
        dim_k=32,       # 键的维度
        dim_v=32,       # 值的维度
        nhead=8,        # 注意力头数
        dim_fc=128,     # 前馈网络维度
        device='cpu'
    )
    print(f"时间注意力输出形状: {temporal_output.shape}\n")
    
    # 4. 测试空间注意力
    print("4. 测试空间注意力...")
    spatial_output = process_spatial_attention(
        temporal_output,
        adj_list,
        d_model=256,
        num_graphs=3,
        device='cpu'
    )
    print(f"空间注意力输出形状: {spatial_output.shape}\n")
    
    print("=== 注意力模块测试完成 ===")
    return temporal_output, spatial_output

if __name__ == "__main__":
    test_attention() 