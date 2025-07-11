import torch
import numpy as np
from temporal_decoder import TemporalDecoder
import os

def test_temporal_decoder():
    print("=== 测试 Temporal Decoder ===")
    
    # 1. 检查空间注意力特征是否存在
    spatial_features_path = '../spatial_attention_features_transposed.npy'
    if not os.path.exists(spatial_features_path):
        print(f"错误：未找到空间注意力特征 {spatial_features_path}")
        print("请先运行 test_spatial_attention1.py 生成空间注意力特征")
        return
    
    # 2. 加载空间注意力特征
    print("\n1. 加载空间注意力特征...")
    spatial_features = np.load(spatial_features_path)
    print(f"空间注意力特征 shape: {spatial_features.shape}")
    
    # 3. 加载掩码索引
    print("\n2. 加载掩码索引...")
    mask_indices = np.load('../data/datanew/temporal_mask_indices.npy')
    print(f"掩码索引 shape: {mask_indices.shape}")
    
    # 4. 准备解码器输入
    print("\n3. 准备解码器输入...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 转换为torch张量
    spatial_features_tensor = torch.from_numpy(spatial_features).float().to(device)
    mask_indices_tensor = torch.from_numpy(mask_indices).long().to(device)
    
    # 获取序列总长度 - 应该是原始数据的完整长度
    # 从原始掩码数据推断完整序列长度
    masked_data = np.load('../data/datanew/temporal_masked.npy')
    if masked_data.ndim == 3:
        total_len = masked_data.shape[1]  # [num_nodes, num_time, num_features]
    elif masked_data.ndim == 4:
        total_len = masked_data.shape[0] * masked_data.shape[1]  # [天, 槽, 节点, 特征]
    else:
        raise ValueError("masked_data shape 不支持")
    
    print(f"原始序列总长度: {total_len}")
    print(f"未掩码时间步数: {spatial_features_tensor.shape[1]}")
    print(f"掩码时间步数: {mask_indices_tensor.shape[1]}")
    print(f"验证: {spatial_features_tensor.shape[1]} + {mask_indices_tensor.shape[1]} = {spatial_features_tensor.shape[1] + mask_indices_tensor.shape[1]}")
    
    # 5. 初始化解码器
    print("\n4. 初始化解码器...")
    input_dim = 2  # 空间注意力输出维度
    d_model = 256  # 内部模型维度
    decoder = TemporalDecoder(
        input_dim=input_dim,
        d_model=d_model,
        n_heads=8,
        e_layers=3,
        dropout=0.1
    ).to(device)
    
    # 6. 测试解码器
    print("\n5. 测试解码器...")
    num_nodes = spatial_features_tensor.shape[0]
    all_outputs = []
    
    for node_idx in range(num_nodes):
        if node_idx % 10 == 0:
            print(f"处理节点 {node_idx}/{num_nodes}...")
        
        # 获取该节点的特征和掩码索引
        node_features = spatial_features_tensor[node_idx]  # [时间步, 特征数] - 已经是未掩码的特征
        node_mask_indices = mask_indices_tensor[node_idx]  # [掩码数]
        
        # 直接使用空间注意力特征作为normal_tokens（它们已经是未掩码的数据）
        normal_tokens = node_features.unsqueeze(0)  # [1, 未掩码数, 特征数]
        mask_indices = node_mask_indices.unsqueeze(0)  # [1, 掩码数]
        
        print(f"  节点{node_idx}: normal_tokens shape: {normal_tokens.shape}, mask_indices shape: {mask_indices.shape}")
        
        # 前向传播
        with torch.no_grad():
            output = decoder(
                normal_tokens=normal_tokens,
                mask_indices=mask_indices,
                total_len=total_len
            )
        
        print(f"  节点{node_idx}: 输出 shape: {output.shape}")
        all_outputs.append(output.squeeze(0).cpu().numpy())  # 去掉batch维度
        
        # 每处理5个节点清理一次显存
        if (node_idx + 1) % 5 == 0:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 7. 合并所有输出
    print("\n6. 合并输出结果...")
    final_output = np.concatenate(all_outputs, axis=0)
    print(f"最终输出 shape: {final_output.shape}")
    
    # 8. 保存结果
    print("\n7. 保存结果...")
    np.save('temporal_decoder_output.npy', final_output)
    print("解码器输出已保存到 temporal_decoder_output.npy")
    
    print("\n=== 测试完成 ===")

if __name__ == '__main__':
    test_temporal_decoder() 