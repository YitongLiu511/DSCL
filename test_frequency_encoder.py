import numpy as np
import torch
import os
import sys

# 将项目根目录添加到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from new_version.temporal_frequency_masking import TemporalFrequencyMasking
from new_version.frequency_decoder import FrequencyEncoder # 导入新的编码器

def run_encoder_test():
    """
    测试从TFM到FrequencyEncoder的完整流程。
    """
    print("==============================================")
    print("=  开始测试：TFM -> FrequencyEncoder 流程  =")
    print("==============================================\n")

    # --- 1. 定义文件路径和参数 ---
    print("--- 步骤 1: 定义文件路径和参数 ---")
    input_data_path = 'data/processed/train_data_with_anomalies_3d.npy'
    if not os.path.exists(input_data_path):
        print(f"错误: 输入文件 '{input_data_path}' 不存在!")
        return
    print(f"输入数据文件: {input_data_path}\n")

    # --- 2. 加载并重塑数据 ---
    print("--- 步骤 2: 加载并重塑数据 ---")
    data_3d = np.load(input_data_path)
    num_days, slots_per_day, n_zones, n_features = 14, 144, 263, 2
    data_4d = data_3d.reshape(num_days, slots_per_day, n_zones, n_features)
    data_tensor = torch.from_numpy(data_4d).float()
    print(f"数据已加载并重塑为: {data_tensor.shape}\n")

    # --- 3. 运行TFM模块以获取"提纯后"的数据 ---
    print("--- 步骤 3: 运行TFM模块 (掩盖最低幅度频率) ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tfm_module = TemporalFrequencyMasking(
        window_size=10,
        d_model=n_zones,
        n_features=n_features,
        device=device
    )
    # 我们只需要频率掩码后的数据作为下一阶段的输入
    _, _, frequency_masked_x, _ = tfm_module(data_tensor)
    print(f"TFM模块输出的频率掩码数据形状: {frequency_masked_x.shape}\n")
    
    # 注意：TFM模块已经自动保存了它的输出，我们这里只取内存中的变量继续

    # --- 4. 运行新的FrequencyEncoder ---
    print("--- 步骤 4: 将提纯后数据送入 FrequencyEncoder ---")
    
    # 将TFM的4D输出 (B, T, N, F) reshape为编码器期望的3D输入 (B, T, N*F)
    B, T, N, F = frequency_masked_x.shape
    encoder_input = frequency_masked_x.reshape(B, T, N * F)
    print(f"送入编码器的张量形状 (B, T, N*F): {encoder_input.shape}")

    freq_encoder = FrequencyEncoder(
        c_in=N * F, # 输入维度是 区域数 * 特征数
        d_model=512, # 增大d_model以匹配更大的输入维度
        e_layers=3,
        device=device
    )
    attention_weights = freq_encoder(encoder_input)
    print("FrequencyEncoder处理完成。\n")

    # --- 5. 验证并保存输出 ---
    print("--- 步骤 5: 验证并保存注意力权重 ---")
    print(f"编码器共输出 {len(attention_weights)} 组权重(3层注意力 + 1层投影)。")
    assert len(attention_weights) == 4, "注意力权重数量应为 e_layers + 1"
    
    # 检查其中一个注意力图的形状
    # [B, H, T, T]
    sample_attn_shape = attention_weights[0].shape
    print(f"第一层注意力权重形状: {sample_attn_shape}")
    assert len(sample_attn_shape) == 4, "注意力权重应为4维"
    assert sample_attn_shape[0] == B
    assert sample_attn_shape[2] == T and sample_attn_shape[3] == T
    print("  ✅ 检查通过: 注意力权重形状正确。\n")

    output_path = 'data/processed/frequency_encoder_attention_weights.pt'
    torch.save(attention_weights, output_path)

    print(f"成功将最终的频域注意力权重保存到: {output_path}")
    print("\n===================================")
    print("=      ✅ 全流程测试完成 ✅       =")
    print("===================================")

def test_frequency_encoder_outputs():
    """
    测试FrequencyEncoder是否能正确输出特征列表和注意力列表，并保存它们。
    现在将使用真实的TFM掩码后数据作为输入，并采用逐节点处理以避免内存溢出。
    """
    print("--- 测试FrequencyEncoder模块输出 (逐节点处理, 修正内存问题) ---")
    
    # --- 1. 加载真实的频率掩码后数据 ---
    input_path = 'C:/Users/86155/Downloads/DSCLBEW/data/processed/tfm_frequency_masked_data.npy'
    try:
        masked_data = np.load(input_path)
        print(f"成功加载数据: {input_path}, 原始形状: {masked_data.shape}")
    except FileNotFoundError:
        print(f"❌ 错误: 输入文件 {input_path} 未找到。")
        print("   - 请先运行TFM相关脚本生成该文件。")
        return

    # --- 2. 准备数据和模型 ---
    days, slots, nodes, features = masked_data.shape
    total_time = days * slots
    
    # 将数据reshape为 (nodes, total_time, features) -> (263, 2016, 2)
    full_data_tensor = torch.from_numpy(masked_data).float().permute(2, 0, 1, 3).reshape(nodes, total_time, features)
    print(f"准备处理的数据形状 (N, T, F): {full_data_tensor.shape}")

    d_model = 256
    n_heads = 8
    e_layers = 3
    # 强制使用CPU以避免显存不足
    device = 'cpu'
    print(f"\n检测到可能存在显存不足风险，强制使用CPU: {device}")
    
    freq_encoder = FrequencyEncoder(c_in=features, d_model=d_model, n_heads=n_heads, e_layers=e_layers).to(device)
    
    # --- 3. 逐节点处理以节省内存 ---
    print("\n开始逐节点处理以避免内存溢出...")
    all_feature_outputs = []
    for i in range(nodes):
        # 获取当前节点的数据 (1, T, F)
        node_input = full_data_tensor[i:i+1].to(device)
        
        # 通过编码器
        feature_outputs, _ = freq_encoder(node_input)
        
        # 保存最后一层的特征，并移回CPU
        all_feature_outputs.append(feature_outputs[-1].detach().cpu())
        
        if (i + 1) % 50 == 0:
            print(f"  - 已处理 {i + 1}/{nodes} 个节点...")

    # --- 4. 合并所有节点的输出 ---
    print("\n所有节点处理完毕，正在合并结果...")
    final_features = torch.cat(all_feature_outputs, dim=0)
    print(f"合并后的特征形状 (N, T, D): {final_features.shape}")
    
    # --- 5. 保存输出 ---
    feature_path = 'data/processed/frequency_feature_output.npy'
    
    # 将特征 (N, T, D) 转换为 (T, N, D) 以匹配时域流
    final_features_transposed = final_features.permute(1, 0, 2)

    np.save(feature_path, final_features_transposed.numpy())
    
    print(f"\n已将特征reshape并保存到: {feature_path} (形状: {final_features_transposed.shape})")
    print("\n--- 测试完成 ---")

if __name__ == '__main__':
    # 只运行我们新的、正确的测试函数
    test_frequency_encoder_outputs() 