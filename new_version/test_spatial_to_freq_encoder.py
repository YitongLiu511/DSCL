import torch
import numpy as np
import os
from frequency_decoder import Encoder, AttentionLayer

def test_downstream_encoder_processing():
    """
    测试加载空间注意力模块的特征输出，并将其送入下游的Encoder模块进行处理。
    并将最终特征输出保存为temporal_feature_output.npy。
    """
    print("--- 开始测试 空间特征 -> 下游Encoder Pipeline ---")

    # --- 1. 定义和加载输入文件 ---
    # 这是由 test_spatial_attention1.py 生成的高维特征输出
    spatial_feature_path = 'C:/Users/86155/Downloads/DSCLBEW/data/processed/dynamic_stream_output.npy'
    
    print(f"1. 准备加载空间注意力模块的特征输出: {os.path.basename(spatial_feature_path)}")
    try:
        spatial_feature_output = np.load(spatial_feature_path)
        print(f"   - 成功加载特征文件，形状: {spatial_feature_output.shape}")
    except FileNotFoundError:
        print(f"❌ 错误: 输入文件 {spatial_feature_path} 未找到。")
        print("   - 请先运行 'python new_version/test_spatial_attention1.py' 来生成该文件。")
        return

    # --- 2. 准备送入Encoder的数据 ---
    # 将numpy数组转换为torch张量
    # 当前形状已经是 [B, T, D] 的格式 (e.g., [2016, 263, 512])，可以直接使用
    spatial_feature_tensor = torch.from_numpy(spatial_feature_output).float()
    print(f"\n2. 已将数据转换为Torch张量，准备送入Encoder。")

    # --- 3. 初始化并运行下游的Encoder模块 ---
    # Encoder是通用的，可以处理任意高维特征
    print("3. 正在初始化并运行下游的Encoder模块...")
    
    # 动态获取d_model
    d_model = spatial_feature_tensor.shape[-1]
    print(f"   - 从输入数据推断出 d_model = {d_model}")
    
    n_heads = 8 # 假设 n_heads 为 8
    e_layers = 3 # 假设Encoder有3层

    # 初始化一个与 frequency_decoder.py 中结构相同的Encoder
    downstream_encoder = Encoder(
        [
            AttentionLayer(d_model=d_model, n_heads=n_heads, dropout=0.1) for _ in range(e_layers)
        ],
        norm_layer=torch.nn.LayerNorm(d_model)
    )
    
    # 将空间特征送入Encoder
    # Encoder 返回 (最终输出, 每一层的特征输出列表, 每一层的注意力权重列表)
    final_output, feature_outputs_list, attention_weights_list = downstream_encoder(spatial_feature_tensor)
    
    print(f"   - Encoder处理后的最终特征形状: {final_output.shape}")
    print(f"   - Encoder共输出了 {len(attention_weights_list)} 层注意力权重")
    if attention_weights_list:
        print(f"   - 其中第一层注意力权重的形状: {attention_weights_list[0].shape}")

    # --- 4. 保存最终特征输出 ---
    save_path = 'C:/Users/86155/Downloads/DSCLBEW/data/processed/temporal_feature_output.npy'
    np.save(save_path, final_output.detach().numpy())
    print(f"\n已将Encoder最终特征输出保存到: {save_path}")

    # --- 5. 结论 ---
    print("\n--- 测试完成 ---")
    print("结论: 已成功将'spatial_attention1.py'的特征输出，送入一个独立的、与'frequency_decoder.py'结构相同的Encoder中进行处理，并保存为时频对比损失的时域输入。")
    print("✓ Pipeline 运行成功！")


if __name__ == '__main__':
    test_downstream_encoder_processing() 