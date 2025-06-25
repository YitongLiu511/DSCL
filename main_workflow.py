import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys

# 确保所有 new_version 模块都可以被导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from new_version.frequency_decoder import FrequencyEncoder
from new_version.temporal_attention import process_temporal_masked_data
from new_version.spatial_attention1 import SpatialSelfAttentionBlock
from new_version.process_normal_data import MultipleGCN
from new_version.dual_stream_contrastive import DualStreamContrastiveLoss, sym_kl_loss
from new_version.test_temporal_frequency_contrastive_loss import calculate_contrastive_loss

# --- 1. 定义整合后的主模型 ---
class DSCLModel(nn.Module):
    def __init__(self, n_nodes, n_features, d_model, n_heads, static_adj_matrices, device='cpu'):
        super(DSCLModel, self).__init__()
        self.device = device
        self.n_nodes = n_nodes
        
        # --- 子模块定义 ---
        # 1. 频域流的Encoder
        self.freq_encoder = FrequencyEncoder(c_in=n_features, d_model=d_model, n_heads=n_heads)
        
        # 2. 动态流的模块
        self.dynamic_temp_attn_processor = lambda x: process_temporal_masked_data(x, d_model=d_model, n_heads=n_heads, device=device)
        self.dynamic_spat_attn = SpatialSelfAttentionBlock(d_model=d_model, n_heads=n_heads)
        self.dynamic_encoder = FrequencyEncoder(c_in=d_model, d_model=d_model, n_heads=n_heads)
        
        # 3. 静态流的模块
        self.static_gcn = MultipleGCN(in_channels=d_model, out_channels=d_model, matrices=static_adj_matrices)

        self.to(device)

    def forward(self, freq_masked_data, temp_masked_data):
        """
        freq_masked_data: 形状 (days, slots, nodes, features) -> (14, 144, 263, 2)
        temp_masked_data: 形状 (days, slots, nodes, features) -> (14, 144, 263, 2)
        """
        # --- 数据预处理 ---
        days, slots, _, features = temp_masked_data.shape
        total_time = days * slots
        
        # 统一处理成 (nodes, total_time, features)
        freq_input = freq_masked_data.permute(2, 0, 1, 3).reshape(self.n_nodes, total_time, features)
        temp_input = temp_masked_data.permute(2, 0, 1, 3).reshape(self.n_nodes, total_time, features)

        # --- 流 1: 频域流 ---
        print("  - [Flow 1] Running Frequency Stream...")
        freq_features_list, _ = self.freq_encoder(freq_input)

        # --- 流 2: 动态流 (时域 -> 空间 -> Encoder) ---
        print("  - [Flow 2] Running Dynamic Stream...")
        # a. 时间注意力
        temp_features_dynamic, _ = self.dynamic_temp_attn_processor(temp_input.unsqueeze(0)) # 加一个batch维度
        temp_features_dynamic = temp_features_dynamic.squeeze(0) # 移除batch维度
        # b. 空间注意力
        spatial_features, dynamic_scores = self.dynamic_spat_attn(temp_features_dynamic)
        # c. 后续Encoder
        dynamic_features_list, _ = self.dynamic_encoder(spatial_features)

        # --- 流 3: 静态流 (时域 -> GCN) ---
        print("  - [Flow 3] Running Static Stream...")
        # a. 时间注意力 (与动态流共享)
        # b. GCN
        # GCN期望输入 (B, N, D), 我们这里的时间特征是 (N, T, D), 需要permute
        _, static_scores = self.static_gcn(temp_features_dynamic.permute(1, 0, 2))


        return freq_features_list, dynamic_features_list, dynamic_scores, static_scores

# --- 2. 定义主训练函数 ---
def main_training_workflow():
    print("==============================================")
    print("=      开始执行DSCL模型完整训练流程      =")
    print("==============================================\n")

    # --- 参数设置 ---
    device = 'cpu' # 强制使用CPU以避免内存问题
    n_epochs = 1    # 训练周期，已设为1，快速验证流程
    lr = 0.001      # 学习率
    d_model = 64    # 降低模型维度以减少内存占用
    n_heads = 4     # 减少注意力头数
    
    # --- 加载数据 ---
    print("--- 步骤 1: 加载所有数据 ---")
    try:
        print("[INFO] 正在加载频域掩码数据: data/processed/tfm_frequency_masked_data.npy")
        freq_data = torch.from_numpy(np.load('data/processed/tfm_frequency_masked_data.npy')).float().to(device)
        print(f"[INFO] 频域掩码数据 shape: {freq_data.shape}")
        print("[INFO] 正在加载时域掩码数据: data/processed/tfm_temporal_masked_data.npy")
        temp_data = torch.from_numpy(np.load('data/processed/tfm_temporal_masked_data.npy')).float().to(device)
        print(f"[INFO] 时域掩码数据 shape: {temp_data.shape}")
        print("[INFO] 正在加载静态邻接矩阵: dist.npy, adj.npy, poi_sim.npy")
        adj_dist = torch.from_numpy(np.load('data/processed/dist.npy')).float()
        adj_corr = torch.from_numpy(np.load('data/processed/adj.npy')).float()
        adj_poi = torch.from_numpy(np.load('data/processed/poi_sim.npy')).float()
        static_adj_matrices = torch.stack([adj_dist, adj_corr, adj_poi]).to(device)
        print(f"[INFO] 静态邻接矩阵 shape: {static_adj_matrices.shape}")
        n_nodes = temp_data.shape[2]
        n_features = temp_data.shape[3]
        print(f"数据加载成功! 节点数: {n_nodes}, 特征数: {n_features}\n")
    except FileNotFoundError as e:
        print(f"❌ 错误: 数据文件未找到: {e}")
        return

    # --- 初始化模型和优化器 ---
    print("--- 步骤 2: 初始化模型和优化器 ---")
    print(f"[INFO] d_model: {d_model}, n_heads: {n_heads}, lr: {lr}")
    model = DSCLModel(n_nodes, n_features, d_model, n_heads, static_adj_matrices, device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dual_stream_loss_fn = lambda dy, st: sym_kl_loss(torch.softmax(dy, dim=-1), torch.softmax(st, dim=-1)).mean()
    print("模型和优化器初始化成功!\n")

    # --- 开始训练循环 ---
    print("--- 步骤 3: 开始训练 ---")
    for epoch in range(n_epochs):
        print(f"\n--- Epoch {epoch + 1}/{n_epochs} ---")
        model.train()
        print("[DEBUG] 开始前向传播...")
        freq_feats, dynamic_feats, dynamic_scores, static_scores = model(freq_data, temp_data)
        print(f"[DEBUG] freq_feats[-1] shape: {freq_feats[-1].shape}")
        print(f"[DEBUG] dynamic_feats[-1] shape: {dynamic_feats[-1].shape}")
        print(f"[DEBUG] dynamic_scores shape: {dynamic_scores.shape}")
        print(f"[DEBUG] static_scores shape: {static_scores.shape}")
        
        # 计算损失
        print("  - Calculating losses...")
        # a. 时频对比损失
        # 我们只使用最后一层的特征进行对比
        tf_loss, _, _ = calculate_contrastive_loss([dynamic_feats[-1]], [freq_feats[-1]])
        
        # b. 双流对比损失
        ds_loss = dual_stream_loss_fn(dynamic_scores, static_scores)
        
        # c. 总损失
        total_loss = tf_loss + ds_loss
        print(f"  - 时频损失(TF Loss): {tf_loss.item():.6f}")
        print(f"  - 双流损失(DS Loss): {ds_loss.item():.6f}")
        print(f"  - 总损失(Total Loss): {total_loss.item():.6f}")

        # 反向传播和优化
        print("  - Optimizing...")
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        print(f"--- Epoch {epoch + 1} 完成 ---")

    print("\n===================================")
    print("=      ✅ 训练流程执行完毕 ✅      =")
    print("===================================")


if __name__ == '__main__':
    main_training_workflow() 