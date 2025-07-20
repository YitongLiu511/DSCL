import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
from scipy.stats import multivariate_normal
import gc

# 确保所有 new_version 模块都可以被导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from new_version.frequency_decoder import FrequencyEncoder
from new_version.temporal_attention import TemporalAttentionProcessor
from new_version.spatial_attention1 import SpatialSelfAttentionBlock
from new_version.process_normal_data import MultipleGCN
from new_version.test_temporal_frequency_contrastive_loss import calculate_contrastive_loss
from new_version.dual_stream_contrastive import sym_kl_loss, HardNegativeMiner, ClusterContrastiveTrainer
from new_version.temporal_frequency_masking import TemporalFrequencyMasking
from new_version.temporal_decoder import TemporalDecoder

torch.autograd.set_detect_anomaly(True)

# --- 1. 定义整合后的主模型 ---
class DSCLModel(nn.Module):
    def __init__(self, n_nodes, n_features, d_model, n_heads, static_adj_matrices, device='cpu'):
        super(DSCLModel, self).__init__()
        self.device = str(device)
        self.n_nodes = n_nodes
        self.d_model = d_model
        self.n_heads = n_heads
        
        # --- 子模块定义 ---
        # 1. 频域流的Encoder
        self.freq_encoder = FrequencyEncoder(c_in=n_features, d_model=d_model, n_heads=n_heads)
        
        # 2. 动态流的模块
        self.temporal_processor = TemporalAttentionProcessor(device=self.device)
        self.temporal_decoder = TemporalDecoder(input_dim=2, d_model=d_model, n_heads=n_heads, e_layers=3, dropout=0.1)
        # 新增：空间注意力相关
        self.spatial_input_proj = nn.Linear(n_features, d_model)
        self.spatial_attention = SpatialSelfAttentionBlock(d_model, n_heads)
        self.spatial_proj_to_2 = nn.Linear(d_model, 2)
        
        # 3. 静态流的模块
        self.static_gcn = MultipleGCN(in_channels=n_features, out_channels=d_model, matrices=static_adj_matrices)
        
        # 4. 特征降维层（用于融合重构）
        self.dynamic_proj = nn.Linear(d_model, 2)
        self.static_proj = nn.Linear(d_model, 2)
        
        # 5. 时域解码器输出投影层
        self.temporal_proj_to_d_model = nn.Linear(2, d_model)

        self.to(self.device)

    def forward(self, freq_masked_data, temp_masked_data):
        """
        前向传播，处理三个流
        freq_masked_data: 频域掩码数据 [N, T, F]
        temp_masked_data: 时域掩码数据 [N, T, F]
        """
        # --- 流 1: 频域流 ---
        print("  - [Flow 1] Running Frequency Stream...")
        freq_features_list = []
        for i in range(freq_masked_data.shape[0]):
            node_input = torch.FloatTensor(freq_masked_data[i:i+1]).to(self.device)  # [1, T, F]
            feature_outputs = self.freq_encoder(node_input)
            freq_features_list.append(feature_outputs.detach())
            if (i + 1) % 50 == 0 or (i + 1) == freq_masked_data.shape[0]:
                print(f"    已处理 {i + 1}/{freq_masked_data.shape[0]} 个节点...")
        freq_features = torch.cat(freq_features_list, dim=0)  # [N, T, d_model]

        # --- 流 2: 动态流 (时域 -> 空间 -> 频域) ---
        print("  - [Flow 2] Running Dynamic Stream...")
        # a. 时域注意力处理（分批处理避免内存不足）
        batch_size = 8  # 可根据显存情况调整
        N, T, C = temp_masked_data.shape
        temp_features_list = []
        tematt_attn_list = [[] for _ in range(3)]
        for batch_idx, start in enumerate(range(0, N, batch_size)):
            end = min(start + batch_size, N)
            batch_data = torch.FloatTensor(temp_masked_data[start:end]).to(self.device)
            output, tematt_attention_weights = self.temporal_processor(batch_data)
            temp_features_list.append(output.detach().cpu().numpy())
            for l, attn in enumerate(tematt_attention_weights):
                tematt_attn_list[l].append(attn.mean(dim=1))
            del output, tematt_attention_weights, batch_data
            torch.cuda.empty_cache()
        temp_features = np.concatenate(temp_features_list, axis=0)
        temp_features = torch.from_numpy(temp_features).float().to(self.device)
        del temp_features_list
        torch.cuda.empty_cache()
        tematt_attn = [torch.cat(attn_list, dim=0).mean(dim=0) for attn_list in tematt_attn_list]
        del tematt_attn_list
        torch.cuda.empty_cache()
        
        # b. 空间注意力处理（用成员变量）
        x = temp_features.permute(1, 0, 2)  # [T, N, C]
        x_proj = self.spatial_input_proj(x)  # [T, N, d_model]
        spatial_features_list = []
        spatial_attn_weights_list = []
        for i in range(x_proj.shape[0]):  # 遍历每个时间步
            out, attn = self.spatial_attention(x_proj[i:i+1])  # [1, N, d_model]
            spatial_features_list.append(out)
            spatial_attn_weights_list.append(attn)
        spatial_features = torch.cat(spatial_features_list, dim=0)  # [T, N, d_model]
        spatial_attn_weights = torch.cat(spatial_attn_weights_list, dim=0)  # [T, n_heads, N, N]
        # 投影到2维
        spatial_features_2d = self.spatial_proj_to_2(spatial_features)  # [T, N, 2]
        # 新增：dynamic_scores
        dynamic_scores = spatial_attn_weights.mean(0)  # [n_heads, N, N]
        
        # 保存用于重构损失的动态流特征（仅包含时间注意力+空间注意力）
        dynamic_features_for_recon = spatial_features  # [T, N, d_model]
        
        # c. 时域解码器处理（分批处理）- 用于时频对比学习
        dynamic_features_list = []
        for i in range(spatial_features_2d.shape[1]):
            # 获取当前节点的空间特征 [T, 2]
            node_spatial_features = spatial_features_2d[:, i:i+1, :].to(self.device)  # [T, 1, 2]
            
            # 为时域解码器准备输入
            # 我们需要构造normal_tokens和mask_indices
            T = node_spatial_features.shape[0]
            
            # 简单处理：使用所有时间步作为normal_tokens，没有mask
            normal_tokens = node_spatial_features.squeeze(1)  # [T, 2]
            mask_indices = torch.empty(0, dtype=torch.long, device=self.device)  # 空mask
            
            # 调用时域解码器
            decoded_features = self.temporal_decoder(normal_tokens.unsqueeze(0), mask_indices.unsqueeze(0), T)
            # decoded_features shape: [1, T, 2]
            
            # 将2维特征投影到d_model维
            decoded_features = decoded_features.squeeze(0)  # [T, 2]
            # 使用线性层投影到d_model维
            projected_features = self.temporal_proj_to_d_model(decoded_features)  # [T, d_model]
            dynamic_features_list.append(projected_features.unsqueeze(1))  # [T, 1, d_model]
            
            del decoded_features, projected_features, node_spatial_features
            torch.cuda.empty_cache()
        
        dynamic_features = torch.cat(dynamic_features_list, dim=1)  # [T, N, d_model] - 用于时频对比学习
        del dynamic_features_list, spatial_features_2d
        torch.cuda.empty_cache()

        # --- 流 3: 静态流 ---
        print("  - [Flow 3] Running Static Stream...")
        temp_features_for_gcn = torch.FloatTensor(temp_masked_data).to(self.device)
        temp_features_for_gcn = temp_features_for_gcn.permute(1, 0, 2)
        static_out, static_scores, static_reconstruction = self.static_gcn(temp_features_for_gcn)

        return freq_features, dynamic_features, dynamic_scores, static_scores, static_out, tematt_attn, dynamic_features_for_recon

            # 我们需要构造normal_tokens和maskindices
def main_training_workflow():
    print("==============================================")
    print("=      开始执行DSCL模型完整训练流程      =")
    print("==============================================\n")

    # --- 参数设置 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_str = str(device)
    n_epochs = 20
    lr = 0.01
    d_model = 64
    n_heads = 4
    
    # --- 加载原始数据 ---
    print("--- 步骤 1: 加载原始数据 ---")
    try:
        print("[INFO] 正在加载原始数据: data/datanew/normalized_train.npy")
        original_data = np.load('data/datanew1/normalized_train_injected.npy')
        print(f"[INFO] 原始数据 shape: {original_data.shape}")
        
        # 加载活跃区域索引
        active_zones = np.load('data/datanew1/active_zones.npy')
        print(f"[INFO] 加载活跃区域索引: {active_zones.shape}")
        
        # 加载全节点邻接矩阵
        adj_dist_full = np.load('data/processed/dist.npy')
        adj_corr_full = np.load('data/processed/adj.npy')
        adj_poi_full = np.load('data/processed/poi_sim.npy')
        # 同步筛选邻接矩阵
        adj_dist = adj_dist_full[np.ix_(active_zones, active_zones)]
        adj_corr = adj_corr_full[np.ix_(active_zones, active_zones)]
        adj_poi = adj_poi_full[np.ix_(active_zones, active_zones)]
        static_adj_matrices = torch.stack([
            torch.from_numpy(adj_dist).float(),
            torch.from_numpy(adj_corr).float(),
            torch.from_numpy(adj_poi).float()
        ]).to(device)
        print(f"[INFO] 静态邻接矩阵 shape: {static_adj_matrices.shape}")
        
        # 转为torch
        original_data = torch.from_numpy(original_data).float()
        
        # 数据预处理
        n_nodes = original_data.shape[1]  # 节点数
        n_features = original_data.shape[2]  # 特征数
        T_total = original_data.shape[0]  # 总时间步
        print(f"数据加载成功! 节点数: {n_nodes}, 特征数: {n_features}, 总时间步: {T_total}\n")
        
        # 新增：加载空间距离和POI相似度矩阵（已同步筛选）
        dist_mat = adj_dist
        poi_sim_mat = adj_poi
        # 初始化困难负样本挖掘器
        hard_negative_miner = HardNegativeMiner(dist_mat, poi_sim_mat, top_k=10, true_neg_thresh=0.3)
        node_features = original_data.mean(axis=0).numpy()  # [N, 2]
        print('node_features shape:', node_features.shape)
        cluster_trainer = ClusterContrastiveTrainer(node_features, dist_mat, poi_sim_mat, n_clusters=10, top_k=10, true_neg_thresh=0.3, temperature=0.2)
    except FileNotFoundError as e:
        print(f"❌ 错误: 数据文件未找到: {e}")
    print("--- 步骤 1: 加载原始数据 ---")

    # ====== 删除时间分片，使用完整数据集 ======
    print(f"将使用完整数据集进行处理，总时间步: {T_total}")

    # --- 初始化模型和优化器 ---
    print("--- 步骤 2: 初始化模型和优化器 ---")
    print(f"[INFO] d_model: {d_model}, n_heads: {n_heads}, lr: {lr}")
    model = DSCLModel(n_nodes, n_features, d_model, n_heads, static_adj_matrices, device_str)
    
    # 初始化masker模块
    masker = TemporalFrequencyMasking(
        window_size=10,  # 可根据实际情况调整
        temporal_mask_ratio=0.1,
        frequency_mask_ratio=0.1,
        d_model=d_model,
        n_features=n_features,
        device=device_str
    )
    
    # 新增：可学习损失权重参数
    loss_weights = nn.Parameter(torch.ones(4, dtype=torch.float32, device=device_str))
    # 优化器同时管理主模型、masker和损失权重参数
    optimizer = optim.Adam(list(model.parameters()) + list(masker.parameters()) + [loss_weights], lr=lr)
    print("模型、掩码模块、损失权重和优化器初始化成功!\n")

    # === 步骤 3: 开始训练 ===
    print("\n--- 步骤 3: 开始训练 ---\n")
    model.train()
    masker.train()
    lambda_tf = 1.0    # 时频对比损失权重
    lambda_recon = 0.5 # 融合重构损失权重
        print(f"数据加载成功! 节点数: {n_nodes}, 特征数: {n_features}, 总时间步: {T_total}\n")

    for epoch in range(n_epochs):
        print(f"\n--- Epoch {epoch+1}/{n_epochs} ---")
        optimizer.zero_grad()
        total_tf_loss = 0.0
        total_recon_loss = 0.0
        total_ds_loss = 0.0
        total_cluster_loss = 0.0
        print('node_features shape:', node_features.shape)
        
        # 打印参数norm
        freq_encoder_params = list(model.freq_encoder.parameters())
        temporal_processor_params = list(model.temporal_processor.parameters())
        print("freq_encoder param norm:", sum([p.data.norm().item() for p in freq_encoder_params if p.requires_grad]))
        print("temporal_processor param norm:", sum([p.data.norm().item() for p in temporal_processor_params if p.requires_grad]))
    print(f"将使用完整数据集进行处理，总时间步: {T_total}")
        # ====== 处理完整数据集 ======
        print(f"  处理完整数据集: 时间步 0~{T_total-1}")
    print("--- 步骤 2: 初始化模型和优化器 ---")
    print(f"[INFO] d_model: {d_model}, n_heads: {n_heads}, lr: {lr}")
        # 将数据转换为模型期望的格式 [N, T, F]
        data_for_masking = original_data.permute(1, 0, 2)  # [N, T, F]
        
        # 使用masker生成时频掩码数据
        with torch.no_grad():
            temp_masked_data, temp_mask_indices, freq_masked_data, freq_mask_indices = masker(data_for_masking)
            temp_masked_data = temp_masked_data.detach().cpu().numpy()
            freq_masked_data = freq_masked_data.detach().cpu().numpy()
        print('freq_masked_data shape:', freq_masked_data.shape)
        print('temp_masked_data shape:', temp_masked_data.shape)
        
        # ====== 以下所有流程都用动态生成的掩码数据 ======
        # 1. 频域流 - 修复：处理所有节点
        freq_features_list = []
        for i in range(freq_masked_data.shape[0]):
            node_input = torch.FloatTensor(freq_masked_data[i:i+1]).to(device)
            feature_outputs = model.freq_encoder(node_input)
            freq_features_list.append(feature_outputs.detach())
            del feature_outputs, node_input
            torch.cuda.empty_cache()
        freq_features = torch.cat(freq_features_list, dim=0)  # [N, T, d_model]
        del freq_features_list
        torch.cuda.empty_cache()

        # 2. 动态流（分批处理，防止显存溢出）
        batch_size = 15  # 分批节点数，可根据显存调整
        N, T, C = temp_masked_data.shape
        temp_features_list = []
        tematt_attn_sum = [0 for _ in range(3)]
        tematt_attn_count = [0 for _ in range(3)]
        for start_b in range(0, N, batch_size):
            end_b = min(start_b + batch_size, N)
            batch_data = torch.FloatTensor(temp_masked_data[start_b:end_b]).to(device)
            output, tematt_attention_weights = model.temporal_processor(batch_data)
            temp_features_list.append(output.detach().cpu().numpy())
            # 注意：如果后续不再用tematt_attention_weights，可以不统计
            del output, tematt_attention_weights, batch_data
            torch.cuda.empty_cache()
        temp_features = np.concatenate(temp_features_list, axis=0)
        temp_features = torch.from_numpy(temp_features).float().to(device)
        del temp_features_list
        torch.cuda.empty_cache()
        # 暂时设为空列表，因为不再统计注意力权重
        tematt_attn = []
        del tematt_attn_sum, tematt_attn_count
        torch.cuda.empty_cache()
        
        # 空间注意力
        x = temp_features.permute(1, 0, 2)  # [T, N, C]
        x_proj = model.spatial_input_proj(x)  # [T, N, d_model]
        spatial_features_list = []
        spatial_attn_list = []
        for i in range(x_proj.shape[0]):
            out, attn = model.spatial_attention(x_proj[i:i+1])  # [1, N, d_model]
            spatial_features_list.append(out.squeeze(0))  # [N, d_model]
            spatial_attn_list.append(attn.mean(dim=1))  # [n_heads, N, N]
            del out, attn
            torch.cuda.empty_cache()
        spatial_features = torch.stack(spatial_features_list, dim=0)  # [T, N, d_model]
        spatial_attn_mean = torch.stack(spatial_attn_list, dim=0).mean(dim=0)  # [n_heads, N, N]
        spatial_features_2d = model.spatial_proj_to_2(spatial_features)  # [T, N, 2]
        print(f"spatial_features_2d shape: {spatial_features_2d.shape}")
        
        # 保存用于重构损失的动态流特征（仅包含时间注意力+空间注意力）
        dynamic_features_for_recon = spatial_features  # [T, N, d_model]
        
        del spatial_features_list, spatial_attn_list, x_proj, x
        torch.cuda.empty_cache()
        
        # 时域解码器处理（分批处理，减少内存使用）
        dynamic_features_list = []
        batch_size = 5  # 减少批处理大小
        for batch_start in range(0, spatial_features_2d.shape[1], batch_size):
            batch_end = min(batch_start + batch_size, spatial_features_2d.shape[1])
            batch_features = []
            
            for i in range(batch_start, batch_end):
                # 获取当前节点的空间特征 [T, 2]
                node_spatial_features = spatial_features_2d[:, i:i+1, :].to(device)  # [T, 1, 2]
                
                # 为时域解码器准备输入
                T = node_spatial_features.shape[0]
                
                # 简单处理：使用所有时间步作为normal_tokens，没有mask
                normal_tokens = node_spatial_features.squeeze(1)  # [T, 2]
                mask_indices = torch.empty(0, dtype=torch.long, device=device)  # 空mask
                
                # 调用时域解码器
                with torch.no_grad():  # 减少内存使用
                    decoded_features = model.temporal_decoder(normal_tokens.unsqueeze(0), mask_indices.unsqueeze(0), T)
                    # decoded_features shape: [1, T, 2]
                
                # 将2维特征投影到d_model维
                decoded_features = decoded_features.squeeze(0)  # [T, 2]
                projected_features = model.temporal_proj_to_d_model(decoded_features)  # [T, d_model]
                batch_features.append(projected_features.unsqueeze(1))  # [T, 1, d_model]
                
                del decoded_features, projected_features, node_spatial_features, normal_tokens, mask_indices
                torch.cuda.empty_cache()
            
            # 处理完一个批次后，合并并清理
            batch_tensor = torch.cat(batch_features, dim=1)  # [T, batch_size, d_model]
            dynamic_features_list.append(batch_tensor)
            del batch_features, batch_tensor
            torch.cuda.empty_cache()
            
            # 打印进度
            if (batch_end % 50 == 0) or (batch_end == spatial_features_2d.shape[1]):
                print(f"    已处理 {batch_end}/{spatial_features_2d.shape[1]} 个节点...")
        
        dynamic_features = torch.cat(dynamic_features_list, dim=1)  # [T, N, d_model]
        del dynamic_features_list, spatial_features_2d
        torch.cuda.empty_cache()
        
        # 静态流
        temp_features_for_gcn = torch.FloatTensor(temp_masked_data).to(device)
        temp_features_for_gcn = temp_features_for_gcn.permute(1, 0, 2)
        static_out, static_scores, static_reconstruction = model.static_gcn(temp_features_for_gcn)
        del temp_features_for_gcn
        torch.cuda.empty_cache()
        
        # 4. 时频对比损失 - 计算时域和频域特征的对比损失
        print(f"  [DEBUG] 开始计算时频对比损失...")
        # 收集时域和频域编码器的注意力权重
        tematt_list = []
        freatt_list = []
        
        # 时域编码器注意力权重（从temporal_processor获取）
        if hasattr(model.temporal_processor, 'attention_weights'):
            tematt_list = model.temporal_processor.attention_weights
        
        # 频域编码器注意力权重（从freq_encoder获取）
        if hasattr(model.freq_encoder, 'attention_weights'):
            freatt_list = model.freq_encoder.attention_weights
        
        # 如果没有收集到注意力权重，使用特征进行对比
        if not tematt_list and not freatt_list:
            # 使用频域特征和动态特征进行对比
            if freq_features is not None and dynamic_features is not None:
                print(f"freq_features shape: {freq_features.shape}")
                print(f"dynamic_features shape: {dynamic_features.shape}")
                print(f"  [DEBUG] 频域特征范围: [{freq_features.min():.6f}, {freq_features.max():.6f}]")
                print(f"  [DEBUG] 动态特征范围: [{dynamic_features.min():.6f}, {dynamic_features.max():.6f}]")
                
                # 确保特征维度匹配 - 修复维度处理逻辑
                if len(freq_features.shape) == 3:
                    # freq_features: [N, T, d_model] -> [T, d_model]
                    freq_feat = freq_features.mean(dim=0)  # [T, d_model]
                else:
                    freq_feat = freq_features
                
                if len(dynamic_features.shape) == 3:
                    # dynamic_features: [T, N, d_model] -> [T, d_model]
                    dynamic_feat = dynamic_features.mean(dim=1)  # [T, d_model]
                else:
                    dynamic_feat = dynamic_features
                
                print(f"freq_feat shape: {freq_feat.shape}")
                print(f"dynamic_feat shape: {dynamic_feat.shape}")
                print(f"  [DEBUG] 聚合后频域特征范围: [{freq_feat.min():.6f}, {freq_feat.max():.6f}]")
                print(f"  [DEBUG] 聚合后动态特征范围: [{dynamic_feat.min():.6f}, {dynamic_feat.max():.6f}]")
                
                # 确保两个特征的时间维度匹配
                if freq_feat.shape[0] != dynamic_feat.shape[0]:
                    min_time = min(freq_feat.shape[0], dynamic_feat.shape[0])
                    freq_feat = freq_feat[:min_time]
                    dynamic_feat = dynamic_feat[:min_time]
                    print(f"调整后 freq_feat shape: {freq_feat.shape}")
                    print(f"调整后 dynamic_feat shape: {dynamic_feat.shape}")
                
                # 归一化特征 - 使用原始实现的方式
                
                # 计算KL散度损失 - 使用原始实现的方式
                def my_kl_loss(p, q):
                    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
                    return torch.sum(res, dim=-1)
                
                # 检查特征是否包含NaN或Inf
                if torch.isnan(freq_feat).any() or torch.isinf(freq_feat).any():
                    print("警告：频域特征包含NaN或Inf值，进行清理")
                    freq_feat = torch.nan_to_num(freq_feat, nan=0.0, posinf=1.0, neginf=-1.0)
                
                if torch.isnan(dynamic_feat).any() or torch.isinf(dynamic_feat).any():
                    print("警告：动态特征包含NaN或Inf值，进行清理")
                    dynamic_feat = torch.nan_to_num(dynamic_feat, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # 使用Softmax归一化，确保特征为正且和为1
                freq_feat_norm = torch.softmax(freq_feat, dim=-1)
                dynamic_feat_norm = torch.softmax(dynamic_feat, dim=-1)
                
                # 检查归一化后的特征
                print(f"freq_feat_norm range: [{freq_feat_norm.min().item():.6f}, {freq_feat_norm.max().item():.6f}]")
                print(f"dynamic_feat_norm range: [{dynamic_feat_norm.min().item():.6f}, {dynamic_feat_norm.max().item():.6f}]")
                
                # 实现正确的时频对比损失
                # P^{(L)}: dynamic_feat_norm (时域表示)
                # F^{(L)}: freq_feat_norm (频域表示)
                
                # 计算KL散度
                def kl_divergence(p, q):
                    """计算KL散度 D_KL(p||q)"""
                    return torch.sum(p * (torch.log(p + 1e-8) - torch.log(q + 1e-8)), dim=-1)
                
                # 根据公式：L = min_{F^{(L)}} max_{P^{(L)}} (D_KL(P^{(L)}, F^{(L)}) + D_KL(F^{(L)}, P^{(L)}))
                
                # 第一部分：D_KL(P^{(L)}, F^{(L)}) - 时域到频域的KL散度
                kl_p_to_f = torch.mean(kl_divergence(dynamic_feat_norm, freq_feat_norm))
                
                # 第二部分：D_KL(F^{(L)}, P^{(L)}) - 频域到时域的KL散度  
                kl_f_to_p = torch.mean(kl_divergence(freq_feat_norm, dynamic_feat_norm))
                
                # 总损失：两部分KL散度的和
                tf_loss = kl_p_to_f + kl_f_to_p
                
                print(f"KL(P->F): {kl_p_to_f.item():.6f}")
                print(f"KL(F->P): {kl_f_to_p.item():.6f}")
                print(f"tf_loss (KL(P->F) + KL(F->P)): {tf_loss.item():.6f}")
            else:
                tf_loss = torch.tensor(0.0, device=device)
        else:
            # 使用注意力权重计算对比损失
            tf_loss, adv_loss, con_loss = calculate_contrastive_loss(tematt_list, freatt_list)
        
        # 5. 融合重构损失
        print(f"  [DEBUG] 开始计算融合重构损失...")
        # 使用正确的动态流特征（仅包含时间注意力+空间注意力，不包含时域解码器）
        dynamic_feat_2d = model.dynamic_proj(dynamic_features_for_recon)
        static_feat_2d = model.static_proj(static_out)
        print(f"    [DEBUG] 动态特征2D投影维度: {dynamic_feat_2d.shape}")
        print(f"    [DEBUG] 静态特征2D投影维度: {static_feat_2d.shape}")
        print(f"    [DEBUG] 动态特征2D投影范围: [{dynamic_feat_2d.min():.6f}, {dynamic_feat_2d.max():.6f}]")
        print(f"    [DEBUG] 静态特征2D投影范围: [{static_feat_2d.min():.6f}, {static_feat_2d.max():.6f}]")
        
        # 按照DSCL-master (1)的门控融合方式
        gate = torch.sigmoid(dynamic_feat_2d + static_feat_2d)
        fused_feat = gate * dynamic_feat_2d + (1 - gate) * static_feat_2d
        print(f"    [DEBUG] 门控权重维度: {gate.shape}")
        print(f"    [DEBUG] 门控权重范围: [{gate.min():.6f}, {gate.max():.6f}]")
        print(f"    [DEBUG] 融合特征维度: {fused_feat.shape}")
        print(f"    [DEBUG] 融合特征范围: [{fused_feat.min():.6f}, {fused_feat.max():.6f}]")
        
        # 使用静态流的重构输出计算重构损失
        raw_tensor = original_data
        if list(raw_tensor.shape) != list(fused_feat.shape):
            if raw_tensor.shape[0] == fused_feat.shape[1] and raw_tensor.shape[1] == fused_feat.shape[0]:
                raw_tensor = raw_tensor.permute(1, 0, 2)
        raw_tensor = raw_tensor.to(fused_feat.device)
        print(f"    [DEBUG] 原始数据维度: {raw_tensor.shape}")
        print(f"    [DEBUG] 原始数据范围: [{raw_tensor.min():.6f}, {raw_tensor.max():.6f}]")
        
        # 计算重构损失：使用静态流的重构输出
        static_recon_loss = torch.abs(static_reconstruction - raw_tensor).mean()
        # 原有的融合重构损失
        recon_loss = torch.abs(fused_feat - raw_tensor).mean()
        # 总重构损失
        total_recon_loss = recon_loss + static_recon_loss
        print(f"    [DEBUG] 静态重构损失: {static_recon_loss.item():.6f}")
        print(f"    [DEBUG] 融合重构损失: {recon_loss.item():.6f}")
        print(f"    [DEBUG] 总重构损失: {total_recon_loss.item():.6f}")
        
        del dynamic_feat_2d, static_feat_2d, gate, fused_feat, raw_tensor, static_reconstruction
        torch.cuda.empty_cache()
        
        # 6. 双流对比损失
        print(f"  [DEBUG] 开始计算双流对比损失...")
        ds_loss = compute_dual_stream_loss(spatial_attn_mean, static_scores)
        del static_out, static_scores, spatial_attn_mean
        torch.cuda.empty_cache()
        
        # === 合并损失 ===
        # 新增：每个epoch调用聚类对比损失
        print(f"  [DEBUG] 开始计算聚类对比损失...")
        print('dynamic_features shape:', dynamic_features.shape)
        if len(dynamic_features.shape) == 2:
            dynamic_feat_np = dynamic_features.detach().cpu().numpy()
        elif len(dynamic_features.shape) == 3:
            # 使用时间维度的平均值，而不是最后一个时间步
            dynamic_feat_np = dynamic_features.mean(dim=0).detach().cpu().numpy()
        else:
            raise ValueError('dynamic_features shape 不支持')
        print('dynamic_feat_np shape:', dynamic_feat_np.shape)
        print(f"  [DEBUG] 动态特征numpy范围: [{dynamic_feat_np.min():.6f}, {dynamic_feat_np.max():.6f}]")
        
        # 更新聚类训练器的特征
        cluster_trainer.node_features = dynamic_feat_np
        
        # 挖掘困难负样本
        anchor2negs = hard_negative_miner.mine(dynamic_feat_np)
        print(f"[INFO] Epoch {epoch+1} 挖掘困难真负样本完成，示例: 前3个anchor的负样本索引: {[anchor2negs[i] for i in range(3)]}")
        
        # 计算聚类对比损失
        cluster_contrastive_loss = cluster_trainer.step(device=str(device))
        print(f"[INFO] Epoch {epoch+1} 聚类对比损失: {safe_item(cluster_contrastive_loss):.6f}")
        
        # 损失权重归一化（softmax）
        norm_weights = torch.softmax(loss_weights, dim=0)
        
        # 检查各个损失值
        print(f"tf_loss: {tf_loss.item():.6f}")
        print(f"recon_loss: {total_recon_loss.item():.6f}")
        print(f"ds_loss: {ds_loss.item():.6f}")
        print(f"cluster_contrastive_loss: {safe_item(cluster_contrastive_loss):.6f}")
        print(f"norm_weights: {norm_weights.detach().cpu().numpy()}")
        
        # 检查损失是否包含NaN
        if torch.isnan(tf_loss) or torch.isinf(tf_loss):
            print("警告：tf_loss包含NaN或Inf，设为0")
            tf_loss = torch.tensor(0.0, device=device)
        
        if torch.isnan(total_recon_loss) or torch.isinf(total_recon_loss):
            print("警告：total_recon_loss包含NaN或Inf，设为0")
            total_recon_loss = torch.tensor(0.0, device=device)
        
        if torch.isnan(ds_loss) or torch.isinf(ds_loss):
            print("警告：ds_loss包含NaN或Inf，设为0")
            ds_loss = torch.tensor(0.0, device=device)
        
        total_loss = (
            norm_weights[0] * tf_loss +
            norm_weights[1] * total_recon_loss +
            norm_weights[2] * ds_loss +
            norm_weights[3] * cluster_contrastive_loss
        )
        print(f"  [DEBUG] 总损失计算:")
        print(f"    - 时频对比损失贡献: {norm_weights[0].item():.6f} * {tf_loss.item():.6f} = {(norm_weights[0] * tf_loss).item():.6f}")
        print(f"    - 重构损失贡献: {norm_weights[1].item():.6f} * {total_recon_loss.item():.6f} = {(norm_weights[1] * total_recon_loss).item():.6f}")
        print(f"    - 双流对比损失贡献: {norm_weights[2].item():.6f} * {ds_loss.item():.6f} = {(norm_weights[2] * ds_loss).item():.6f}")
        print(f"    - 聚类对比损失贡献: {norm_weights[3].item():.6f} * {safe_item(cluster_contrastive_loss):.6f} = {(norm_weights[3] * cluster_contrastive_loss).item():.6f}")
        print(f"    - 总损失: {total_loss.item():.6f}")
        
        total_loss.backward()
        total_tf_loss = tf_loss.item()
        total_recon_loss_val = total_recon_loss.item()
        total_ds_loss = ds_loss.item()
        total_cluster_loss = safe_item(cluster_contrastive_loss)
        optimizer.step()
        optimizer.zero_grad()
        gc.collect()
        torch.cuda.empty_cache()
            
        # 打印optimizer参数数量
        print("optimizer param_groups lens:", [len(g['params']) for g in optimizer.param_groups])
        print(f"  - Epoch {epoch+1}/{n_epochs} | TF Loss: {total_tf_loss:.6f} | Recon Loss: {total_recon_loss_val:.6f} | DS Loss: {total_ds_loss:.6f} | Cluster Loss: {total_cluster_loss:.6f} | Total Loss: {total_loss:.6f}")

    print("\n===================================")
    print("=      ✅ 训练流程执行完毕 ✅      =")
    print("===================================")

    # === 保存训练好的模型权重 ===
    save_path = 'dscl_trained_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'masker_state_dict': masker.state_dict(),
        'loss_weights': loss_weights.data,
        'optimizer_state_dict': optimizer.state_dict()
    }, save_path)
    print(f"[INFO] 训练好的模型已保存到: {save_path}")

    # === 步骤 4: 训练后异常检测与评估 ===
    print("\n--- 步骤 4: 训练后异常检测与评估 ---\n")
    model.eval()
    masker.eval()
    print(f"[INFO] 模型设置为评估模式")
    with torch.no_grad():
        # 用训练集重新推理，获得重建误差
        print(f"[DEBUG] 开始推理过程...")
        # 动态生成掩码数据
        data_for_masking = original_data.permute(1, 0, 2)  # [N, T, F]
        print(f"[DEBUG] 评估用掩码输入维度: {data_for_masking.shape}")
        temp_masked_data, _, freq_masked_data, _ = masker(data_for_masking)
        temp_masked_data = temp_masked_data.detach().cpu().numpy()
        freq_masked_data = freq_masked_data.detach().cpu().numpy()
        print(f"[DEBUG] 评估用掩码数据维度:")
        print(f"  - 时域掩码数据: {temp_masked_data.shape}")
        print(f"  - 频域掩码数据: {freq_masked_data.shape}")
        
        # 频域流
        print(f"[DEBUG] 开始频域流推理...")
        freq_features_list = []
        for i in range(freq_masked_data.shape[0]):
            node_input = torch.FloatTensor(freq_masked_data[i:i+1]).to(device)
            feature_outputs = model.freq_encoder(node_input)
            freq_features_list.append(feature_outputs.detach())
        freq_features = torch.cat(freq_features_list, dim=0)  # [N, T, d_model]
        print(f"[DEBUG] 频域流推理输出维度: {freq_features.shape}")

        # 动态流
        print(f"[DEBUG] 开始动态流推理...")
        batch_size = 8
        N, T, C = temp_masked_data.shape
        temp_features_list = []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_data = torch.FloatTensor(temp_masked_data[start:end]).to(device)
            output, _ = model.temporal_processor(batch_data)
            temp_features_list.append(output.detach().cpu().numpy())
        temp_features = np.concatenate(temp_features_list, axis=0)
        temp_features = torch.from_numpy(temp_features).float()
        print(f"[DEBUG] 动态流推理输出维度: {temp_features.shape}")
        
        # 静态流
        print(f"[DEBUG] 开始静态流推理...")
        temp_features_for_gcn = torch.FloatTensor(temp_masked_data).to(device)
        temp_features_for_gcn = temp_features_for_gcn.permute(1, 0, 2)
        static_out, static_scores, static_reconstruction = model.static_gcn(temp_features_for_gcn)
        print(f"[DEBUG] 静态流推理输出维度:")
        print(f"  - static_out: {static_out.shape}")
        print(f"  - static_scores: {static_scores.shape}")
        print(f"  - static_reconstruction: {static_reconstruction.shape}")
        
        # 融合重构
        print(f"[DEBUG] 开始融合重构...")
        dynamic_feat_2d = model.dynamic_proj(spatial_features)  # [T, N, 2]
        static_feat_2d = model.static_proj(static_out)          # [T, N, 2]
        gate = torch.sigmoid(dynamic_feat_2d + static_feat_2d)  # [T, N, 2]
        fused_feat = gate * dynamic_feat_2d + (1 - gate) * static_feat_2d  # [T, N, 2]
        raw_tensor = original_data
        if list(raw_tensor.shape) != list(fused_feat.shape):
            if raw_tensor.shape[0] == fused_feat.shape[1] and raw_tensor.shape[1] == fused_feat.shape[0]:
                raw_tensor = raw_tensor.permute(1, 0, 2)
        raw_tensor = raw_tensor.to(fused_feat.device)
        print(f"[DEBUG] 融合重构维度:")
        print(f"  - dynamic_feat_2d: {dynamic_feat_2d.shape}")
        print(f"  - static_feat_2d: {static_feat_2d.shape}")
        print(f"  - gate: {gate.shape}")
        print(f"  - fused_feat: {fused_feat.shape}")
        print(f"  - raw_tensor: {raw_tensor.shape}")
        
        # 计算重建误差
        recon_error = torch.abs(fused_feat - raw_tensor).mean(dim=2).cpu().numpy()  # [T, N]
        recon_error = recon_error.T  # [N, T]
        print(f"[DEBUG] 重建误差维度: {recon_error.shape}")
        print(f"[DEBUG] 重建误差范围: [{recon_error.min():.6f}, {recon_error.max():.6f}]")

    # 计算每个区域和每个时间的异常分数
    print(f"[DEBUG] 开始计算异常分数...")
    S_region = np.mean(recon_error, axis=1)  # (N,)
    S_time = np.mean(recon_error, axis=0)    # (T,)
    print(f"[DEBUG] 区域异常分数维度: {S_region.shape}")
    print(f"[DEBUG] 时间异常分数维度: {S_time.shape}")
    print(f"[DEBUG] 区域异常分数范围: [{S_region.min():.6f}, {S_region.max():.6f}]")
    print(f"[DEBUG] 时间异常分数范围: [{S_time.min():.6f}, {S_time.max():.6f}]")

    # 构造二维特征并拟合高斯分布
    N, T = recon_error.shape
    X = np.array([[S_region[n], S_time[t]] for n in range(N) for t in range(T)])  # (N*T, 2)
    print(f"[DEBUG] 二维特征维度: {X.shape}")
    print(f"[DEBUG] 二维特征范围: [{X.min():.6f}, {X.max():.6f}]")
    mean = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    print(f"[DEBUG] 高斯分布参数:")
    print(f"  - 均值: {mean}")
    print(f"  - 协方差矩阵维度: {cov.shape}")
    mvn = multivariate_normal(mean=mean, cov=cov)
    logpdf = mvn.logpdf(X)  # (N*T,)
    anomaly_score = -logpdf  # 越大越异常
    anomaly_score_2d = anomaly_score.reshape(N, T)
    print(f"[DEBUG] 异常分数维度: {anomaly_score_2d.shape}")
    print(f"[DEBUG] 异常分数范围: [{anomaly_score_2d.min():.6f}, {anomaly_score_2d.max():.6f}]")

    # 读取标签
    print(f"[DEBUG] 开始读取标签...")
    labels = np.load('data/datanew1/anomaly_labels_train_injected.npy')  # (N, T, 2)
    print(f"[DEBUG] 标签维度: {labels.shape}")
    labels_combined = ((labels[:, :, 0] == 1) | (labels[:, :, 1] == 1)).astype(int)  # (N, T)
    print(f"[DEBUG] 合并标签维度: {labels_combined.shape}")
    print(f"[DEBUG] 异常样本数量: {labels_combined.sum()}")

    # 展平成一维
    anomaly_score_flat = anomaly_score_2d.flatten()
    labels_flat = labels_combined.flatten()
    n = len(labels_flat)
    print(f"[DEBUG] 展平后维度:")
    print(f"  - 异常分数: {anomaly_score_flat.shape}")
    print(f"  - 标签: {labels_flat.shape}")
    print(f"  - 总样本数: {n}")

    # Recall@5%、Recall@10%
    idx_5 = np.argsort(anomaly_score_flat)[-int(n*0.05):]
    idx_10 = np.argsort(anomaly_score_flat)[-int(n*0.10):]
    recall_5 = labels_flat[idx_5].sum() / labels_flat.sum() if labels_flat.sum() > 0 else 0
    recall_10 = labels_flat[idx_10].sum() / labels_flat.sum() if labels_flat.sum() > 0 else 0
    print(f"[DEBUG] 评估指标:")
    print(f"  - Top 5% 异常样本数: {labels_flat[idx_5].sum()}")
    print(f"  - Top 10% 异常样本数: {labels_flat[idx_10].sum()}")
    print(f"  - 总异常样本数: {labels_flat.sum()}")

    # ROC-AUC
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(labels_flat, anomaly_score_flat)
        print(f"[DEBUG] ROC-AUC计算成功: {auc:.6f}")
    except Exception as e:
        print(f"AUC计算错误: {e}")
        auc = float('nan')

    print("\n=== 异常检测评估结果 ===")
    print(f"Recall@5%:  {recall_5:.4f}")
    print(f"Recall@10%: {recall_10:.4f}")
    print(f"ROC-AUC:    {auc:.4f}")

# --- 辅助函数：融合重构损失计算 ---
def compute_fusion_reconstruction_loss(dynamic_features, static_features, original_data, dynamic_proj, static_proj):
    """
    计算融合重构损失
    """
    print(f"[DEBUG] 融合重构损失计算:")
    print(f"  - 动态特征维度: {dynamic_features.shape}")
    print(f"  - 静态特征维度: {static_features.shape}")
    print(f"  - 原始数据维度: {original_data.shape}")
    
    # 动态特征降维到2维
    dynamic_2d = dynamic_proj(dynamic_features)  # [T, N, 2]
    print(f"  - 动态特征2D投影维度: {dynamic_2d.shape}")
    print(f"  - 动态特征2D投影范围: [{dynamic_2d.min():.6f}, {dynamic_2d.max():.6f}]")
    
    # 静态特征降维到2维
    static_2d = static_proj(static_features)     # [T, N, 2]
    print(f"  - 静态特征2D投影维度: {static_2d.shape}")
    print(f"  - 静态特征2D投影范围: [{static_2d.min():.6f}, {static_2d.max():.6f}]")
    
    # 门控融合
    gate = torch.sigmoid(dynamic_2d + static_2d)
    fused_feat = gate * dynamic_2d + (1 - gate) * static_2d
    print(f"  - 门控权重维度: {gate.shape}")
    print(f"  - 门控权重范围: [{gate.min():.6f}, {gate.max():.6f}]")
    print(f"  - 融合特征维度: {fused_feat.shape}")
    print(f"  - 融合特征范围: [{fused_feat.min():.6f}, {fused_feat.max():.6f}]")
    
    # 计算MAE损失
    if original_data.shape[0] == dynamic_2d.shape[1]:  # 如果原始数据是[N, T, 2]
        original_data = original_data.permute(1, 0, 2)  # 转为[T, N, 2]
        print(f"  - 调整后原始数据维度: {original_data.shape}")
    
    recon_loss = torch.abs(fused_feat - original_data).mean()
    print(f"  - 重构损失: {recon_loss.item():.6f}")
    return recon_loss

# --- 辅助函数：双流对比损失计算 ---
def compute_dual_stream_loss(dynamic_scores, static_scores):
    """
    计算双流对比损失
    Args:
        dynamic_scores: 动态流注意力分数 [n_heads, N, N] 或 [N, N]
        static_scores: 静态流注意力分数 [N, N]
    """
    print(f"[DEBUG] 双流对比损失计算:")
    print(f"  - 动态分数维度: {dynamic_scores.shape}")
    print(f"  - 静态分数维度: {static_scores.shape}")
    
    # 确保两个分数在同一个设备上
    dynamic_scores = dynamic_scores.to(static_scores.device)
    
    # 处理动态分数：如果是多头注意力，取平均
    if len(dynamic_scores.shape) == 3:
        # [n_heads, N, N] -> [N, N]
        dynamic_scores = dynamic_scores.mean(dim=0)
        print(f"  - 动态分数平均后维度: {dynamic_scores.shape}")
    
    # 确保形状匹配
    if dynamic_scores.shape != static_scores.shape:
        print(f"警告：动态分数形状 {dynamic_scores.shape} 与静态分数形状 {static_scores.shape} 不匹配")
        # 取较小的维度
        min_size = min(dynamic_scores.shape[0], static_scores.shape[0])
        dynamic_scores = dynamic_scores[:min_size, :min_size]
        static_scores = static_scores[:min_size, :min_size]
        print(f"  - 调整后动态分数维度: {dynamic_scores.shape}")
        print(f"  - 调整后静态分数维度: {static_scores.shape}")
    
    # 检查数值范围
    print(f"动态分数范围: [{dynamic_scores.min().item():.6f}, {dynamic_scores.max().item():.6f}]")
    print(f"静态分数范围: [{static_scores.min().item():.6f}, {static_scores.max().item():.6f}]")
    
    # 归一化分数 - 使用softmax确保和为1
    dynamic_scores_norm = torch.softmax(dynamic_scores.flatten(), dim=0).reshape(dynamic_scores.shape)
    static_scores_norm = torch.softmax(static_scores.flatten(), dim=0).reshape(static_scores.shape)
    print(f"  - 归一化后动态分数范围: [{dynamic_scores_norm.min().item():.6f}, {dynamic_scores_norm.max().item():.6f}]")
    print(f"  - 归一化后静态分数范围: [{static_scores_norm.min().item():.6f}, {static_scores_norm.max().item():.6f}]")
    
    # 计算对称KL散度
    def sym_kl_loss(p, q):
        """计算对称KL散度"""
        kl_pq = torch.sum(p * (torch.log(p + 1e-8) - torch.log(q + 1e-8)))
        kl_qp = torch.sum(q * (torch.log(q + 1e-8) - torch.log(p + 1e-8)))
        return kl_pq + kl_qp
    
    ds_loss = sym_kl_loss(dynamic_scores_norm, static_scores_norm)
    
    print(f"双流对比损失: {ds_loss.item():.6f}")
    return ds_loss

def safe_item(x):
    if isinstance(x, torch.Tensor):
        return x.item()
    return float(x)

if __name__ == '__main__':
    main_training_workflow() 