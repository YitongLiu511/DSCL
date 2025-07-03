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
        self.dynamic_encoder = FrequencyEncoder(c_in=2, d_model=d_model, n_heads=n_heads)
        # 新增：空间注意力相关
        self.spatial_input_proj = nn.Linear(n_features, d_model)
        self.spatial_attention = SpatialSelfAttentionBlock(d_model, n_heads)
        self.spatial_proj_to_2 = nn.Linear(d_model, 2)
        
        # 3. 静态流的模块
        self.static_gcn = MultipleGCN(in_channels=n_features, out_channels=d_model, matrices=static_adj_matrices)
        
        # 4. 特征降维层（用于融合重构）
        self.dynamic_proj = nn.Linear(d_model, 2)
        self.static_proj = nn.Linear(d_model, 2)

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
            feature_outputs, freatt_attention_weights = self.freq_encoder(node_input)
            freq_features_list.append(feature_outputs[-1].detach())
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
        
        # c. 频域编码器处理（分批处理）
        dynamic_features_list = []
        for i in range(spatial_features_2d.shape[1]):
            node_input = spatial_features_2d[:, i:i+1, :].to(self.device)
            feature_outputs, _ = self.dynamic_encoder(node_input)
            # 取最后一层特征，shape: [T, 1, d_model]
            dynamic_features_list.append(feature_outputs[-1].detach())
            del feature_outputs, node_input
            torch.cuda.empty_cache()
        dynamic_features = torch.cat(dynamic_features_list, dim=1)  # [T, N, d_model]
        del dynamic_features_list, spatial_features_2d
        torch.cuda.empty_cache()

        # --- 流 3: 静态流 ---
        print("  - [Flow 3] Running Static Stream...")
        temp_features_for_gcn = torch.FloatTensor(temp_masked_data).to(self.device)
        temp_features_for_gcn = temp_features_for_gcn.permute(1, 0, 2)
        static_out, static_scores = self.static_gcn(temp_features_for_gcn)

        return freq_features, dynamic_features, dynamic_scores, static_scores, static_out, tematt_attn, freatt_attention_weights

# --- 2. 定义主训练函数 ---
def main_training_workflow():
    print("==============================================")
    print("=      开始执行DSCL模型完整训练流程      =")
    print("==============================================\n")

    # --- 参数设置 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_str = str(device)
    n_epochs = 50
    lr = 0.01
    d_model = 64
    n_heads = 4
    
    # --- 加载数据 ---
    print("--- 步骤 1: 加载所有数据 ---")
    try:
        print("[INFO] 正在加载频域掩码数据: data/datanew/frequency_masked.npy")
        freq_data = np.load('data/datanew/frequency_masked.npy')  # [N, T, C]
        print(f"[INFO] 频域掩码数据 shape: {freq_data.shape}")
        print("[INFO] 正在加载时域掩码数据: data/datanew/temporal_masked.npy")
        temp_data = np.load('data/datanew/temporal_masked.npy')   # [N, T, C]
        print(f"[INFO] 时域掩码数据 shape: {temp_data.shape}")
        print("[INFO] 正在加载原始数据: data/datanew/normalized_train.npy")
        original_data = torch.from_numpy(np.load('data/datanew/normalized_train.npy')).float()
        print(f"[INFO] 原始数据 shape: {original_data.shape}")
        print("[INFO] 正在加载邻接矩阵: data/processed/dist.npy, adj.npy, poi_sim.npy")
        adj_dist = torch.from_numpy(np.load('data/processed/dist.npy')).float()
        adj_corr = torch.from_numpy(np.load('data/processed/adj.npy')).float()
        adj_poi = torch.from_numpy(np.load('data/processed/poi_sim.npy')).float()
        static_adj_matrices = torch.stack([adj_dist, adj_corr, adj_poi]).to(device)
        print(f"[INFO] 静态邻接矩阵 shape: {static_adj_matrices.shape}")
        n_nodes = temp_data.shape[0]
        n_features = temp_data.shape[2]
        T_total = temp_data.shape[1]
        print(f"数据加载成功! 节点数: {n_nodes}, 特征数: {n_features}, 总时间步: {T_total}\n")
        # 新增：加载空间距离和POI相似度矩阵
        dist_mat = np.load('data/processed/dist.npy')
        poi_sim_mat = np.load('data/processed/poi_sim.npy')
        # 初始化困难负样本挖掘器
        hard_negative_miner = HardNegativeMiner(dist_mat, poi_sim_mat, top_k=10, true_neg_thresh=0.7)
        # 新增：初始化聚类对比训练器
        normalized_data = np.load('data/datanew/normalized_train.npy')  # [2016, 263, 2]
        print('normalized_data shape:', normalized_data.shape)
        node_features = normalized_data.mean(axis=0)  # [263, 2]
        print('node_features shape:', node_features.shape)
        cluster_trainer = ClusterContrastiveTrainer(node_features, dist_mat, poi_sim_mat, n_clusters=10, top_k=10, true_neg_thresh=0.7, temperature=0.2)
    except FileNotFoundError as e:
        print(f"❌ 错误: 数据文件未找到: {e}")
        return

    # ====== 分段滑窗参数 ======
    T_limit = 512
    stride = 512  # 无重叠
    num_chunks = (T_total + T_limit - 1) // T_limit
    print(f"将按窗口{T_limit}步分{num_chunks}段处理")

    # --- 初始化模型和优化器 ---
    print("--- 步骤 2: 初始化模型和优化器 ---")
    print(f"[INFO] d_model: {d_model}, n_heads: {n_heads}, lr: {lr}")
    model = DSCLModel(n_nodes, n_features, d_model, n_heads, static_adj_matrices, device_str)
    # 新增：初始化masker模块
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
    lambda_tf = 1.0    # 时频对比损失权重
    lambda_recon = 0.5 # 融合重构损失权重
    lambda_ds = 1.0    # 双流对比损失权重

    for epoch in range(n_epochs):
        print(f"\n--- Epoch {epoch+1}/{n_epochs} ---")
        optimizer.zero_grad()
        total_tf_loss = 0.0
        total_recon_loss = 0.0
        total_ds_loss = 0.0
        total_cluster_loss = 0.0
        total_loss = 0.0
        # 打印参数norm
        freq_encoder_params = list(model.freq_encoder.parameters())
        temporal_processor_params = list(model.temporal_processor.parameters())
        print("freq_encoder param norm:", sum([p.data.norm().item() for p in freq_encoder_params if p.requires_grad]))
        print("temporal_processor param norm:", sum([p.data.norm().item() for p in temporal_processor_params if p.requires_grad]))
        for chunk_idx in range(num_chunks):
            start = chunk_idx * stride
            end = min(start + T_limit, T_total)
            freq_data_chunk = freq_data[:, start:end, :]
            temp_data_chunk = temp_data[:, start:end, :]
            original_data_chunk = original_data[start:end, :, :]
            print(f"  处理窗口 {chunk_idx+1}/{num_chunks}: 时间步 {start}~{end-1}")
            # ====== 以下所有流程都用 *_chunk 变量 ======
            # 1. 频域流
            freq_features_sum = 0
            freatt_attn_sum = [0 for _ in range(3)]
            freatt_attn_count = [0 for _ in range(3)]
            for i in range(freq_data_chunk.shape[0]):
                node_input = torch.FloatTensor(freq_data_chunk[i:i+1]).to(device)
                feature_outputs, freatt_attention_weights = model.freq_encoder(node_input)
                freq_features_sum += feature_outputs[-1].detach()
                for l, attn in enumerate(freatt_attention_weights):
                    attn_mean = attn.mean(dim=1)
                    freatt_attn_sum[l] += attn_mean.sum(dim=0)
                    freatt_attn_count[l] += attn_mean.shape[0]
                del feature_outputs, freatt_attention_weights, node_input
                torch.cuda.empty_cache()
            freq_features = freq_features_sum / freq_data_chunk.shape[0]
            freatt_attn = [s / c for s, c in zip(freatt_attn_sum, freatt_attn_count)]
            del freq_features_sum, freatt_attn_sum, freatt_attn_count
            torch.cuda.empty_cache()

            # 2. 动态流
            batch_size = 8
            N, T, C = temp_data_chunk.shape
            temp_features_list = []
            tematt_attn_sum = [0 for _ in range(3)]
            tematt_attn_count = [0 for _ in range(3)]
            for batch_idx, start_b in enumerate(range(0, N, batch_size)):
                end_b = min(start_b + batch_size, N)
                batch_data = torch.FloatTensor(temp_data_chunk[start_b:end_b]).to(device)
                output, tematt_attention_weights = model.temporal_processor(batch_data)
                temp_features_list.append(output.detach().cpu().numpy())
                for l, attn in enumerate(tematt_attention_weights):
                    attn_mean = attn.mean(dim=1)
                    tematt_attn_sum[l] += attn_mean.sum(dim=0)
                    tematt_attn_count[l] += attn_mean.shape[0]
                del output, tematt_attention_weights, batch_data
                torch.cuda.empty_cache()
            temp_features = np.concatenate(temp_features_list, axis=0)
            temp_features = torch.from_numpy(temp_features).float().to(device)
            del temp_features_list
            torch.cuda.empty_cache()
            tematt_attn = [s / c for s, c in zip(tematt_attn_sum, tematt_attn_count)]
            del tematt_attn_sum, tematt_attn_count
            torch.cuda.empty_cache()
            # 空间注意力
            x = temp_features.permute(1, 0, 2)
            x_proj = model.spatial_input_proj(x)
            spatial_features_sum = 0
            spatial_attn_sum = 0
            for i in range(x_proj.shape[0]):
                out, attn = model.spatial_attention(x_proj[i:i+1])
                spatial_features_sum += out.detach()
                spatial_attn_sum += attn.mean(dim=1).detach()
                del out, attn
                torch.cuda.empty_cache()
            spatial_features = spatial_features_sum / x_proj.shape[0]
            spatial_attn_mean = spatial_attn_sum / x_proj.shape[0]
            spatial_features_2d = model.spatial_proj_to_2(spatial_features)
            del spatial_features_sum, spatial_attn_sum, x_proj, x
            torch.cuda.empty_cache()
            # 频域编码器处理
            dynamic_features_list = []
            for i in range(spatial_features_2d.shape[1]):
                node_input = spatial_features_2d[:, i:i+1, :].to(device)
                feature_outputs, _ = model.dynamic_encoder(node_input)
                # 取最后一层特征，shape: [T, 1, d_model]
                dynamic_features_list.append(feature_outputs[-1].detach())
                del feature_outputs, node_input
                torch.cuda.empty_cache()
            dynamic_features = torch.cat(dynamic_features_list, dim=1)  # [T, N, d_model]
            del dynamic_features_list, spatial_features_2d
            torch.cuda.empty_cache()
            # 静态流
            temp_features_for_gcn = torch.FloatTensor(temp_data_chunk).to(device)
            temp_features_for_gcn = temp_features_for_gcn.permute(1, 0, 2)
            static_out, static_scores = model.static_gcn(temp_features_for_gcn)
            del temp_features_for_gcn
            torch.cuda.empty_cache()
            # 4. 时频对比损失
            def my_kl_loss(p, q):
                res = p * (torch.log(p + 1e-4) - torch.log(q.detach() + 1e-4))
                return res.sum(dim=-1).mean()
            tf_loss = 0
            for u in range(len(freatt_attn)):
                kl1 = my_kl_loss(freatt_attn[u], tematt_attn[u])
                kl2 = my_kl_loss(tematt_attn[u], freatt_attn[u])
                tf_loss += kl1 + kl2
            tf_loss = tf_loss / len(freatt_attn)
            # 5. 融合重构损失
            dynamic_feat_2d = model.dynamic_proj(dynamic_features)
            static_feat_2d = model.static_proj(static_out)
            gate = torch.sigmoid(dynamic_feat_2d + static_feat_2d)
            fused_feat = gate * dynamic_feat_2d + (1 - gate) * static_feat_2d
            raw_tensor = original_data_chunk
            if list(raw_tensor.shape) != list(fused_feat.shape):
                if raw_tensor.shape[0] == fused_feat.shape[1] and raw_tensor.shape[1] == fused_feat.shape[0]:
                    raw_tensor = raw_tensor.permute(1, 0, 2)
            raw_tensor = raw_tensor.to(fused_feat.device)
            recon_loss = torch.abs(fused_feat - raw_tensor).mean()
            del dynamic_feat_2d, static_feat_2d, gate, fused_feat, raw_tensor
            torch.cuda.empty_cache()
            # 6. 双流对比损失
            ds_loss = compute_dual_stream_loss(spatial_attn_mean, static_scores)
            del static_out, static_scores, spatial_attn_mean
            torch.cuda.empty_cache()
            # === 合并损失 ===
            # 新增：每个chunk调用聚类对比损失
            with torch.no_grad():
                print('dynamic_features shape:', dynamic_features.shape)
                if len(dynamic_features.shape) == 2:
                    dynamic_feat_np = dynamic_features.detach().cpu().numpy()
                elif len(dynamic_features.shape) == 3:
                    dynamic_feat_np = dynamic_features[-1].detach().cpu().numpy()
                else:
                    raise ValueError('dynamic_features shape 不支持')
                print('dynamic_feat_np shape:', dynamic_feat_np.shape)
                anchor2negs = hard_negative_miner.mine(dynamic_feat_np)
                print(f"[INFO] Epoch {epoch+1} 挖掘困难真负样本完成，示例: 前3个anchor的负样本索引: {[anchor2negs[i] for i in range(3)]}")
                # 新增：每个epoch前更新聚类对比特征
                cluster_trainer.node_features = dynamic_feat_np
            cluster_contrastive_loss = cluster_trainer.step(device=str(device))
            print(f"[INFO] Epoch {epoch+1} 聚类对比损失: {safe_item(cluster_contrastive_loss):.6f}")
            # 损失权重归一化（softmax）
            norm_weights = torch.softmax(loss_weights, dim=0)
            chunk_loss = (
                norm_weights[0] * tf_loss +
                norm_weights[1] * recon_loss +
                norm_weights[2] * ds_loss +
                norm_weights[3] * cluster_contrastive_loss
            )
            chunk_loss.backward()
            total_tf_loss += tf_loss.item()
            total_recon_loss += recon_loss.item()
            total_ds_loss += ds_loss.item()
            total_loss += chunk_loss.item()
            optimizer.step()
            optimizer.zero_grad()
            gc.collect()
            torch.cuda.empty_cache()
        # 打印optimizer参数数量
        print("optimizer param_groups lens:", [len(g['params']) for g in optimizer.param_groups])
        print(f"  - Epoch {epoch+1}/{n_epochs} | TF Loss: {total_tf_loss/num_chunks:.6f} | Recon Loss: {total_recon_loss/num_chunks:.6f} | DS Loss: {total_ds_loss/num_chunks:.6f} | Cluster Loss: {(total_cluster_loss/num_chunks):.6f} | Total Loss: {total_loss/num_chunks:.6f}")

    print("\n===================================")
    print("=      ✅ 训练流程执行完毕 ✅      =")
    print("===================================")

    # === 保存训练好的模型权重 ===
    save_path = 'dscl_trained_model.pth'
    torch.save(model.state_dict(), save_path)
    print(f"[INFO] 训练好的模型已保存到: {save_path}")

    # === 步骤 4: 训练后异常检测与评估 ===
    print("\n--- 步骤 4: 训练后异常检测与评估 ---\n")
    model.eval()
    with torch.no_grad():
        # 用训练集重新推理，获得重建误差
        # 频域流
        freq_features_list = []
        for i in range(freq_data.shape[0]):
            node_input = torch.FloatTensor(freq_data[i:i+1]).to(device)
            feature_outputs, _ = model.freq_encoder(node_input)
            freq_features_list.append(feature_outputs[-1].detach())
        freq_features = torch.cat(freq_features_list, dim=0)  # [N, T, d_model]

        # 动态流
        batch_size = 8
        N, T, C = temp_data.shape
        temp_features_list = []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_data = torch.FloatTensor(temp_data[start:end]).to(device)
            output, _ = model.temporal_processor(batch_data)
            temp_features_list.append(output.detach().cpu().numpy())
        temp_features = np.concatenate(temp_features_list, axis=0)
        temp_features = torch.from_numpy(temp_features).float()
        # 静态流
        temp_features_for_gcn = torch.FloatTensor(temp_data).to(device)
        temp_features_for_gcn = temp_features_for_gcn.permute(1, 0, 2)
        static_out, static_scores = model.static_gcn(temp_features_for_gcn)
        # 融合重构
        dynamic_feat_2d = model.dynamic_proj(spatial_features)  # [T, N, 2]
        static_feat_2d = model.static_proj(static_out)          # [T, N, 2]
        gate = torch.sigmoid(dynamic_feat_2d + static_feat_2d)  # [T, N, 2]
        fused_feat = gate * dynamic_feat_2d + (1 - gate) * static_feat_2d  # [T, N, 2]
        raw_tensor = original_data
        if list(raw_tensor.shape) != list(fused_feat.shape):
            if raw_tensor.shape[0] == fused_feat.shape[1] and raw_tensor.shape[1] == fused_feat.shape[0]:
                raw_tensor = raw_tensor.permute(1, 0, 2)
        raw_tensor = raw_tensor.to(fused_feat.device)
        # 计算重建误差
        recon_error = torch.abs(fused_feat - raw_tensor).mean(dim=2).cpu().numpy()  # [T, N]
        recon_error = recon_error.T  # [N, T]

    # 计算每个区域和每个时间的异常分数
    S_region = np.mean(recon_error, axis=1)  # (N,)
    S_time = np.mean(recon_error, axis=0)    # (T,)

    # 构造二维特征并拟合高斯分布
    N, T = recon_error.shape
    X = np.array([[S_region[n], S_time[t]] for n in range(N) for t in range(T)])  # (N*T, 2)
    mean = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    mvn = multivariate_normal(mean=mean, cov=cov)
    logpdf = mvn.logpdf(X)  # (N*T,)
    anomaly_score = -logpdf  # 越大越异常
    anomaly_score_2d = anomaly_score.reshape(N, T)

    # 读取标签
    labels = np.load('data/datanew/anomaly_labels_train.npy')  # (N, T, 2)
    labels_combined = ((labels[:, :, 0] == 1) | (labels[:, :, 1] == 1)).astype(int)  # (N, T)

    # 展平成一维
    anomaly_score_flat = anomaly_score_2d.flatten()
    labels_flat = labels_combined.flatten()
    n = len(labels_flat)

    # Recall@5%、Recall@10%
    idx_5 = np.argsort(anomaly_score_flat)[-int(n*0.05):]
    idx_10 = np.argsort(anomaly_score_flat)[-int(n*0.10):]
    recall_5 = labels_flat[idx_5].sum() / labels_flat.sum() if labels_flat.sum() > 0 else 0
    recall_10 = labels_flat[idx_10].sum() / labels_flat.sum() if labels_flat.sum() > 0 else 0

    # ROC-AUC
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(labels_flat, anomaly_score_flat)
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
    # 动态特征降维到2维
    dynamic_2d = dynamic_proj(dynamic_features)  # [T, N, 2]
    
    # 静态特征降维到2维
    static_2d = static_proj(static_features)     # [T, N, 2]
    
    # 门控融合
    gate = torch.sigmoid(dynamic_2d + static_2d)
    fused_feat = gate * dynamic_2d + (1 - gate) * static_2d
    
    # 计算MAE损失
    if original_data.shape[0] == dynamic_2d.shape[1]:  # 如果原始数据是[N, T, 2]
        original_data = original_data.permute(1, 0, 2)  # 转为[T, N, 2]
    
    recon_loss = torch.abs(fused_feat - original_data).mean()
    return recon_loss

# --- 辅助函数：双流对比损失计算 ---
def compute_dual_stream_loss(dynamic_scores, static_scores):
    # 归一化分数
    dynamic_scores = dynamic_scores.to(static_scores.device)
    dynamic_scores_norm = torch.softmax(dynamic_scores, dim=-1)
    static_scores_norm = torch.softmax(static_scores, dim=-1)
    ds_loss = sym_kl_loss(dynamic_scores_norm, static_scores_norm).mean()
    return ds_loss

def safe_item(x):
    if isinstance(x, torch.Tensor):
        return x.item()
    return float(x)

if __name__ == '__main__':
    main_training_workflow() 