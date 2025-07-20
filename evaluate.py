import torch
import numpy as np
from scipy.stats import multivariate_normal
from main_workflow import DSCLModel, TemporalFrequencyMasking
import os

def evaluate():
    device = torch.device('cpu')
    device_str = 'cpu'
    d_model = 64
    n_heads = 4

    # 加载测试集数据和标签
    test_data = np.load('data/datanew1/normalized_test_injected.npy')
    test_labels = np.load('data/datanew1/anomaly_labels_test_injected.npy')
    active_zones = np.load('data/datanew1/active_zones.npy')
    adj_dist = np.load('data/processed/dist.npy')
    adj_corr = np.load('data/processed/adj.npy')
    adj_poi = np.load('data/processed/poi_sim.npy')
    adj_dist = adj_dist[np.ix_(active_zones, active_zones)]
    adj_corr = adj_corr[np.ix_(active_zones, active_zones)]
    adj_poi = adj_poi[np.ix_(active_zones, active_zones)]
    static_adj_matrices = torch.stack([
        torch.from_numpy(adj_dist).float(),
        torch.from_numpy(adj_corr).float(),
        torch.from_numpy(adj_poi).float()
    ]).to(device)

    test_data = torch.from_numpy(test_data).float()
    n_nodes = test_data.shape[1]
    n_features = test_data.shape[2]

    # 初始化模型和masker
    model = DSCLModel(n_nodes, n_features, d_model, n_heads, static_adj_matrices, device_str)
    masker = TemporalFrequencyMasking(
        window_size=10,
        temporal_mask_ratio=0.1,
        frequency_mask_ratio=0.1,
        d_model=d_model,
        n_features=n_features,
        device=device_str
    )

    # 加载训练好的权重
    checkpoint = torch.load('dscl_trained_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    masker.load_state_dict(checkpoint['masker_state_dict'])
    model.eval()
    masker.eval()

    with torch.no_grad():
        data_for_masking = test_data.permute(1, 0, 2)  # [N, T, F]
        temp_masked_data, _, freq_masked_data, _ = masker(data_for_masking)
        temp_masked_data = temp_masked_data.detach().cpu().numpy()
        freq_masked_data = freq_masked_data.detach().cpu().numpy()

        # 频域流
        freq_features_list = []
        for i in range(freq_masked_data.shape[0]):
            node_input = torch.FloatTensor(freq_masked_data[i:i+1]).to(device)
            feature_outputs = model.freq_encoder(node_input)
            freq_features_list.append(feature_outputs.detach())
        freq_features = torch.cat(freq_features_list, dim=0)  # [N, T, d_model]

        # 动态流
        batch_size = 8
        N, T, C = temp_masked_data.shape
        temp_features_list = []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_data = torch.FloatTensor(temp_masked_data[start:end]).to(device)
            output, _ = model.temporal_processor(batch_data)
            temp_features_list.append(output.detach().cpu().numpy())
        temp_features = np.concatenate(temp_features_list, axis=0)
        temp_features = torch.from_numpy(temp_features).float().to(device)

        # 静态流
        temp_features_for_gcn = torch.FloatTensor(temp_masked_data).to(device)
        temp_features_for_gcn = temp_features_for_gcn.permute(1, 0, 2)
        static_out, static_scores, static_reconstruction = model.static_gcn(temp_features_for_gcn)

        # 合并空间注意力分数和融合重构的遍历，只做一次空间注意力
        x = temp_features.permute(1, 0, 2)  # [T, N, C]
        x_proj = model.spatial_input_proj(x)  # [T, N, d_model]
        spatial_features_list = []
        spatial_attn_list = []
        for i in range(x_proj.shape[0]):
            out, attn = model.spatial_attention(x_proj[i:i+1])  # [1, N, d_model]
            spatial_features_list.append(out.squeeze(0))  # [N, d_model]
            spatial_attn_list.append(attn)
        spatial_features = torch.stack(spatial_features_list, dim=0)  # [T, N, d_model]
        spatial_attn_mean = torch.stack(spatial_attn_list, dim=0).mean(dim=0)  # [n_heads, N, N]
        dynamic_features_for_recon = spatial_features  # [T, N, d_model]
        dynamic_feat_2d = model.dynamic_proj(dynamic_features_for_recon)  # [T, N, 2]
        static_feat_2d = model.static_proj(static_out)
        gate = torch.sigmoid(dynamic_feat_2d + static_feat_2d)
        fused_feat = gate * dynamic_feat_2d + (1 - gate) * static_feat_2d
        raw_tensor = test_data
        if list(raw_tensor.shape) != list(fused_feat.shape):
            if raw_tensor.shape[0] == fused_feat.shape[1] and raw_tensor.shape[1] == fused_feat.shape[0]:
                raw_tensor = raw_tensor.permute(1, 0, 2)
        raw_tensor = raw_tensor.to(fused_feat.device)

        # === 按照理论公式计算异常分数 ===
        
        # 1. 计算重构损失 (L2范数平方)
        recon_loss = torch.square(fused_feat - raw_tensor).mean(dim=2)  # [T, N]
        recon_loss = recon_loss.T  # [N, T]
        
        # 2. 计算对称KL散度 (动态流和静态流的注意力分数)
        # 获取动态流注意力分数
        dynamic_attn = spatial_attn_mean  # [n_heads, N, N] 或 [N, N]
        if len(dynamic_attn.shape) == 3:
            dynamic_attn = dynamic_attn.mean(dim=0)  # [N, N]
        
        # 获取静态流注意力分数
        static_attn = static_scores  # [N, N]
        
        # 确保形状匹配
        if dynamic_attn.shape != static_attn.shape:
            min_size = min(dynamic_attn.shape[0], static_attn.shape[0])
            dynamic_attn = dynamic_attn[:min_size, :min_size]
            static_attn = static_attn[:min_size, :min_size]
        
        # 归一化注意力分数
        dynamic_attn_norm = torch.softmax(dynamic_attn.flatten(), dim=0).reshape(dynamic_attn.shape)
        static_attn_norm = torch.softmax(static_attn.flatten(), dim=0).reshape(static_attn.shape)
        
        # 计算对称KL散度
        def sym_kl_loss(p, q):
            kl_pq = torch.sum(p * (torch.log(p + 1e-8) - torch.log(q + 1e-8)))
            kl_qp = torch.sum(q * (torch.log(q + 1e-8) - torch.log(p + 1e-8)))
            return kl_pq + kl_qp
        
        sys_kl_loss = sym_kl_loss(dynamic_attn_norm, static_attn_norm)
        
        # 3. 计算时间异常分数 (时频对比损失)
        # 获取频域特征
        freq_feat = freq_features.mean(dim=0)  # [T, d_model]
        
        # 获取动态特征（和主流程一致，经过temporal_decoder+投影）
        # 首先将spatial_features投影到2维
        spatial_features_2d = model.spatial_proj_to_2(spatial_features)  # [T, N, 2]
        dynamic_features_list = []
        for i in range(spatial_features_2d.shape[1]):
            node_spatial_features = spatial_features_2d[:, i:i+1, :].to(device)  # [T, 1, 2]
            T = node_spatial_features.shape[0]
            normal_tokens = node_spatial_features.squeeze(1)  # [T, 2]
            mask_indices = torch.empty(0, dtype=torch.long, device=device)
            decoded_features = model.temporal_decoder(normal_tokens.unsqueeze(0), mask_indices.unsqueeze(0), T)
            decoded_features = decoded_features.squeeze(0)  # [T, 2]
            projected_features = model.temporal_proj_to_d_model(decoded_features)  # [T, d_model]
            dynamic_features_list.append(projected_features.unsqueeze(1))  # [T, 1, d_model]
        dynamic_features = torch.cat(dynamic_features_list, dim=1)  # [T, N, d_model]
        dynamic_feat = dynamic_features.mean(dim=1)  # [T, d_model]

        # 检查时间步数
        if freq_feat.shape[0] != dynamic_feat.shape[0]:
            raise ValueError(f"时频对比损失时间步数不一致！freq_feat.shape={freq_feat.shape}, dynamic_feat.shape={dynamic_feat.shape}")

        # 归一化特征
        freq_feat_norm = torch.softmax(freq_feat, dim=-1)
        dynamic_feat_norm = torch.softmax(dynamic_feat, dim=-1)

        # 计算每个时间步的KL散度
        def kl_divergence(p, q):
            return torch.sum(p * (torch.log(p + 1e-8) - torch.log(q + 1e-8)), dim=-1)  # [T]
        kl_p_to_f = kl_divergence(dynamic_feat_norm, freq_feat_norm)  # [T]
        kl_f_to_p = kl_divergence(freq_feat_norm, dynamic_feat_norm)  # [T]
        time_anomaly_score = kl_p_to_f + kl_f_to_p  # [T]
        
        # 4. 按照理论公式计算区域异常分数（每个区域每个时间点）
        beta = 0.01  # 控制对比损失相对重要性的超参数
        
        # 2. 区域对比损失项（每个区域一个标量，需要广播到时间维度）
        # 处理动态流注意力分数：从多头注意力转换为单头
        print(f"DEBUG: spatial_attn_mean.shape = {spatial_attn_mean.shape}")
        if len(spatial_attn_mean.shape) == 3:
            dynamic_attn = spatial_attn_mean.mean(dim=0)  # [N, N] - 动态流注意力分数
            print(f"DEBUG: After mean, dynamic_attn.shape = {dynamic_attn.shape}")
        elif len(spatial_attn_mean.shape) == 4:
            # 4维张量：[1, n_heads, N, N] -> [N, N]
            dynamic_attn = spatial_attn_mean.squeeze(0).mean(dim=0)  # [N, N] - 动态流注意力分数
            print(f"DEBUG: After squeeze and mean, dynamic_attn.shape = {dynamic_attn.shape}")
        else:
            dynamic_attn = spatial_attn_mean  # [N, N] - 动态流注意力分数
            print(f"DEBUG: No processing needed, dynamic_attn.shape = {dynamic_attn.shape}")
        static_attn = static_scores    # [N, N] - 静态流注意力分数
        print(f"DEBUG: static_attn.shape = {static_attn.shape}")
        
        dynamic_attn_norm = torch.softmax(dynamic_attn, dim=1)  # [N, N]
        static_attn_norm = torch.softmax(static_attn, dim=1)    # [N, N]
        print(f"DEBUG: dynamic_attn_norm.shape = {dynamic_attn_norm.shape}")
        print(f"DEBUG: static_attn_norm.shape = {static_attn_norm.shape}")
        
        def sym_kl_row(p, q):
            print(f"DEBUG: sym_kl_row input p.shape = {p.shape}")
            print(f"DEBUG: sym_kl_row input q.shape = {q.shape}")
            kl_pq = torch.sum(p * (torch.log(p + 1e-8) - torch.log(q + 1e-8)), dim=1)  # [N]
            kl_qp = torch.sum(q * (torch.log(q + 1e-8) - torch.log(p + 1e-8)), dim=1)  # [N]
            result = kl_pq + kl_qp  # [N]
            print(f"DEBUG: sym_kl_row result.shape = {result.shape}")
            return result
        sys_kl_loss_per_region = sym_kl_row(dynamic_attn_norm, static_attn_norm)  # [N]
        print(f"DEBUG: sys_kl_loss_per_region.shape = {sys_kl_loss_per_region.shape}")
        print(f"DEBUG: recon_loss.shape = {recon_loss.shape}")
        # 将区域对比损失广播到时间维度
        sys_kl_loss = sys_kl_loss_per_region.unsqueeze(1).expand(-1, recon_loss.shape[1])  # [N, T]
        
        # 3. 区域异常分数（每个区域每个时间点）
        spatial_anomaly_scores = recon_loss + beta * sys_kl_loss  # [N, T]
        # 5. 计算区域异常分数和时间异常分数
        # 区域异常分数：每个区域一个标量（所有时间步的平均）
        S_region = spatial_anomaly_scores.mean(dim=1).cpu().numpy()  # (N,) - 区域异常分数
        S_time = time_anomaly_score.cpu().numpy()  # (T,) - 时间异常分数

        # 6. 直接加权融合：构造（区域，时间）组合的异常分数
        N = S_region.shape[0]
        T = S_time.shape[0]
        print(f"\n=== 直接加权融合 ===")
        print(f"区域异常分数范围: [{S_region.min():.6f}, {S_region.max():.6f}]")
        print(f"时间异常分数范围: [{S_time.min():.6f}, {S_time.max():.6f}]")
        
        # 直接计算每个区域每个时间点的异常分数
        # 使用加权组合：alpha * 区域异常分数 + (1-alpha) * 时间异常分数
        alpha = 0.2  # 区域权重，可以调整
        anomaly_score_2d = np.zeros((N, T))
        
        for n in range(N):
            for t in range(T):
                # 直接加权组合
                anomaly_score_2d[n, t] = alpha * S_region[n] + (1 - alpha) * S_time[t]
        
        print(f"加权融合异常分数范围: [{anomaly_score_2d.min():.6f}, {anomaly_score_2d.max():.6f}]")
        print(f"权重设置: 区域权重={alpha}, 时间权重={1-alpha}")
        
        # 展平为一维用于后续评估
        anomaly_score = anomaly_score_2d.flatten()

    # 读取标签
    labels = test_labels  # (T, N) - 现在标签形状是 (T, N)，不考虑特征维度
    # 转置为(N, T)以匹配异常分数的形状
    labels_combined = labels.T  # (N, T)

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

    print("\n=== 测试集异常检测评估结果 ===")
    print(f"Recall@5%:  {recall_5:.4f}")
    print(f"Recall@10%: {recall_10:.4f}")
    print(f"ROC-AUC:    {auc:.4f}")
    
    # 打印调试信息
    print(f"\n=== 调试信息 ===")
    print(f"重构损失范围: [{recon_loss.min().item():.6f}, {recon_loss.max().item():.6f}]")
    print(f"对称KL散度: {sys_kl_loss.mean().item():.6f}")
    print(f"时间异常分数: {time_anomaly_score.mean().item():.6f}")
    print(f"空间异常分数范围: [{spatial_anomaly_scores.min().item():.6f}, {spatial_anomaly_scores.max().item():.6f}]")
    print(f"最终异常分数范围: [{anomaly_score_2d.min():.6f}, {anomaly_score_2d.max():.6f}]")
    print(f"原始T: {test_data.shape[0]}")
    print(f"temp_masked_data.shape: {temp_masked_data.shape}")
    print(f"freq_features.shape: {freq_features.shape}")
    print(f"dynamic_features_for_recon.shape: {dynamic_features_for_recon.shape}")
    print(f"static_out.shape: {static_out.shape}")
    print(f"spatial_attn_mean.shape: {spatial_attn_mean.shape}")
    print(f"recon_loss.shape: {recon_loss.shape}")
    print(f"spatial_anomaly_scores.shape: {spatial_anomaly_scores.shape}")
    print(f"S_region.shape: {S_region.shape}")
    print(f"S_time.shape: {S_time.shape}")
    print(f"N: {N}, T: {T}")
    print(f"time_anomaly_score.shape: {time_anomaly_score.shape}")

    # === 区域分数和时间分数单独AUC评估 ===
    from sklearn.metrics import roc_auc_score
    # 区域分数AUC（每个区域每个时间点）
    region_scores_flat = spatial_anomaly_scores.flatten()
    # 使用原始标签形状(T, N)，转置为(N, T)以匹配评估逻辑
    labels_for_eval = test_labels.transpose(1, 0)  # (N, T)
    region_labels_flat = labels_for_eval.flatten()
    try:
        region_auc = roc_auc_score(region_labels_flat, region_scores_flat)
    except Exception as e:
        print(f"区域AUC计算错误: {e}")
        region_auc = float('nan')

    # 时间分数AUC（每个区域每个时间点）
    time_scores_flat = np.full_like(labels_for_eval, time_anomaly_score.cpu().numpy()).flatten()
    time_labels_flat = labels_for_eval.flatten()
    try:
        time_auc = roc_auc_score(time_labels_flat, time_scores_flat)
    except Exception as e:
        print(f"时间AUC计算错误: {e}")
        time_auc = float('nan')

    print(f"区域分数AUC: {region_auc:.4f}")
    print(f"时间分数AUC: {time_auc:.4f}")

if __name__ == '__main__':
    evaluate() 