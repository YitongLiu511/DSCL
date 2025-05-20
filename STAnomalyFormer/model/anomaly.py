import torch
from torch import nn
import numpy as np
from sklearn.metrics import roc_auc_score

def kl_loss(p, q):
    res = p * (torch.log(p + 1e-8) - torch.log(q + 1e-8))
    return res  # 不取平均，保持维度信息

def recall_k(y_true, y_score, k):
    """自定义recall@k函数，计算y_true中为1的样本在y_score排序后前k个样本中出现的比例。"""
    if k <= 0:
        raise ValueError("k必须为正整数")
    # 将y_true、y_score转为numpy数组，并确保一维
    y_true = np.asarray(y_true).flatten()
    y_score = np.asarray(y_score).flatten()
    if len(y_true) != len(y_score):
        raise ValueError("y_true与y_score长度不一致")
    # 按y_score降序排序，取前k个索引
    sort_index = np.argsort(y_score)[::-1][:k]
    # 计算前k个样本中，y_true为1的样本数，并除以y_true中1的总数（若总数为0则返回0）
    n_anomaly = np.sum(y_true == 1)
    if n_anomaly == 0:
        return 0.0
    return np.sum(y_true[sort_index] == 1) / n_anomaly

from .embed import TemporalEmbedding
from .module import (
    SingleGCN,
    MultipleGCN,
    MultiheadAttention,
)
from .revin import RevIN
from .mask import TemporalMask, FrequencyMask
from .contrastive import ContrastiveLoss, AdversarialTraining


class STAnomalyFormer_v1(nn.Module):

    def __init__(
        self,
        dist_mat,
        d_in: int,
        d_model: int,
        dim_k: int,
        dim_v: int,
        n_heads: int,
        n_gcn: int,
        batch_size: int,
    ) -> None:
        super().__init__()
        self.embed = TemporalEmbedding(d_in, d_model)
        self.temporal_attn = MultiheadAttention(
            d_model,
            dim_k,
            dim_v,
            n_heads,
            batch_size=batch_size,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

        self.spatial_attn = MultiheadAttention(d_model, dim_k, dim_v, n_heads)
        self.norm2 = nn.LayerNorm(d_model)

        self.gcn = SingleGCN(
            d_model,
            d_model,
            dist_mat=dist_mat,
            n_layers=n_gcn,
        )

        self.proj1 = nn.Linear(d_model, d_in)
        self.proj2 = nn.Linear(d_model, d_in)

    def forward(self, x, return_dict=False):
        # temporal part
        x = self.embed(x)
        # x = self.auto_corr(x, x)[0]
        x, attn1 = self.temporal_attn(x, x)
        x = self.norm1(self.dropout(x))

        x = x.swapaxes(0, 1)
        # dynamic spatial part
        output_dy, score_dy = self.spatial_attn(x, x)
        output_dy = self.norm2(self.dropout(x + output_dy))

        # static spatial part
        output_st, score_st = self.gcn(x)

        # fusion
        output_dy, output_st = self.proj1(output_dy), self.proj2(output_st)
        g = torch.sigmoid(output_dy + output_st)
        output = g * output_dy + (1 - g) * output_st
        output = output.swapaxes(0, 1)
        score_dy = score_dy.mean(0)
        score_st = score_st.mean(0)

        # 计算区域异常分数（空间维度）
        region_scores = score_st.mean(dim=1)  # 对每个区域的所有时间点取平均
        # 计算时间异常分数（时间维度）
        time_scores = score_dy.mean(dim=0)  # 对每个时间点的所有区域取平均

        if return_dict:
            return {
                'reconstruction': (x, output),
                'attention': (attn1.mean(0), score_st),
                'region_scores': region_scores,
                'time_scores': time_scores,
                'score_dy': score_dy,
                'score_st': score_st
            }
        else:
            return output, score_dy, score_st


class STAnomalyFormer_v2(STAnomalyFormer_v1):

    def __init__(
        self,
        dist_mat,
        d_in: int,
        d_model: int,
        dim_k: int,
        dim_v: int,
        n_heads: int,
        n_gcn: int,
        batch_size: int,
    ) -> None:
        super().__init__(dist_mat, d_in, d_model, dim_k, dim_v, n_heads, n_gcn,
                         batch_size)
        self.embed_gcn = SingleGCN(
            d_in,
            d_model,
            dist_mat,
            n_layers=1,
            activation=lambda x: x,
        )
        self.revin = RevIN(d_in)

    def forward(self, x, return_dict=False):
        x = self.revin(x, 'norm')
        x1 = self.embed(x)
        branch1 = x1
        branch2 = x1 + self.embed_gcn(x.swapaxes(0, 1))[0].swapaxes(0, 1)

        branch1_, attn1 = self.temporal_attn(branch1, branch1)
        branch2_, attn2 = self.temporal_attn(branch2, branch2)

        branch1_ = self.norm1(self.dropout(branch1 + branch1_))
        branch2_ = self.norm1(self.dropout(branch2 + branch2_))

        branch1_ = branch1_.swapaxes(0, 1)
        branch2_ = branch2_.swapaxes(0, 1)

        output_dy, score_dy = self.spatial_attn(branch1_, branch1_)
        output_dy = self.norm2(self.dropout(branch1_ + output_dy))

        output_st, score_st = self.gcn(branch2_)

        output_dy, output_st = self.proj1(output_dy), self.proj2(output_st)
        g = torch.sigmoid(output_dy + output_st)
        output = g * output_dy + (1 - g) * output_st
        output = output.swapaxes(0, 1)
        score_dy = score_dy.mean(0)

        if return_dict:
            # 计算区域异常分数（空间维度）
            region_scores = score_st.mean(dim=1)  # 对每个区域的所有时间点取平均
            
            # 计算时间异常分数（时间维度）
            time_scores = score_dy.mean(dim=0)  # 对每个时间点的所有区域取平均
            
            return {
                'reconstruction': self.revin(output, 'denorm'),
                'attention': (attn1.mean(0), attn2.mean(0)),
                'region_scores': region_scores,
                'time_scores': time_scores,
                'score_dy': score_dy,
                'score_st': score_st
            }
        else:
            return (
                self.revin(output, 'denorm'),
                attn1.mean(0),
                attn2.mean(0),
                score_dy,
                score_st,
            )


from .patch import Patch, PatchEncoder, RandomMasking
from .tsfm import TemporalTransformer


class STPatchFormer(nn.Module):

    def __init__(
        self,
        seq_len: int,
        patch_len: int,
        stride: int,
        d_in: int,
        d_model: int,
        n_heads: int,
        dist_mat,
        n_gcn: int = 3,
        temporal_half: bool = False,
        spatial_half: bool = False,
        static_only: bool = False,
        dynamic_only: bool = False,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_in = d_in
        self.d_model = d_model
        self.n_heads = n_heads
        self.temporal_half = temporal_half
        self.spatial_half = spatial_half
        self.n_gcn = n_gcn
        self.static_only = static_only
        self.dynamic_only = dynamic_only

        self.revin = RevIN(d_in)
        self.patch = Patch(seq_len, patch_len, stride)
        self.patch_tsfm = PatchEncoder(
            d_in,
            self.patch.num_patch,
            patch_len,
            d_model,
            n_heads,
            256,
            False,
            0.1,
            0.1,
            temporal_half,
        )
        self.spatial_tsfm = TemporalTransformer(
            d_model,
            d_model // n_heads,
            d_model // n_heads,
            n_heads,
            d_model,
            0.1,
            spatial_half,
            True,
        )
        self.da_gcn = SingleGCN(
            d_model,
            d_model,
            dist_mat,
            n_layers=n_gcn,
        )
        self.proj_dy = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(d_model, patch_len),
        )
        self.proj_st = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(d_model, patch_len),
        )

    def forward(self, x, return_recon=True):
        # x : (N, T, d)
        patch_x = self.patch(x)  # (N, NP, VAR, PL)
        x = self.revin(patch_x.transpose(2, 3), 'norm').transpose(2, 3)
        z = self.patch_tsfm(x)  # (N, VAR, NP, D)
        z = z.permute(1, 2, 0, 3)  # (VAR, NP, N, D)
        z = z.reshape(-1, z.shape[2], z.shape[3])  # (VAR * NP, N, D)

        dy_z, attn = self.spatial_tsfm(z)  # (VAR * NP, N, D)
        st_z, graph = self.da_gcn(z)  # (VAR * NP, N, D)
        if not return_recon:
            return attn, graph

        if self.dynamic_only:
            z = self.proj_dy(dy_z)  # (VAR * NP, N, PL)
        elif self.static_only:
            z = self.proj_st(st_z)
        else:
            dy_out = self.proj_dy(dy_z)  # (VAR * NP, N, PL)
            st_out = self.proj_st(st_z)  # (VAR * NP, N, PL)
            g = torch.sigmoid(dy_out + st_out)
            z = g * dy_out + (1 - g) * st_out
        z = z.reshape(
            self.d_in,
            -1,
            x.shape[0],
            self.patch_len,  # (VAR, NP, N, PL)
        ).permute(2, 1, 3, 0)  # (N, NP, PL, VAR)
        z = self.revin(z, 'denorm')
        return (patch_x.transpose(2, 3), z), (attn, graph)


class STPatchMaskFormer(STPatchFormer):

    def __init__(
        self,
        seq_len: int,
        patch_len: int,
        stride: int,
        d_in: int,
        d_model: int,
        n_heads: int,
        dist_mat,
        n_gcn: int = 3,
        temporal_half: bool = False,
        spatial_half: bool = False,
        static_only: bool = False,
        dynamic_only: bool = False,
        mask_ratio: float = 0.4,
        temperature: float = 50.0,
        anormly_ratio: float = 0.1,
    ) -> None:
        super().__init__(seq_len, patch_len, stride, d_in, d_model, n_heads,
                         dist_mat, n_gcn, temporal_half, spatial_half,
                         static_only, dynamic_only)
        self.mask_ratio = mask_ratio
        self.random_mask = RandomMasking(self.mask_ratio)
        self.temperature = temperature
        self.anormly_ratio = anormly_ratio

    def forward(self, x):
        patch_x = self.patch(x)  # (N, NP, VAR, PL)
        x = self.revin(patch_x.transpose(2, 3), 'norm').transpose(2, 3)
        mask_x = self.random_mask(x)
        z = self.patch_tsfm(mask_x)  # (N, VAR, NP, D)
        z = z.permute(1, 2, 0, 3)  # (VAR, NP, N, D)
        z = z.reshape(-1, z.shape[2], z.shape[3])  # (VAR * NP, N, D)

        dy_z, attn = self.spatial_tsfm(z)  # (VAR * NP, N, D)
        st_z, graph = self.da_gcn(z)  # (VAR * NP, N, D)

        dy_out = self.proj_dy(dy_z)  # (VAR * NP, N, PL)
        st_out = self.proj_st(st_z)  # (VAR * NP, N, PL)
        g = torch.sigmoid(dy_out + st_out)
        z = g * dy_out + (1 - g) * st_out
        z = z.reshape(
            self.d_in,
            -1,
            x.shape[0],
            self.patch_len,  # (VAR, NP, N, PL)
        ).permute(2, 1, 3, 0)  # (N, NP, PL, VAR)
        z = self.revin(z, 'denorm')

        # Calculate anomaly scores
        adv_loss = 0.0
        con_loss = 0.0
        
        # 计算KL散度损失
        for i in range(attn.shape[0]):
            if i == 0:
                adv_loss = my_kl_loss(attn[i], 
                    (graph[i] / torch.unsqueeze(torch.sum(graph[i], dim=-1), dim=-1)).detach()) * self.temperature
                con_loss = my_kl_loss(
                    (graph[i] / torch.unsqueeze(torch.sum(graph[i], dim=-1), dim=-1)),
                    attn[i].detach()) * self.temperature
            else:
                adv_loss += my_kl_loss(attn[i], 
                    (graph[i] / torch.unsqueeze(torch.sum(graph[i], dim=-1), dim=-1)).detach()) * self.temperature
                con_loss += my_kl_loss(
                    (graph[i] / torch.unsqueeze(torch.sum(graph[i], dim=-1), dim=-1)),
                    attn[i].detach()) * self.temperature
        
        # 计算异常分数
        metric = torch.softmax((adv_loss + con_loss), dim=-1)
        anomaly_scores = metric.detach().cpu().numpy()
        
        # 计算阈值
        thresh = np.percentile(anomaly_scores, 100 - self.anormly_ratio)
        
        # 检测异常
        pred = (anomaly_scores > thresh).astype(int)
        
        # 获取异常区域和时间戳
        anomaly_regions = []
        anomaly_timestamps = []
        
        anomaly_state = False
        for i in range(len(pred)):
            if pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                start_idx = i
            elif pred[i] == 0 and anomaly_state:
                anomaly_state = False
                end_idx = i - 1
                anomaly_regions.append((start_idx, end_idx))
                # 转换索引为时间戳
                start_time = start_idx * self.stride
                end_time = end_idx * self.stride + self.patch_len
                anomaly_timestamps.append((start_time, end_time))
        
        if anomaly_state:
            end_idx = len(pred) - 1
            anomaly_regions.append((start_idx, end_idx))
            start_time = start_idx * self.stride
            end_time = end_idx * self.stride + self.patch_len
            anomaly_timestamps.append((start_time, end_time))

        return {
            'reconstruction': (patch_x.transpose(2, 3), z),
            'attention': (attn, graph),
            'anomaly_scores': anomaly_scores,
            'anomaly_regions': anomaly_regions,
            'anomaly_timestamps': anomaly_timestamps,
            'threshold': thresh
        }


class STPatch_MGCNFormer(nn.Module):

    def __init__(
        self,
        seq_len: int,
        patch_len: int,
        stride: int,
        d_in: int,
        d_model: int,
        n_heads: int,
        dist_mats,
        n_gcn: int = 3,
        temporal_half: bool = False,
        spatial_half: bool = False,
        static_only: bool = False,
        dynamic_only: bool = False,
        temperature: float = 50.0,
        anormly_ratio: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_in = d_in
        self.d_model = d_model
        self.n_heads = n_heads
        self.temporal_half = temporal_half
        self.spatial_half = spatial_half
        self.n_gcn = n_gcn
        self.static_only = static_only
        self.dynamic_only = dynamic_only
        self.temperature = temperature
        self.anormly_ratio = anormly_ratio

        # 基础组件
        self.revin = RevIN(d_in)
        self.patch = Patch(seq_len, patch_len, stride)
        
        # 编码器
        self.patch_encoder = PatchEncoder(
            d_in,
            self.patch.num_patch,
            patch_len,
            d_model,
            n_heads,
            256,
            False,
            0.1,
            0.1,
            temporal_half,
        )
        
        # 空间转换器
        self.spatial_transformer = TemporalTransformer(
            d_model,
            d_model // n_heads,
            d_model // n_heads,
            n_heads,
            d_model,
            0.1,
            spatial_half,
            True,
        )
        
        # 多视图GCN
        self.multi_gcn = MultipleGCN(
            d_model,
            d_model,
            dist_mats,
            n_layers=n_gcn,
        )
        
        # 动态和静态投影层
        self.proj_dynamic = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, patch_len)
        )
        
        self.proj_static = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, patch_len)
        )
        
        # 异常检测层
        self.anomaly_detector_dynamic = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
        self.anomaly_detector_static = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
        # 重构损失权重
        self.recon_weight = nn.Parameter(torch.ones(1))
        # 对比损失权重
        self.contrast_weight = nn.Parameter(torch.ones(1))

    def forward(self, x, return_dict=False):
        # 1. 数据预处理
        patch_x = self.patch(x)  # (N, NP, VAR, PL)
        x = self.revin(patch_x.transpose(2, 3), 'norm').transpose(2, 3)
        
        # 2. 特征提取
        z = self.patch_encoder(x)  # (N, VAR, NP, D)
        z = z.permute(1, 2, 0, 3)  # (VAR, NP, N, D)
        z = z.reshape(-1, z.shape[2], z.shape[3])  # (VAR * NP, N, D)

        # 3. 动态和静态特征
        dy_z, attn = self.spatial_transformer(z)  # (VAR * NP, N, D)
        st_z, graph = self.multi_gcn(z)  # (VAR * NP, N, D)

        # 4. 特征融合
        if self.dynamic_only:
            z = self.proj_dynamic(dy_z)
            anomaly_scores_dy = self.anomaly_detector_dynamic(dy_z)
            anomaly_scores_st = torch.zeros_like(anomaly_scores_dy)
        elif self.static_only:
            z = self.proj_static(st_z)
            anomaly_scores_st = self.anomaly_detector_static(st_z)
            anomaly_scores_dy = torch.zeros_like(anomaly_scores_st)
        else:
            dy_out = self.proj_dynamic(dy_z)
            st_out = self.proj_static(st_z)
            g = torch.sigmoid(dy_out + st_out)
            z = g * dy_out + (1 - g) * st_out

            # 计算动态和静态异常分数
            anomaly_scores_dy = self.anomaly_detector_dynamic(dy_z)
            anomaly_scores_st = self.anomaly_detector_static(st_z)
        
        # 5. 重构
        z = z.reshape(
            self.d_in,
            -1,
            x.shape[0],
            self.patch_len,
        ).permute(2, 1, 3, 0)
        z = self.revin(z, 'denorm')

        # 6. 计算重构损失
        recon_loss = torch.mean((patch_x.transpose(2, 3) - z) ** 2)
        
        # 7. 计算对比损失
        # 动态依赖矩阵
        dy_attn = attn.mean(0)  # (N, N)
        dy_attn = dy_attn.reshape(dy_attn.size(0), -1)  # 展平为2D
        dy_attn = torch.softmax(dy_attn / self.temperature, dim=-1)
        
        # 静态依赖矩阵
        st_attn = graph.mean(0)  # (N, N)
        st_attn = st_attn.reshape(st_attn.size(0), -1)  # 展平为2D
        st_attn = torch.softmax(st_attn / self.temperature, dim=-1)
        
        # 确保两个注意力矩阵具有相同的维度
        if dy_attn.size() != st_attn.size():
            min_size = min(dy_attn.size(0), st_attn.size(0))
            dy_attn = dy_attn[:min_size, :min_size]
            st_attn = st_attn[:min_size, :min_size]
        
        # 计算对称KL散度
        kl_loss_1 = torch.sum(dy_attn * (torch.log(dy_attn + 1e-8) - torch.log(st_attn + 1e-8)), dim=-1)
        kl_loss_2 = torch.sum(st_attn * (torch.log(st_attn + 1e-8) - torch.log(dy_attn + 1e-8)), dim=-1)
        contrast_loss = (kl_loss_1 + kl_loss_2) / 2
        
        # 8. 计算最终异常分数
        # 确保损失维度一致
        if isinstance(recon_loss, torch.Tensor):
            recon_loss = recon_loss.mean()
        if isinstance(contrast_loss, torch.Tensor):
            contrast_loss = contrast_loss.mean()
            
        # 计算时间异常分数（周期性拥堵）
        # 1. 计算每个区域的历史最大值
        max_flow = torch.max(x, dim=1)[0]  # (N, D)
        # 2. 计算当前流量与历史最大值的比例
        flow_ratio = x / (max_flow.unsqueeze(1) + 1e-6)  # (N, T, D)
        # 3. 计算时间异常分数
        time_scores = torch.mean(flow_ratio, dim=-1)  # (N, T)
        
        # 计算空间异常分数（跨区域异常关联）
        # 1. 计算区域间的流量差异
        region_flow = torch.mean(x, dim=1)  # (N, D)
        region_diff = torch.cdist(region_flow, region_flow)  # (N, N)
        
        # 2. 对每个区域，随机选择k个其他区域
        k = max(1, int(0.1 * (x.shape[0] - 1)))  # 确保k至少为1
        region_scores = torch.zeros(x.shape[0], device=x.device)  # (N,)
        
        for i in range(x.shape[0]):
            # 随机选择k个其他区域
            other_regions = list(range(x.shape[0]))
            other_regions.remove(i)
            selected_indices = torch.randperm(len(other_regions))[:k]
            selected_regions = [other_regions[idx] for idx in selected_indices]
            
            # 计算与选中区域的流量差异
            current_flow = region_flow[i]  # (D,)
            selected_flows = region_flow[selected_regions]  # (k, D)
            flow_diff = torch.norm(selected_flows - current_flow, dim=1)  # (k,)
            
            # 使用最大差异作为异常分数
            region_scores[i] = torch.max(flow_diff)
        
        # 调整维度以匹配标签维度 (263, 144, 14, 2)
        # 时间异常分数: (N, T) -> (N, T, 1)
        time_scores = time_scores.unsqueeze(-1)  # (N, T, 1)
        # 区域异常分数: (N,) -> (N, 1, 1)
        region_scores = region_scores.unsqueeze(-1).unsqueeze(-1)  # (N, 1, 1)
        
        # 计算阈值
        time_thresh = torch.quantile(time_scores, 1 - self.anormly_ratio)
        region_thresh = torch.quantile(region_scores, 1 - self.anormly_ratio)
        
        # 标记异常
        time_anomalies = (time_scores > time_thresh).float()  # (N, T, 1)
        region_anomalies = (region_scores > region_thresh).float()  # (N, 1, 1)
        
        # 确保分数不为0且数值稳定
        time_scores = torch.clamp(time_scores + 1e-6, min=1e-6, max=1e6)
        region_scores = torch.clamp(region_scores + 1e-6, min=1e-6, max=1e6)
        
        # 打印调试信息
        print(f"[DEBUG] 时间异常分数形状: {time_scores.shape}")
        print(f"[DEBUG] 时间异常分数范围: [{time_scores.min().item():.4f}, {time_scores.max().item():.4f}]")
        print(f"[DEBUG] 区域异常分数形状: {region_scores.shape}")
        print(f"[DEBUG] 区域异常分数范围: [{region_scores.min().item():.4f}, {region_scores.max().item():.4f}]")
        
        if return_dict:
            return {
                'reconstruction': (patch_x.transpose(2, 3), z),
                'attention': (dy_attn, st_attn),
                'time_scores': time_scores,  # (N, T, 1)
                'region_scores': region_scores,  # (N, 1, 1)
                'time_anomalies': time_anomalies,  # (N, T, 1)
                'region_anomalies': region_anomalies,  # (N, 1, 1)
                'dynamic_scores': anomaly_scores_dy,
                'static_scores': anomaly_scores_st,
                'recon_loss': recon_loss,
                'contrast_loss': contrast_loss
            }
        else:
            return z, time_scores, region_scores